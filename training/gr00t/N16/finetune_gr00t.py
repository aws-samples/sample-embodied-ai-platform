#!/usr/bin/env python3
"""
Workflow script for fine-tuning GR00T N1.6 models with configurable parameters.
This script orchestrates the model-specific portion of the workflow:
1. Validate dataset path prepared by the entrypoint shell script
2. Build FinetuneConfig from environment variables
3. Run training via N1.6's experiment.run API
"""

import os
import sys
import logging
import json
import importlib.util
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import torch
from torch.distributed.run import main as torchrun

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run
from gr00t.data.embodiment_tags import EmbodimentTag

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/workspace/finetune_gr00t.log"),
    ],
)
logger = logging.getLogger(__name__)


class FinetuneWorkflow:
    """Main workflow class for fine-tuning GR00T N1.6 models."""

    def __init__(self):
        """Initialize the workflow with environment variables."""
        # Dataset directory
        self.dataset_local_dir = os.getenv("DATASET_LOCAL_DIR")

        # Output directories (prefer EFS paths if provided by the Batch Job Definition)
        self.output_dir = os.getenv("OUTPUT_DIR", "/workspace/checkpoints")

        # Training parameters
        self.max_steps = int(os.getenv("MAX_STEPS", "6000"))
        self.save_steps = int(os.getenv("SAVE_STEPS", "2000"))
        self.num_gpus = int(os.getenv("NUM_GPUS", "1"))
        self.base_model_path = os.getenv("BASE_MODEL_PATH", "nvidia/GR00T-N1.6-3B")
        self.embodiment_tag = os.getenv("EMBODIMENT_TAG", "new_embodiment")
        self.modality_config_path = os.getenv(
            "MODALITY_CONFIG_PATH", "/workspace/scripts/so101_modality_config.py"
        )
        self.report_to = os.getenv("REPORT_TO", "wandb")

        # Batch size: N1.6 uses GLOBAL_BATCH_SIZE; support BATCH_SIZE as alias
        self.global_batch_size = int(
            os.getenv("GLOBAL_BATCH_SIZE", os.getenv("BATCH_SIZE", "32"))
        )

        # Optimizer parameters
        self.learning_rate = float(os.getenv("LEARNING_RATE", "1e-4"))
        self.weight_decay = float(os.getenv("WEIGHT_DECAY", "1e-5"))
        self.warmup_ratio = float(os.getenv("WARMUP_RATIO", "0.05"))
        self.gradient_accumulation_steps = int(
            os.getenv("GRADIENT_ACCUMULATION_STEPS", "1")
        )

        # Dataloader
        self.dataloader_num_workers = int(os.getenv("DATALOADER_NUM_WORKERS", "4"))

        # Tuning flags
        self.tune_llm = os.getenv("TUNE_LLM", "false").lower() == "true"
        self.tune_visual = os.getenv("TUNE_VISUAL", "false").lower() == "true"
        self.tune_projector = os.getenv("TUNE_PROJECTOR", "true").lower() == "true"
        self.tune_diffusion_model = (
            os.getenv("TUNE_DIFFUSION_MODEL", "true").lower() == "true"
        )

        # Optional N1.6 parameters
        self.state_dropout_prob = float(os.getenv("STATE_DROPOUT_PROB", "0.0"))
        self.resume = os.getenv("RESUME", "false").lower() == "true"

        # Validate required parameters
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate required environment variables."""
        if not self.dataset_local_dir:
            raise ValueError("Missing required environment variable: DATASET_LOCAL_DIR")
        logger.info("All required parameters validated successfully")

    def validate_dataset(self):
        """
        Ensure a local dataset directory is ready.
        Patches parquet files and creates modality.json for N1.6 compatibility.
        """
        logger.info("Validating dataset...")

        if not os.path.isdir(self.dataset_local_dir) or not os.listdir(
            self.dataset_local_dir
        ):
            raise RuntimeError(
                "Dataset directory not prepared. Ensure entrypoint script resolved and downloaded the dataset."
            )

        logger.info(f"Using dataset directory: {self.dataset_local_dir}")

        # Create or patch modality.json to ensure annotation key is present (N1.6 requirement)
        meta_dir = os.path.join(self.dataset_local_dir, "meta")
        modality_json_path = os.path.join(meta_dir, "modality.json")
        annotation_key = "human.task_description"
        if not os.path.isfile(modality_json_path):
            os.makedirs(meta_dir, exist_ok=True)
            modality_content = {
                "state": {
                    "single_arm": {"start": 0, "end": 5},
                    "gripper": {"start": 5, "end": 6},
                },
                "action": {
                    "single_arm": {"start": 0, "end": 5},
                    "gripper": {"start": 5, "end": 6},
                },
                "video": {
                    "wrist": {"original_key": "observation.images.wrist"},
                    "front": {"original_key": "observation.images.front"},
                },
                "annotation": {
                    annotation_key: {"original_key": "task_index"}
                },
            }
            with open(modality_json_path, "w") as f:
                json.dump(modality_content, f, indent=4)
            logger.info(f"Created missing modality.json at {modality_json_path}")
        else:
            # Dataset already has modality.json — patch annotation key if absent
            with open(modality_json_path, "r") as f:
                modality_content = json.load(f)
            if annotation_key not in modality_content.get("annotation", {}):
                modality_content.setdefault("annotation", {})[annotation_key] = {
                    "original_key": "task_index"
                }
                with open(modality_json_path, "w") as f:
                    json.dump(modality_content, f, indent=4)
                logger.info(f"Patched modality.json: added annotation.{annotation_key}")

        # Patch parquet files: add annotation.human.task_description column
        # if missing (LeIsaac compatible key)
        self._patch_parquet_annotations()

    def _patch_parquet_annotations(self):
        """Add annotation.human.task_description column to parquet files if missing."""
        annotation_col = "annotation.human.task_description"
        data_dir = os.path.join(self.dataset_local_dir, "data")
        if not os.path.isdir(data_dir):
            # Try root-level parquet files
            data_dir = self.dataset_local_dir

        for parquet_file in Path(data_dir).rglob("*.parquet"):
            try:
                table = pq.read_table(str(parquet_file))
                if annotation_col not in table.column_names:
                    # Use task_index as fallback source
                    if "task_index" in table.column_names:
                        task_col = table.column("task_index")
                        # Convert integers to strings if needed
                        str_col = pa.array(
                            [str(v.as_py()) for v in task_col], type=pa.string()
                        )
                        table = table.append_column(annotation_col, str_col)
                        pq.write_table(table, str(parquet_file))
                        logger.info(
                            f"Patched {parquet_file.name}: added {annotation_col} from task_index"
                        )
                    else:
                        logger.warning(
                            f"{parquet_file.name}: no task_index column to copy from"
                        )
            except Exception as e:
                logger.warning(f"Could not patch {parquet_file}: {e}")

    def _load_modality_config(self):
        """Load modality config from the specified Python file."""
        config_path = self.modality_config_path
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Modality config not found: {config_path}")

        spec = importlib.util.spec_from_file_location("modality_config", config_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        logger.info(f"Loaded modality config from {config_path}")

    def build_finetune_config(self):
        """Build FinetuneConfig from environment variables."""
        config = FinetuneConfig(
            base_model_path=self.base_model_path,
            dataset_path=self.dataset_local_dir,
            output_dir=self.output_dir,
            embodiment_tag=self.embodiment_tag,
            modality_config_path=self.modality_config_path,
            global_batch_size=self.global_batch_size,
            max_steps=self.max_steps,
            save_steps=self.save_steps,
            num_gpus=self.num_gpus,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            dataloader_num_workers=self.dataloader_num_workers,
            tune_llm=self.tune_llm,
            tune_visual=self.tune_visual,
            tune_projector=self.tune_projector,
            tune_diffusion_model=self.tune_diffusion_model,
            use_wandb=(self.report_to == "wandb"),
        )
        return config

    def _train_once(self):
        """Run the fine-tuning via N1.6's experiment.run API."""
        logger.info("Starting training...")

        # Load modality config (registers via register_modality_config)
        self._load_modality_config()

        # Build FinetuneConfig
        ft = self.build_finetune_config()
        embodiment_tag = ft.embodiment_tag.lower() if isinstance(ft.embodiment_tag, str) else ft.embodiment_tag.value

        # Build base experiment Config following launch_finetune.py pattern
        config = get_default_config().load_dict(
            {
                "data": {
                    "download_cache": False,
                    "datasets": [
                        {
                            "dataset_paths": [ft.dataset_path],
                            "mix_ratio": 1.0,
                            "embodiment_tag": embodiment_tag,
                        }
                    ],
                }
            }
        )
        config.load_config_path = None

        # Model tuning flags
        config.model.tune_llm = ft.tune_llm
        config.model.tune_visual = ft.tune_visual
        config.model.tune_projector = ft.tune_projector
        config.model.tune_diffusion_model = ft.tune_diffusion_model
        config.model.state_dropout_prob = ft.state_dropout_prob
        config.model.random_rotation_angle = ft.random_rotation_angle
        config.model.color_jitter_params = ft.color_jitter_params
        config.model.load_bf16 = False
        config.model.reproject_vision = False
        config.model.eagle_collator = True
        config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
        config.model.backbone_trainable_params_fp32 = True
        config.model.use_relative_action = True

        # Training parameters
        config.training.start_from_checkpoint = ft.base_model_path
        config.training.optim = "adamw_torch"
        config.training.global_batch_size = ft.global_batch_size
        config.training.dataloader_num_workers = ft.dataloader_num_workers
        config.training.learning_rate = ft.learning_rate
        config.training.gradient_accumulation_steps = ft.gradient_accumulation_steps
        config.training.output_dir = ft.output_dir
        config.training.save_steps = ft.save_steps
        config.training.save_total_limit = ft.save_total_limit
        config.training.num_gpus = ft.num_gpus
        config.training.use_wandb = ft.use_wandb
        config.training.max_steps = ft.max_steps
        config.training.weight_decay = ft.weight_decay
        config.training.warmup_ratio = ft.warmup_ratio
        config.training.wandb_project = os.getenv("WANDB_PROJECT", "finetune-gr00t-n1d6")

        # Data parameters
        config.data.shard_size = ft.shard_size
        config.data.episode_sampling_rate = ft.episode_sampling_rate
        config.data.num_shards_per_epoch = ft.num_shards_per_epoch

        # Run experiment
        run(config)

        logger.info("Training completed successfully")

    def run_training(self):
        """Run the training with optional multi-GPU support (torchrun)."""
        os.makedirs(self.output_dir, exist_ok=True)

        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

        assert (
            self.num_gpus <= available_gpus
        ), f"Number of GPUs requested ({self.num_gpus}) is greater than the available GPUs ({available_gpus})"
        assert self.num_gpus > 0, "Number of GPUs must be greater than 0"
        print(f"Using {self.num_gpus} GPUs")

        if self.num_gpus == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            self._train_once()
        else:
            if os.environ.get("IS_TORCHRUN", "0") == "1":
                self._train_once()
            else:
                script_path = Path(__file__).absolute()
                if "CUDA_VISIBLE_DEVICES" in os.environ:
                    del os.environ["CUDA_VISIBLE_DEVICES"]

                args = [
                    "--standalone",
                    f"--nproc_per_node={self.num_gpus}",
                    "--nnodes=1",
                    str(script_path),
                ]

                print("Running torchrun with args: ", args)
                os.environ["IS_TORCHRUN"] = "1"
                try:
                    torchrun(args=args)
                except SystemExit as e:
                    code = e.code if isinstance(e.code, int) else 0
                    sys.exit(code)

    def run_workflow(self):
        """Run the complete workflow."""
        try:
            logger.info("Starting GR00T N1.6 fine-tuning...")

            # Step 1: Validate dataset was prepared by entrypoint script
            self.validate_dataset()

            # Step 2: Run training
            self.run_training()

            logger.info("GR00T N1.6 fine-tuning completed successfully!")

        except Exception as e:
            logger.error(f"Workflow failed: {str(e)}")
            sys.exit(1)


def main():
    """Main entry point."""
    workflow = FinetuneWorkflow()
    workflow.run_workflow()


if __name__ == "__main__":
    main()
