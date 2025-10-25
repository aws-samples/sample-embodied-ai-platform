# NVIDIA Isaac GR00T Training Component

Fine-tune NVIDIA Isaac GR00T VLA models using teleoperation/simulation datasets. Supports AWS Batch training with GPU and NICE DCV for monitoring/evaluation. Use this README as a bridge: high-level usage and structure here; detailed infrastructure/deployment in `infra/README.md`.

## Links

- Component docs (this): [README.md](README.md)
- Infrastructure and deployment: [infra/README.md](infra/README.md)
- Workflow scripts: [run_finetune_workflow.sh](run_finetune_workflow.sh), [finetune_gr00t.py](finetune_gr00t.py)

## Deployment

See [infra/README.md](infra/README.md).

## Module Structure

```text
training/gr00t/
├── README.md                  # GR00T training overview
├── Dockerfile                 # Training container
├── build_container.sh         # Build/test/push helper
├── env.example                # Example environment variables
├── finetune_gr00t.py          # GR00T training script
├── run_finetune_workflow.sh   # Entrypoint: dataset, auth, uploads
├── sample_dataset/            # Example data layout and samples
└── infra/                     # AWS CDK stacks for Batch and DCV
    ├── README.md              # Deployment guide (paths 1–3, troubleshooting)
    ├── app.py
    ├── batch_stack.py
    ├── dcv_stack.py
    ├── configure_dcv_instance.sh
    ├── requirements.txt
    ├── cdk.json               # Context (VPC/EFS/SG IDs) when importing existing resources
    └── architecture.drawio.png
```

## Configuration (env vars)

See [env.example](env.example) for configurable environment variables:
- Dataset sources: `DATASET_LOCAL_DIR`, `DATASET_S3_URI`, `HF_DATASET_ID`
- Uploads: `UPLOAD_TARGET` (hf|s3|none), `HF_TOKEN`, `HF_MODEL_REPO_ID`, `S3_UPLOAD_URI`
- Training: `MAX_STEPS`, `SAVE_STEPS`, `NUM_GPUS`, `BATCH_SIZE`, `LEARNING_RATE`
- Model/data: `BASE_MODEL_PATH`, `DATA_CONFIG`, `VIDEO_BACKEND`, `EMBODIMENT_TAG`
- Tuning: `TUNE_LLM`, `TUNE_VISUAL`, `TUNE_PROJECTOR`, `TUNE_DIFFUSION_MODEL`, LoRA params
