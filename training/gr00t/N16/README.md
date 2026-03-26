# NVIDIA Isaac GR00T N1.6 Training Component

Fine-tune NVIDIA Isaac GR00T N1.6 VLA models using teleoperation/simulation datasets. Supports AWS Batch training with GPU and Amazon DCV for monitoring/evaluation. Use this README as a bridge: high-level usage and structure here; detailed infrastructure/deployment in `infra/README.md`.

## Links

- Component docs (this): [README.md](README.md)
- Infrastructure and deployment: [infra/README.md](infra/README.md)
- Workflow scripts: [run_finetune_workflow.sh](run_finetune_workflow.sh), [finetune_gr00t.py](finetune_gr00t.py), [so101_modality_config.py](so101_modality_config.py)

## Deployment

See [infra/README.md](infra/README.md).

## Module Structure

```text
training/gr00t/
├── README.md                  # GR00T training overview
├── Dockerfile                 # Training container (N1.6, NGC base)
├── build_container.sh         # Build/test/push helper
├── env.example                # Example environment variables
├── finetune_gr00t.py          # GR00T N1.6 training script
├── so101_modality_config.py   # SO-ARM101 modality config for N1.6
├── run_finetune_workflow.sh   # Entrypoint: dataset, auth, uploads
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

## Submitting Jobs

After deploying the infrastructure (see [infra/README.md](infra/README.md)), submit training jobs to AWS Batch:

**AWS CLI:**
```bash
aws batch submit-job \
  --job-name "IsaacGr00tFinetuning" \
  --job-queue "IsaacGr00tJobQueue" \
  --job-definition "IsaacGr00tJobDefinition"
```

**With custom environment variables:**
```bash
aws batch submit-job \
  --job-name "IsaacGr00tFinetuning" \
  --job-queue "IsaacGr00tJobQueue" \
  --job-definition "IsaacGr00tJobDefinition" \
  --container-overrides 'environment=[
    {name=HF_DATASET_ID,value=lerobot/your-dataset},
    {name=MAX_STEPS,value=6000},
    {name=SAVE_STEPS,value=2000},
    {name=MODALITY_CONFIG_PATH,value=/workspace/scripts/so101_modality_config.py}
  ]'
```

**Multi-GPU training (e.g. 4 GPUs with g6e.12xlarge):**
```bash
aws batch submit-job \
  --job-name "IsaacGr00tFinetuning" \
  --job-queue "IsaacGr00tJobQueue" \
  --job-definition "IsaacGr00tJobDefinition" \
  --container-overrides '{
    "environment": [
      {"name":"NUM_GPUS","value":"4"},
      {"name":"GLOBAL_BATCH_SIZE","value":"64"},
      {"name":"DATALOADER_NUM_WORKERS","value":"2"}
    ],
    "resourceRequirements": [
      {"type":"GPU","value":"4"},
      {"type":"VCPU","value":"48"},
      {"type":"MEMORY","value":"393216"}
    ]
  }'
```

> [!IMPORTANT]
> **Multi-GPU Shared Memory**: When using multiple GPUs, you may need to reduce `DATALOADER_NUM_WORKERS` (from default of 4) to avoid shared memory exhaustion. In the provided [batch stack](infra/batch_stack.py), the job definition sets shared memory to 64GB, which is sufficient with reduced workers. Alternatively, you can set the shared memory size to a larger value that your selected instances can support in the job definition. For example with a g6e.12xlarge instance:
> ```python
> ...
> linux_parameters=batch.LinuxParameters(
>     ...
>     shared_memory_size=Size.gibibytes(384),
>     ...
> )
> ...
> ```

**AWS Console:**
1. Go to AWS Batch → Jobs → Submit new job
2. Select `IsaacGr00tJobDefinition` and `IsaacGr00tJobQueue`
3. Add environment variables and select the number of GPUs you want to use as needed
4. Submit the job

> [!NOTE]
> If you use a custom dataset in [LerobotDataset:v3.0 format](https://huggingface.co/blog/lerobot-datasets-v3), you need to first convert it back to v2.1. LerobotDataset:v3.0 support is coming soon.

**Monitor progress:**
```bash
# Check status
aws batch describe-jobs --jobs <JOB_ID>

# Stream logs (once RUNNING)
aws logs tail /aws/batch/job --follow \
  --log-stream-names "$(aws batch describe-jobs --jobs <JOB_ID> \
  --query 'jobs[0].container.logStreamName' --output text)"
```

> Default: 6000 steps (~2 hours on g6e.4xlarge using the provided dataset). Checkpoints saved every 2000 steps at `/mnt/efs/gr00t/checkpoints/<JOB_ID>/` (each job gets its own subdirectory).

## Loss Curve Visualization (W&B Offline)

Training runs log to W&B in **offline mode** by default. Run data is written
to `WANDB_DIR` (defaults to `OUTPUT_DIR` on EFS), so it persists after the
Batch container exits. No server is needed during training.

### Viewing results after training

1. **Start the W&B local server** on the DCV instance:

       ssh ubuntu@<DCV_IP>
       docker run -d --name wandb-local -p 8080:8080 -v wandb-data:/vol wandb/local:latest

2. **Create a local account** at `http://<DCV_IP>:8080` and generate an API key
   from Settings → API Keys.

3. **Sync offline runs** from EFS into the local server:

       export WANDB_BASE_URL=http://localhost:8080
       export WANDB_API_KEY=<your-local-api-key>
       conda activate isaac
       wandb sync /mnt/efs/gr00t/checkpoints/<JOB_ID>/wandb/offline-run-*

4. **View loss curves** at `http://<DCV_IP>:8080` in your browser.

> **Tip:** The W&B server only needs to run when you want to view results.
> Stop it with `docker stop wandb-local` and restart anytime with
> `docker start wandb-local` — data is persisted in the `wandb-data` volume.

### Online mode (optional)

To log directly to a W&B server during training, override the defaults:

```bash
aws batch submit-job ... --container-overrides '{
  "environment": [
    {"name": "WANDB_MODE", "value": "online"},
    {"name": "WANDB_BASE_URL", "value": "http://<DCV_PRIVATE_IP>:8080"},
    {"name": "WANDB_API_KEY", "value": "<your-local-api-key>"}
  ]
}'
```

Use the DCV instance's **private IP** since Batch jobs run in the same VPC.

## Evaluation

### Open-loop evaluation

Run open-loop evaluation against a held-out dataset to compute action prediction
metrics (MSE, cosine similarity). This requires a checkpoint and a dataset with
`stats.json`:

```bash
# On DCV instance (inside the training container)
python -m gr00t.eval.robot_eval \
  --model-path /mnt/efs/gr00t/checkpoints/<JOB_ID>/checkpoint-6000 \
  --embodiment-tag new_embodiment \
  --dataset-path /path/to/eval-dataset \
  --modality-config-path /workspace/scripts/so101_modality_config.py
```

### Closed-loop evaluation (policy server)

Serve a trained checkpoint as a ZMQ policy server for real-time robot control.
On the DCV instance:

```bash
# Authenticate to ECR (instance profile provides credentials)
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-west-2.amazonaws.com

# Pull the training image (contains gr00t eval server)
docker pull <ACCOUNT_ID>.dkr.ecr.us-west-2.amazonaws.com/gr00t-finetune:latest

# Start the policy server
CHECKPOINT_PATH=/mnt/efs/gr00t/checkpoints/<JOB_ID>/checkpoint-6000
ECR_URI=<ACCOUNT_ID>.dkr.ecr.us-west-2.amazonaws.com/gr00t-finetune:latest
docker run --gpus all -d \
  --name gr00t-policy-server \
  --shm-size=8g \
  --network host \
  --entrypoint /bin/sh \
  -v "$CHECKPOINT_PATH:/workspace/checkpoint" \
  $ECR_URI \
  -c '/workspace/gr00t-repo/.venv/bin/python gr00t/eval/run_gr00t_server.py \
    --model_path /workspace/checkpoint \
    --embodiment_tag NEW_EMBODIMENT \
    --host 0.0.0.0'
```

> [!IMPORTANT]
> - Use `--entrypoint /bin/sh` — the NGC base image's `/usr/bin/bash` is broken.
> - The server CLI expects **uppercase** `NEW_EMBODIMENT` (tyro parses enum names, not values).
> - Pass `--host 0.0.0.0` to allow remote clients. The default is `127.0.0.1` (localhost only).

Ensure the DCV security group allows inbound TCP 5555 from the client IP. Clients send observations as msgpack-serialized numpy arrays over ZMQ REQ/REP. See the [SKILL.md](SKILL.md) evaluation section for the full observation/response format specification.

### Closed-loop evaluation with LeIsaac

[LeIsaac](https://github.com/LightwheelAI/leisaac) drives an IsaacSim environment and
feeds observations to the policy server in a closed loop. It is **not** installed by
default on the DCV instance — set it up when you need to run sim evaluations.

**One-time setup on the DCV instance:**

```bash
ssh dcv-isaac

# 1. Clone the leisaac repo (for evaluation scripts)
LEISAAC_COMMIT=d2cbfd2e33517f2094e1904ff817aa17de6e8939
git clone https://github.com/LightwheelAI/leisaac.git ~/leisaac-repo
cd ~/leisaac-repo && git checkout $LEISAAC_COMMIT

# 2. Install the leisaac Python package to the persistent IsaacLab package dir
mkdir -p ~/isaaclab-pkgs
docker run --rm --gpus all \
  -e ACCEPT_EULA=Y \
  -v ~/isaaclab-pkgs:/workspace/isaaclab-pkgs:rw \
  nvcr.io/nvidia/isaac-lab:2.3.0 \
  -c "/workspace/isaaclab/_isaac_sim/python.sh -m pip install \
    --target /workspace/isaaclab-pkgs \
    'leisaac[gr00t] @ git+https://github.com/LightwheelAI/leisaac.git@${LEISAAC_COMMIT}#subdirectory=source/leisaac'"

# 3. Update run-isaaclab.sh to mount the evaluation scripts
sudo sed -i '/--network=host/a\  -v $HOME/leisaac-repo/scripts:/workspace/scripts:ro \\' \
  /usr/local/bin/run-isaaclab.sh
```

> [!NOTE]
> The commit SHA must match the version pinned in `dcv/versions.py` for your IsaacSim
> version. `Gr00t16ServicePolicyClient` (N1.6) was added after the `v0.3.0` tag.

**Run the evaluation** (requires a DCV desktop session for the IsaacSim GUI):

```bash
# Launch the IsaacLab container from a DCV terminal
run-isaaclab.sh

# Inside the container:
/workspace/isaaclab/_isaac_sim/python.sh /workspace/scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --eval_rounds=10 \
    --policy_type=gr00tn1.6 \
    --policy_host=localhost \
    --policy_port=5555 \
    --policy_action_horizon=16 \
    --policy_language_instruction="Pick up the orange and place it on the plate" \
    --device=cuda \
    --enable_cameras
```

See [SKILL.md](SKILL.md) Phase 8a for detailed setup instructions, troubleshooting,
and the observation/response format reference.

## Configuration (env vars)

See [env.example](env.example) for configuring the training job parameters:
- Dataset sources: `DATASET_LOCAL_DIR`, `DATASET_S3_URI`, `HF_DATASET_ID`
- Uploads: `UPLOAD_TARGET` (hf|s3|none), `HF_TOKEN`, `HF_MODEL_REPO_ID`, `S3_UPLOAD_URI`
- Training: `MAX_STEPS`, `SAVE_STEPS`, `NUM_GPUS`, `GLOBAL_BATCH_SIZE`, `LEARNING_RATE`, `GRADIENT_ACCUMULATION_STEPS`
- Model/data: `BASE_MODEL_PATH`, `MODALITY_CONFIG_PATH`, `EMBODIMENT_TAG`
- Tuning: `TUNE_LLM`, `TUNE_VISUAL`, `TUNE_PROJECTOR`, `TUNE_DIFFUSION_MODEL`
