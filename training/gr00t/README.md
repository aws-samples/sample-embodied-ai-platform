# NVIDIA Isaac GR00T Training Component

Fine-tune NVIDIA Isaac GR00T VLA models using teleoperation/simulation datasets. Supports AWS Batch training with GPU and Amazon DCV for monitoring/evaluation. Use this README as a bridge: high-level usage and structure here; detailed infrastructure/deployment in `infra/README.md`.

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
    {name=SAVE_STEPS,value=2000}
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
      {"name":"BATCH_SIZE","value":"8"},
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
> **Multi-GPU Shared Memory**: When using multiple GPUs, you may need to reduce `DATALOADER_NUM_WORKERS` (from default of 8) to avoid shared memory exhaustion. In the provided [batch stack](infra/batch_stack.py), the job definition sets shared memory to 64GB, which is sufficient with reduced workers. Alternatively, you can set the shared memory size to a larger value that your selected instances can support in the job definition. For example with a g6e.12xlarge instance:
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

> Default: 6000 steps (~2 hours on g6e.4xlarge using the provided dataset). Checkpoints saved every 2000 steps at `/mnt/efs/gr00t/checkpoints`.

## Configuration (env vars)

See [env.example](env.example) for configuring the training job parameters:
- Dataset sources: `DATASET_LOCAL_DIR`, `DATASET_S3_URI`, `HF_DATASET_ID`
- Uploads: `UPLOAD_TARGET` (hf|s3|none), `HF_TOKEN`, `HF_MODEL_REPO_ID`, `S3_UPLOAD_URI`
- Training: `MAX_STEPS`, `SAVE_STEPS`, `NUM_GPUS`, `BATCH_SIZE`, `LEARNING_RATE`
- Model/data: `BASE_MODEL_PATH`, `DATA_CONFIG`, `VIDEO_BACKEND`, `EMBODIMENT_TAG`
- Tuning: `TUNE_LLM`, `TUNE_VISUAL`, `TUNE_PROJECTOR`, `TUNE_DIFFUSION_MODEL`, LoRA params
