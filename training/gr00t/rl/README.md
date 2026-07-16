# GR00T RL Post-Training on AWS

Reinforcement learning post-training for NVIDIA GR00T N1.5 on the Assemble Trocar surgical task using PPO (via RLinf) with two compute backend options.

## Compute Backends

| Backend | Command | Instances | Best For |
|---------|---------|-----------|----------|
| **AWS Batch MNP** (homogeneous) | `--context compute_backend=batch-mnp` | 2× g6e.12xlarge (4× L40S each) | Simple setup, lower cost (~$16/hr) |
| **EKS + KubeRay** (heterogeneous) | `--context compute_backend=eks` | 1× g6e.48xlarge + 4× g6e.4xlarge (configurable) | Better GPU utilization, no RAM OOM |

Both backends are validated end-to-end: 2 PPO iterations + checkpoint saved.

## Quick Start

### Prerequisites

- AWS account with GPU quota (384 vCPUs for G instances in your region)
- CDK CLI installed (`npm install -g aws-cdk`)
- Python 3.10+ with CDK dependencies: `pip install -r infra/requirements.txt`
- **EKS backend**: VPC with private subnets that have NAT gateway egress (nodes must reach EKS API + ECR)

### Deploy

```bash
cd training/gr00t/rl/infra

# Option A: Batch MNP (homogeneous, simpler)
AWS_DEFAULT_REGION=us-east-2 cdk deploy --context compute_backend=batch-mnp

# Option B: EKS + KubeRay (heterogeneous, more scalable)
AWS_DEFAULT_REGION=us-east-2 CDK_DEFAULT_REGION=us-east-2 cdk deploy \
  --context compute_backend=eks \
  --context vpc_id=<your-vpc-id> \
  --context efs_id=<your-efs-id> \
  --context efs_sg_id=<efs-mount-target-sg> \
  --context image_uri=<ecr-image-uri> \
  --context learner_instance_type=g6e.48xlarge \
  --context rollout_instance_type=g6e.4xlarge
```

### Stage Training Data (EFS)

After deploying, trigger the CodeBuild project to stage code + model on EFS:

```bash
aws codebuild start-build --project-name GR00T-RL-Stage-EFS --region us-east-2
```

This stages:
- RLinf framework (pinned commit)
- Isaac-GR00T model code (commit `4af2b622`)
- IsaacLab + IsaacLab-Arena
- GR00T N1.5 pre-trained checkpoint
- Training workflows (rlinf_ext, configs, task definition)

### Run Training

**Batch MNP:**
```bash
aws batch submit-job \
  --job-name gr00t-rl-training \
  --job-queue GR00T-RL-JobQueue \
  --job-definition <job-definition-arn> \
  --region us-east-2
```

**EKS:** Training starts automatically when the RayCluster pods are created by CDK deploy. Monitor with:
```bash
# Use the role ARN from the KubeconfigCommand stack output
aws eks update-kubeconfig --name gr00t-rl-eks --region us-east-2 \
  --role-arn arn:aws:iam::<account>:role/gr00t-rl-eks-admin-<region>
kubectl get pods -n training
kubectl logs -n training -l ray.io/node-type=head -f
```

### Monitor

```bash
# GPU utilization (Batch MNP — via SSM)
aws ssm send-command --instance-ids <instance-id> \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["docker exec $(docker ps -q | head -1) nvidia-smi"]'

# GPU utilization (EKS — via kubectl)
kubectl exec -n training <head-pod> -- nvidia-smi

# TensorBoard (results on EFS at /mnt/efs/rl-training/results/)
tensorboard --logdir /mnt/efs/rl-training/results/
```

### Teardown

```bash
# Batch MNP
AWS_DEFAULT_REGION=us-east-2 cdk destroy --force

# EKS (use direct API if CDK hangs on custom resources)
aws eks delete-cluster --name gr00t-rl-eks --region us-east-2
```

## Modes

The EKS + KubeRay backend supports two runtime modes via a `mode` CDK context param. Unset (or `mode=train`) preserves today's behavior; `mode=eval` runs standalone evaluation on a saved checkpoint.

### MODE=train (default)

- **Purpose:** PPO training on the Assemble Trocar task.
- **Topology:** 1 head pod on `<learner-instance>` (8× L40S or 8× H100) + N worker pods on `<rollout-instance>` (1× L40S each). Requires a Capacity Block for the learner if using p5.
- **Invocation:**

```bash
cd training/gr00t/rl/infra
AWS_DEFAULT_REGION=us-east-2 cdk deploy GR00TRLEKSStack \
  --context compute_backend=eks \
  --context vpc_id=<your-vpc-id> \
  --context s3_data_bucket=<your-s3-bucket> \
  --context image_uri=<your-ecr-uri> \
  --context capacity_reservation_id=<your-cr-id>
```

- **Output:** checkpoints at `${LOG_DIR}/checkpoints/global_step_N/`, TensorBoard at `${LOG_DIR}/tensorboard/` under FSx.

### MODE=eval

- **Purpose:** standalone evaluation of a saved RL checkpoint. Produces MP4 rollout videos. No learner GPU required.
- **Topology:** 1 head pod + 1 worker pod, both on `<rollout-instance>` (1× L40S each). 64 episodes total (8 envs × 8 rollout epochs).
- **Invocation:**

```bash
AWS_DEFAULT_REGION=us-east-2 cdk deploy GR00TRLEKSStack \
  --context compute_backend=eks \
  --context vpc_id=<your-vpc-id> \
  --context s3_data_bucket=<your-s3-bucket> \
  --context image_uri=<your-ecr-uri> \
  --context mode=eval \
  --context eval_ckpt=/mnt/fsx/<your-run-path>/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt
```

- **Prerequisites:** the rollout nodegroup can scale to 2 nodes; the training learner nodegroup can be at `desired=0` (no Capacity Block required).
- **Runtime:** ~8-15 minutes end-to-end (setup dominated; the eval loop itself is ~30 min for 64 episodes at N=8 envs, but bootstrap and shutdown are the majority).
- **Output:** MP4 videos at `${LOG_DIR}/video/eval/` on FSx (auto-exported to `s3://<your-s3-bucket>/rl-training/results/…` via the FSx Data Repository Association).
- **Retrieve videos:**

```bash
# From a rollout pod:
kubectl exec -n training <rollout-pod> -- ls -la <LOG_DIR>/video/eval/
kubectl cp training/<rollout-pod>:<LOG_DIR>/video/eval/ ./local-videos/

# Or from S3 (DRA-exported):
aws s3 sync s3://<your-s3-bucket>/<your-run-path>/video/eval/ ./local-videos/
```

- **Metrics:** `eval/success_once` printed to the head-pod logs; retrieve via:

```bash
kubectl logs -n training -l ray.io/node-type=head | grep -A2 'success_once'
```

**Notes:**

- `eval_ckpt` must be a full FSx-visible path (mount root `/mnt/fsx`). Objects on the S3 side of the DRA are auto-imported to FSx lazily on first access.
- MODE=eval fails fast at pod startup if `eval_ckpt` points to a non-existent file. If `eval_ckpt` is omitted or empty, the pod runs base-model eval against the HF snapshot at `MODEL_PATH` (no RL overlay).
- The 2-pod eval topology is fixed at `env.eval.total_num_envs=8, algorithm.eval_rollout_epoch=8` via Hydra overrides in `entrypoint-eks.sh`. To change these, edit that override list.
- Video output size is roughly a few hundred MB per 64-episode run (256 frames per episode × 64 episodes).

## Architecture

### Batch MNP (Homogeneous)

```
AWS Batch MNP Job (2× g6e.12xlarge)
├── Node 0 (Learner): Ray Head + FSDP Actor (GPUs 0-3)
└── Node 1 (Rollout): Ray Worker + Isaac Sim Envs (GPUs 4-7)

Storage: EFS mounted natively at /mnt/efs
Network: NCCL over TCP (no EFA)
```

### EKS + KubeRay (Heterogeneous)

```
EKS Cluster (gr00t-rl-eks)
├── Head Pod (g6e.48xlarge, 8× L40S)
│   ├── Ray Head + FSDP Actor (all 8 GPUs)
│   └── entrypoint-eks.sh → train_embodied_agent.py
└── Worker Pods ×4 (g6e.4xlarge, 1× L40S each)
    ├── Ray Workers
    └── Isaac Sim EnvWorker + RolloutWorker (32 envs each)

Storage: EFS via CSI driver at /mnt/efs
Operators: KubeRay, NVIDIA device plugin
```

**Eval mode** (`--context mode=eval`) drops the learner pod and uses a 2-pod topology: 1 head + 1 worker, both on the rollout nodegroup (1× L40S each). No p5.48xlarge / no Capacity Block required. The head pod runs `entrypoint-eks.sh → eval_embodied_agent.py` and writes MP4 rollouts to `${LOG_DIR}/video/eval/`. See the [Modes](#modes) section above for the invocation.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (Proximal Policy Optimization) |
| Model | GR00T N1.5 (750M params: 550M DiT + 201M SelfAttention) |
| FSDP | Fully Sharded Data Parallel across actor GPUs |
| micro_batch_size | 128 (configurable via `MICRO_BATCH_SIZE` env var) |
| gradient_checkpointing | True (must stay True with batch size 128) |
| Rollout epochs | 8 per iteration |
| Update epochs | 4 per iteration |
| Save interval | Every 2 iterations |
| Max epochs | 1000 |

## Training Outputs

Results are saved to EFS:

```
/mnt/efs/rl-training/results/<config_name>/<timestamp>/
├── tensorboard/events.out.tfevents.*   # Training metrics
└── gr00t_assemble_trocar/
    └── checkpoints/global_step_N/
        └── actor/model_state_dict/full_weights.pt  # Model checkpoint (~5.5 GB)
```

## Key Environment Variables

| Variable | Purpose |
|----------|---------|
| `RLINF_EXT_MODULE=rlinf_ext` | Loads custom extension into RLinf Ray actors |
| `TORCHDYNAMO_DISABLE=1` | Prevents torch.compile deadlock with Isaac Sim |
| `NCCL_IB_DISABLE=1` | Forces NCCL over TCP (no InfiniBand/EFA) |
| `NCCL_SOCKET_IFNAME=eth0` | Explicit NIC for NCCL communication |
| `RAY_memory_usage_threshold=0.99` | Prevents Ray from killing workers on high RAM usage |

## Directory Structure

```
training/gr00t/rl/
├── README.md                    # This file
├── docker/
│   ├── Dockerfile.unified       # Container image (Isaac Sim + PyTorch + deps)
│   ├── entrypoint.sh            # Batch MNP entrypoint
│   ├── entrypoint-eks.sh        # EKS/KubeRay entrypoint
│   ├── buildspec-stage-efs.yml  # CodeBuild: stage code + model to EFS
│   └── requirements-unified.txt # Python dependencies
├── infra/
│   ├── app.py                   # CDK app (routes compute_backend)
│   ├── mnp_batch_stack.py       # Batch MNP CDK stack
│   ├── eks_kuberay_stack.py     # EKS + KubeRay CDK stack
│   └── requirements.txt         # CDK Python dependencies
├── scripts/
│   └── submit_training.sh       # Job submission helper
└── workflows/
    ├── policy/gr00t_config.py   # GR00T data config
    └── simulation/
        ├── rl/rlinf_ext/        # RLinf extension (env registration, model patching)
        └── tasks/assemble_trocar/  # IsaacLab task definition
```

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| CUDA OOM during training step | gradient_checkpointing=False with large batch | Keep GRADIENT_CHECKPOINTING=True (default) when micro_batch_size=128 |
| FSDP NCCL deadlock | cpu_offload enabled | Never set cpu_offload=True |
| torch.compile hangs forever | Incompatible with Isaac Sim multi-process | TORCHDYNAMO_DISABLE=1 |
| Ray kills workers (Batch) | System RAM >95% after 2 iterations | Use EKS backend (768 GB RAM) or set RAY_memory_usage_threshold=0.99 |
| EFS mount timeout (EKS) | Security group misconfigured | efs_sg_id must be the mount target SG |
| Pods Pending (EKS) | Insufficient GPU quota or node not Ready | Check `kubectl describe node` and service quotas |
| `ray: command not found` (EKS) | Ray binary not on PATH | PATH env var includes /isaac-sim/kit/python/bin |
