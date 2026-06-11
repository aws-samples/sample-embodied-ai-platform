# GR00T RL Post-Training on AWS

Reinforcement learning post-training for NVIDIA GR00T N1.5 on the Assemble Trocar surgical task using PPO (via RLinf) with two compute backend options.

## Compute Backends

| Backend | Command | Instances | Best For |
|---------|---------|-----------|----------|
| **AWS Batch MNP** (homogeneous) | `--context compute_backend=batch-mnp` | 2× g6e.12xlarge (4× L40S each) | Simple setup, lower cost (~$16/hr) |
| **EKS + KubeRay** (heterogeneous) | `--context compute_backend=eks` | 1× g6e.48xlarge + 4× g6e.4xlarge | Better GPU utilization, no RAM OOM |

Both backends are validated end-to-end: 2 PPO iterations + checkpoint saved.

## Quick Start

### Prerequisites

- AWS account with GPU quota (384 vCPUs for G instances in your region)
- CDK CLI installed (`npm install -g aws-cdk`)
- Python 3.10+ with CDK dependencies: `pip install -r infra/requirements.txt`

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
  --context image_uri=<ecr-image-uri>
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
aws eks update-kubeconfig --name gr00t-rl-eks --region us-east-2
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

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (Proximal Policy Optimization) |
| Model | GR00T N1.5 (750M params: 550M DiT + 201M SelfAttention) |
| FSDP | Fully Sharded Data Parallel across actor GPUs |
| micro_batch_size | 32 |
| gradient_checkpointing | True |
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
| CUDA OOM during training step | micro_batch_size too large | Already set to 32 + gradient_checkpointing |
| FSDP NCCL deadlock | cpu_offload enabled | Never set cpu_offload=True |
| torch.compile hangs forever | Incompatible with Isaac Sim multi-process | TORCHDYNAMO_DISABLE=1 |
| Ray kills workers (Batch) | System RAM >95% after 2 iterations | Use EKS backend (768 GB RAM) or set RAY_memory_usage_threshold=0.99 |
| EFS mount timeout (EKS) | Security group misconfigured | efs_sg_id must be the mount target SG |
| Pods Pending (EKS) | GPU taint blocks system pods | Tolerations baked into CDK Helm values |
| `ray: command not found` (EKS) | Ray binary not on PATH | PATH env var includes /isaac-sim/kit/python/bin |
