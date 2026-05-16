# GR00T RL Post-Training on AWS Batch MNP

Reinforcement learning post-training for GR00T N1.5 on the Assemble Trocar task using AWS Batch Multi-Node Parallel (MNP) jobs with RLinf (PPO).

## Architecture

```
┌─────────────────── AWS Batch MNP Job ─────────────────────┐
│                                                            │
│  Node 0 (g6e.4xlarge)              Nodes 1-4 (g6e.4xlarge)│
│  ┌────────────────────────┐        ┌─────────────────────┐│
│  │ Ray Head               │        │ Ray Workers         ││
│  │ RLinf Learner (PPO)    │◄──────►│ RLinf Rollout       ││
│  │ 1× L40S (48GB VRAM)   │  Ray   │ Isaac Sim + GR00T   ││
│  │ GR00T + RLinf          │ store  │ 1× L40S each        ││
│  └────────────────────────┘        └─────────────────────┘│
│           │                              │                 │
│           ▼                              ▼                 │
│     EFS (checkpoints,             S3 (episode logs,       │
│      TensorBoard, code)            eval videos)           │
└────────────────────────────────────────────────────────────┘
```

## Compute Backends

Two deployment paths via `--context compute_backend=`:

| Backend | Instance Types | Use Case |
|---------|---------------|----------|
| `batch-mnp` (default) | All g6e.4xlarge (homogeneous) | E2E validation, smaller batch training |
| `sagemaker` (planned) | g6e.48xlarge learner + g6e.4xlarge rollouts | Production 8-GPU FSDP training |

AWS Batch MNP requires homogeneous instance types. For heterogeneous learner/rollout clusters at production scale, the `sagemaker` backend uses Batch → SageMaker Training with InstanceGroups.

## Quick Start

```bash
cd training/gr00t/rl/infra
pip install -r requirements.txt

# Deploy (creates VPC/EFS/S3/Batch/CodeBuild + auto-builds images + stages EFS)
AWS_DEFAULT_REGION=us-west-2 cdk deploy --require-approval never

# Or reuse existing VPC from Part 1:
AWS_DEFAULT_REGION=us-west-2 cdk deploy --require-approval never \
  --context vpc_id=vpc-XXXXX
```

Deployment auto-triggers:
1. **CodeBuild**: Builds learner + rollout container images, pushes to ECR
2. **EFS staging**: Clones i4h-workflows (v0.5.0), RLinf, Isaac-GR00T; downloads model checkpoint

## Submitting Training Jobs

```bash
bash scripts/submit_training.sh \
  --num-nodes 5 \
  --model-path /mnt/efs/models/GR00T-N1.5-RL-Rheo-AssembleTrocar \
  --num-envs 64
```

## Deployment Paths

### Path 1: Fully Automated (Recommended)

Just `cdk deploy`. CodeBuild builds images, stages EFS. No local Docker needed.

### Path 2: Pre-built Images

Build images yourself, skip CodeBuild:

```bash
cdk deploy --require-approval never \
  --context learner_image_uri=<account>.dkr.ecr.<region>.amazonaws.com/gr00t-rl-learner:latest \
  --context rollout_image_uri=<account>.dkr.ecr.<region>.amazonaws.com/gr00t-rl-rollout:latest
```

## Directory Structure

```
training/gr00t/rl/
├── infra/
│   ├── app.py                  # CDK app entry
│   ├── mnp_batch_stack.py      # MNP Batch stack (compute, queue, CodeBuild, EFS staging)
│   ├── cdk.json
│   └── requirements.txt
├── docker/
│   ├── Dockerfile.learner      # Slim: CUDA + PyTorch + Ray + GR00T deps
│   ├── Dockerfile.rollout      # Full: Isaac Sim + Ray + GR00T deps
│   ├── entrypoint.sh           # MNP role dispatch (head vs worker via node index)
│   ├── buildspec.yml           # CodeBuild spec for image builds
│   ├── buildspec-stage-efs.yml # CodeBuild spec for EFS staging
│   ├── requirements-learner.txt
│   └── requirements-rollout.txt
├── scripts/
│   ├── submit_training.sh      # Submit MNP job
│   └── build_and_push.sh       # Manual image build (Path 2)
├── config/
│   └── isaaclab_ppo_gr00t_assemble_trocar.yaml  # Hydra training config
├── SKILL.md                    # Agent-executable deployment guide
├── deployment.md               # Human-readable walkthrough
├── TODO.md                     # Optimization backlog
└── README.md                   # This file
```

## Pinned Versions

All dependencies are pinned in `requirements-learner.txt` and `requirements-rollout.txt` for reproducibility. Key versions:

| Dependency | Version | Note |
|---|---|---|
| i4h-workflows | v0.5.0 | EFS staging pins this tag |
| RLinf | 649e757 | Pinned commit |
| Isaac-GR00T | 4af2b62 | Pinned commit (N1.5) |
| Isaac Sim | 5.1.0 | Rollout base image |
| Ray | 2.47.0 | RLinf minimum requirement |
| PyTorch (learner) | 2.6.0+cu124 | |
| transformers | 4.51.3 | Must be 4.x (5.x removes APIs) |

## Monitoring

```bash
# Job status
aws batch describe-jobs --jobs <JOB_ID> --query 'jobs[0].status'

# Logs
aws logs tail /aws/batch/job --follow

# TensorBoard (from workstation with EFS mounted)
tensorboard --logdir /mnt/efs/rl-training/results/ --port 6006
```

## Cleanup

```bash
cd training/gr00t/rl/infra
cdk destroy --force

# Retained resources (manual cleanup):
aws ecr delete-repository --repository-name gr00t-rl-learner --force
aws ecr delete-repository --repository-name gr00t-rl-rollout --force
```

## Related

- [Blog Part 2 Outline](../../../../Blog/Embodied%20AI%20Blog%20Series%2C%20Part%202.md) — Narrative context
- [Part 1 SFT Stack](../infra/) — Shared VPC/EFS infrastructure
- [i4h-workflows RL Guide](https://github.com/isaac-for-healthcare/i4h-workflows/blob/main/workflows/rheo/docs/assemble_trocar_rl_guide.md) — Upstream training reference
