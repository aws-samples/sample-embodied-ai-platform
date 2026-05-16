# GR00T RL Post-Training Deployment Guide

This guide walks through deploying the GR00T RL post-training infrastructure on AWS
using Batch Multi-Node Parallel (MNP) jobs.

## Overview

RL training has two distinct workloads that run as a single coordinated cluster:

1. **Learner** — Consumes trajectories, computes PPO gradients. Needs GPU for model updates.
2. **Rollouts** — Run the current policy in Isaac Sim, produce trajectories. Each needs one GPU.

AWS Batch MNP maps this as: one main node (learner) + N child nodes (rollouts), gang-scheduled
as a single job with automatic service discovery.

**Constraint:** AWS Batch MNP requires homogeneous instance types. All nodes use g6e.4xlarge
(1× L40S, 48GB VRAM). For production 8-GPU FSDP training, use the `sagemaker` compute backend
(planned) which supports heterogeneous InstanceGroups.

## Architecture

```
┌─────────────────── AWS Batch MNP Job ─────────────────────┐
│                                                            │
│  Node 0: g6e.4xlarge (main)     Nodes 1-4: g6e.4xlarge    │
│  ┌────────────────────────┐     ┌────────────────────────┐ │
│  │ Ray Head               │     │ Ray Workers            │ │
│  │ RLinf Learner (PPO)    │◄───►│ RLinf Rollout Workers  │ │
│  │ 1× L40S (48 GB VRAM)  │ Ray │ Isaac Sim + GR00T N1.5 │ │
│  │ Learner image (slim)   │store│ Rollout image (full)   │ │
│  └────────────────────────┘     └────────────────────────┘ │
│           │                              │                  │
│           ▼                              ▼                  │
│     EFS (checkpoints,             S3 (episode logs,        │
│      TensorBoard, code)            eval videos)            │
└─────────────────────────────────────────────────────────────┘
```

**Three-tier artifact flow:**
- **Ray object store** — Trajectories between rollouts and learner (in-memory, low-latency)
- **EFS** — Checkpoints, TensorBoard logs, model weights, training code (persistent, shared)
- **S3** — Episode logs, eval videos (durable, for downstream analysis)

## Prerequisites

### AWS Account

- **Region:** us-west-2 (or any region with g6e availability)
- **vCPU Quota:** ≥80 for G6e On-Demand (5 × g6e.4xlarge × 16 vCPUs)
- **CDK Bootstrap:** `cdk bootstrap` in the target account/region

### Software

```bash
cd training/gr00t/rl/infra
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cdk --version  # >= 2.180.0
```

## Step 1: Deploy Infrastructure

The CDK stack creates everything needed in a single deploy:

| Resource | Purpose |
|----------|---------|
| VPC + private subnets | Network isolation, NAT for ECR pulls |
| EFS | Shared filesystem for code, model, checkpoints |
| S3 bucket | Durable artifact storage |
| ECR repositories (×2) | Learner and rollout container images |
| CodeBuild projects (×3) | Auto-build learner, rollout, EFS staging |
| Batch compute environment | GPU instance pool (g6e family) |
| Batch job queue + definition | MNP job orchestration |

```bash
AWS_DEFAULT_REGION=us-west-2 cdk deploy --require-approval never
```

**Reusing Part 1 VPC/EFS:**
```bash
AWS_DEFAULT_REGION=us-west-2 cdk deploy --require-approval never \
  --context vpc_id=vpc-XXXXX \
  --context efs_id=fs-XXXXX \
  --context efs_sg_id=sg-XXXXX
```

## Step 2: Wait for Builds

The deploy auto-triggers three CodeBuild jobs:

1. **Learner image build** (~5 min) — Slim image: CUDA 12.8 + PyTorch 2.6 + Ray 2.47 + transformers
2. **Rollout image build** (~15 min) — Full image: Isaac Sim 5.1.0 + Ray 2.47 + GR00T deps
3. **EFS staging** (~10 min) — Clones i4h-workflows (v0.5.0), RLinf, Isaac-GR00T; downloads model

Monitor:
```bash
# Image builds
for proj in GR00T-RL-Learner-Build GR00T-RL-Rollout-Build GR00T-RL-Stage-EFS; do
  STATUS=$(aws codebuild list-builds-for-project --project-name $proj --query 'ids[0]' --output text | \
    xargs -I{} aws codebuild batch-get-builds --ids {} --query 'builds[0].buildStatus' --output text)
  echo "$proj: $STATUS"
done
```

Wait for all three to show `SUCCEEDED`.

## Step 3: Submit Training Job

```bash
bash scripts/submit_training.sh \
  --num-nodes 5 \
  --model-path /mnt/efs/models/GR00T-N1.5-RL-Rheo-AssembleTrocar \
  --num-envs 64
```

**What happens:**
1. Batch gang-schedules 5× g6e.4xlarge instances
2. Node 0: starts Ray head, waits for workers, launches RLinf learner
3. Nodes 1-4: join Ray cluster, become rollout workers with Isaac Sim
4. PPO training runs with stage-gated sparse rewards across 4 curriculum stages
5. Checkpoints save to EFS; artifacts upload to S3

## Step 4: Monitor

```bash
JOB_ID=<from submit output>

# Status
aws batch describe-jobs --jobs $JOB_ID --query 'jobs[0].status'

# Logs
aws logs tail /aws/batch/job --follow

# TensorBoard (from machine with EFS mounted)
tensorboard --logdir /mnt/efs/rl-training/results/ --port 6006
```

## Step 5: Evaluate

After training, run evaluation on a GPU instance with Isaac Sim:

```bash
python -u eval_assemble_trocar.py \
  --enable_cameras \
  --task Isaac-Assemble-Trocar-G129-Dex3-Joint \
  --model_path /mnt/efs/rl-training/results/<timestamp>/checkpoint \
  --rl_ckpt \
  --num_episodes 100 \
  --max_steps 500
```

### Expected Results

| Stage | SFT Baseline | After RL | Improvement |
|-------|-------------|----------|-------------|
| 1 (lift) | 83% | 100% | +17pp |
| 1+2 (align) | 72% | 92% | +20pp |
| 1+2+3 (insert) | 32% | 85% | +53pp |
| 1+2+3+4 (place) | 29% | 82% | +53pp |

## Cleanup

```bash
cd training/gr00t/rl/infra
cdk destroy --force

# Retained resources:
aws ecr delete-repository --repository-name gr00t-rl-learner --force
aws ecr delete-repository --repository-name gr00t-rl-rollout --force
```

## Configuration Reference

| Context Parameter | Default | Description |
|---|---|---|
| `vpc_id` | Creates new | Existing VPC ID |
| `efs_id` | Creates new | Existing EFS file system ID |
| `efs_sg_id` | - | EFS security group (required if efs_id set) |
| `compute_backend` | `batch-mnp` | `batch-mnp` or `sagemaker` (planned) |
| `num_rollout_nodes` | 4 | Number of rollout child nodes |
| `learner_image_uri` | Auto-built | Pre-built ECR URI (skips CodeBuild) |
| `rollout_image_uri` | Auto-built | Pre-built ECR URI (skips CodeBuild) |

## Troubleshooting

### Job stuck in RUNNABLE
Check vCPU quota and instance availability. Need ≥80 vCPUs for G6e On-Demand.

### Ray workers not connecting
Check child node logs for Python import errors. The rollout image uses `--ignore-installed`
to override Isaac Sim's partial packages — if Ray fails to import, a dependency is missing
from `requirements-rollout.txt`.

### Model loading fails
Ensure `/mnt/efs/third_party/Isaac-GR00T` exists with the `gr00t` Python package.
The EFS staging CodeBuild project clones this at the pinned commit.

### Cached stale images
Terminate Batch instances to force fresh image pulls on next job submission.
