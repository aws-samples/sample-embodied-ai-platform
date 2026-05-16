# GR00T RL Post-Training Deployment Guide

This guide walks through deploying the GR00T RL post-training infrastructure on AWS
using Batch Multi-Node Parallel (MNP) jobs. It covers the same steps as `SKILL.md`
in a narrative format for readers who prefer manual deployment.

## Overview

The RL post-training stack extends the Part 1 SFT infrastructure with a multi-node
Ray cluster for running RLinf (PPO). The key insight is that RL training has two
distinct workloads:

1. **Learner** — A single stateful process consuming trajectories and computing PPO
   gradients. Needs many GPUs (FSDP) but no simulation.
2. **Rollouts** — Many stateless copies of the current policy running in Isaac Sim,
   producing trajectories. Each needs one GPU for simulation + inference.

AWS Batch MNP maps this cleanly: one main node runs the learner, child nodes run
rollouts. Gang scheduling ensures they start together; MNP environment variables
provide service discovery (the main node's IP is injected into all containers).

## Architecture

```
┌─────────────────── AWS Batch MNP Job ─────────────────────┐
│                                                            │
│  Node 0: g6e.48xlarge (main)    Nodes 1-4: g6e.4xlarge    │
│  ┌────────────────────────┐     ┌────────────────────────┐ │
│  │ Ray Head               │     │ Ray Workers            │ │
│  │ RLinf Learner (FSDP)   │◄───►│ RLinf Rollout Workers  │ │
│  │ 8× L40S (384 GB VRAM)  │ Ray │ Isaac Sim + GR00T N1.5 │ │
│  │ GR00T + RLinf only      │store│ 1× L40S (48 GB) each  │ │
│  └────────────────────────┘     └────────────────────────┘ │
│           │                              │                  │
│           ▼                              ▼                  │
│     EFS (checkpoints,             S3 (episode logs,        │
│      TensorBoard)                  eval videos)            │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

### AWS Account Setup

- **Region:** us-west-2 (or any region with g6e availability)
- **vCPU Quota:** Request at least 256 vCPUs for G6e On-Demand instances
  - g6e.48xlarge = 192 vCPUs (learner)
  - g6e.4xlarge × 4 = 64 vCPUs (4 rollout nodes)
- **CDK Bootstrap:** `cdk bootstrap` in the target account/region
- **Docker:** Installed locally for building container images

### Software

```bash
# Install CDK dependencies
cd training/gr00t/rl/infra
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Verify
cdk --version
aws sts get-caller-identity
```

## Step 1: Deploy Infrastructure

The CDK stack creates: VPC (or reuses existing), EFS, S3 bucket, ECR repositories,
Batch compute environment, and job queue.

```bash
cd training/gr00t/rl/infra
cdk synth --quiet  # Validate first
cdk deploy --require-approval never
```

**Reusing Part 1 infrastructure:**

If you deployed the SFT stack from Part 1, reuse its VPC and EFS:

```bash
cdk deploy --require-approval never \
  --context vpc_id=vpc-XXXXX \
  --context efs_id=fs-XXXXX \
  --context efs_sg_id=sg-XXXXX
```

## Step 2: Build Container Images

Two separate images optimize for each node's role:

| Image | Base | Contents | Size |
|-------|------|----------|------|
| Learner | `nvidia/cuda:12.8.0-devel-ubuntu22.04` | PyTorch, Ray, GR00T, RLinf | ~15 GB |
| Rollout | `nvcr.io/nvidia/isaac-sim:5.1.0` | Isaac Sim, Ray, GR00T, RLinf | ~45 GB |

```bash
cd training/gr00t/rl
bash scripts/build_and_push.sh
```

This builds both images and pushes them to the ECR repositories created in Step 1.

## Step 3: Stage Model on EFS

The model checkpoint and training code must be accessible from all nodes via EFS.
From a machine with EFS mounted (e.g., DCV workstation):

```bash
# Clone the i4h-workflows repo (contains RLinf, IsaacLab, task definitions)
git clone https://github.com/isaac-for-healthcare/i4h-workflows.git /mnt/efs
cd /mnt/efs && git submodule update --init --recursive

# Download the GR00T checkpoint
mkdir -p /mnt/efs/models
hf download nvidia/GR00T-N1.5-RL-Rheo-AssembleTrocar \
  --local-dir /mnt/efs/models/GR00T-N1.5-RL-Rheo-AssembleTrocar
```

## Step 4: Create Job Definition

Redeploy with container image URIs to create the MNP job definition:

```bash
cd training/gr00t/rl/infra
cdk deploy --require-approval never \
  --context learner_image_uri=<LearnerECR>:latest \
  --context rollout_image_uri=<RolloutECR>:latest \
  --context num_rollout_nodes=4
```

## Step 5: Submit Training Job

```bash
bash scripts/submit_training.sh --num-nodes 5 --num-envs 64
```

The job:
1. Gang-schedules all 5 nodes simultaneously
2. Node 0 starts Ray head, waits for workers, launches RLinf learner
3. Nodes 1-4 join Ray cluster, become rollout workers running Isaac Sim
4. Training runs PPO with stage-gated sparse rewards across 4 curriculum stages
5. Checkpoints save to EFS; episode artifacts upload to S3

### Monitoring

```bash
# Job status
aws batch describe-jobs --jobs <JOB_ID> --query 'jobs[0].status'

# Logs
aws logs tail /aws/batch/job --follow

# TensorBoard (from workstation)
tensorboard --logdir /mnt/efs/rl-training/results/ --port 6006
```

## Step 6: Evaluate

Run evaluation on the DCV workstation using the trained checkpoint:

```bash
python -u scripts/simulation/examples/eval_assemble_trocar.py \
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
# Destroy infrastructure
cd training/gr00t/rl/infra
cdk destroy --force

# Remove retained resources
aws ecr delete-repository --repository-name gr00t-rl-learner --force
aws ecr delete-repository --repository-name gr00t-rl-rollout --force
```

EFS and S3 bucket are retained by default. Delete manually if no longer needed.

## Troubleshooting

### Job stuck in RUNNABLE

- Check vCPU quota: `aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA`
- Verify instance availability in your AZ
- Check compute environment status: `aws batch describe-compute-environments --compute-environments GR00T-RL-ComputeEnv`

### Ray workers not connecting

- Verify security group allows all TCP between nodes (self-referencing rule)
- Check `AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS` is populated in worker logs
- Increase timeout in `entrypoint.sh` if nodes are slow to start

### OOM on rollout nodes

- Reduce `--num-envs` (try 32 or 16)
- Use `g6e.8xlarge` for rollout nodes (128 GB RAM)

### Learner FSDP errors

- Verify all 8 GPUs visible: check `CUDA_VISIBLE_DEVICES` in learner logs
- Ensure model fits in aggregate GPU memory (GR00T N1.5 3B requires ~48 GB with FSDP)
