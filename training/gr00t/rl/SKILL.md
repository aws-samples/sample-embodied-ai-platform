---
name: deploy-rl-stack
description: >
  Guide for deploying the GR00T RL post-training stack on AWS Batch Multi-Node Parallel (MNP).
  Covers: prerequisites, CDK deployment, container builds, model staging on EFS, job submission,
  monitoring, evaluation, and cleanup. Use this skill whenever someone wants to run RL training
  for GR00T on AWS — even if they just say "train with RL", "deploy the RL stack",
  "run PPO on the trocar task", or "scale up RL training".
---

# Deploy GR00T RL Post-Training on AWS Batch MNP

This skill deploys a multi-node Ray cluster via AWS Batch for GR00T RL post-training
using RLinf (PPO) on the Assemble Trocar task from i4h-workflows.

**Architecture:**
- **Main node** (g6e.48xlarge): Ray head + RLinf learner with intra-node FSDP (8× L40S)
- **Child nodes** (g6e.4xlarge × N): Ray workers + RLinf rollout workers with Isaac Sim (1× L40S each)
- **Artifacts:** Ray object store (trajectories), EFS (checkpoints/TensorBoard), S3 (episode logs/videos)

## Phase 1: Prerequisites

```bash
# AWS CLI configured
aws sts get-caller-identity --query '[Account, Arn]' --output text

# CDK available
cdk --version

# Docker running (for container builds)
docker info > /dev/null 2>&1 && echo "Docker OK"

# Python venv for CDK
cd training/gr00t/rl/infra
python3 -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -r requirements.txt
```

Also verify CDK bootstrap:
```bash
aws cloudformation describe-stacks --stack-name CDKToolkit \
  --query 'Stacks[0].StackStatus' --output text
```

**GPU quota:** Ensure sufficient vCPU quota for g6e instances in your target region.
- g6e.48xlarge = 192 vCPUs (learner)
- g6e.4xlarge × 4 = 64 vCPUs (rollouts)
- Total: ~256 vCPUs minimum for G6e family

## Phase 2: Deploy Infrastructure

```bash
cd training/gr00t/rl/infra

# Synth first (validates without creating resources)
cdk synth --quiet

# Deploy (creates VPC, EFS, ECR repos, Batch compute env, job queue)
# To reuse Part 1 VPC/EFS:
#   --context vpc_id=vpc-XXX --context efs_id=fs-XXX --context efs_sg_id=sg-XXX
cdk deploy --require-approval never
```

Capture outputs:
```bash
STACK_OUTPUTS=$(aws cloudformation describe-stacks --stack-name GR00TRLBatchStack \
  --query 'Stacks[0].Outputs' --output json)
export VpcId=$(echo "$STACK_OUTPUTS" | jq -r '.[] | select(.OutputKey=="VpcId") | .OutputValue')
export EFSId=$(echo "$STACK_OUTPUTS" | jq -r '.[] | select(.OutputKey=="EFSFileSystemId") | .OutputValue')
export LearnerECR=$(echo "$STACK_OUTPUTS" | jq -r '.[] | select(.OutputKey=="LearnerECR") | .OutputValue')
export RolloutECR=$(echo "$STACK_OUTPUTS" | jq -r '.[] | select(.OutputKey=="RolloutECR") | .OutputValue')
export ArtifactBucket=$(echo "$STACK_OUTPUTS" | jq -r '.[] | select(.OutputKey=="ArtifactBucket") | .OutputValue')
```

## Phase 3: Build and Push Container Images

```bash
cd training/gr00t/rl
bash scripts/build_and_push.sh
```

This builds both `Dockerfile.learner` (slim: RLinf + GR00T + Ray) and `Dockerfile.rollout`
(full: Isaac Sim + RLinf + GR00T + Ray), then pushes to the ECR repositories created in Phase 2.

## Phase 4: Stage Model and Code on EFS

Mount EFS from a workstation (DCV instance or similar) and stage:

```bash
# On a machine with EFS mounted at /mnt/efs:

# 1. Clone i4h-workflows (provides RLinf, IsaacLab, task definitions)
git clone https://github.com/isaac-for-healthcare/i4h-workflows.git /mnt/efs
cd /mnt/efs && git submodule update --init --recursive

# 2. Download the base model checkpoint
mkdir -p /mnt/efs/models
hf download nvidia/GR00T-N1.5-RL-Rheo-AssembleTrocar \
  --local-dir /mnt/efs/models/GR00T-N1.5-RL-Rheo-AssembleTrocar
```

## Phase 5: Redeploy with Container Image URIs

```bash
cd training/gr00t/rl/infra
cdk deploy --require-approval never \
  --context learner_image_uri="${LearnerECR}:latest" \
  --context rollout_image_uri="${RolloutECR}:latest"
```

This creates the MNP job definition now that both container images are available.

## Phase 6: Submit Training Job

```bash
cd training/gr00t/rl
bash scripts/submit_training.sh \
  --num-nodes 5 \
  --model-path /mnt/efs/models/GR00T-N1.5-RL-Rheo-AssembleTrocar \
  --num-envs 64
```

**Low memory tip:** Reduce `--num-envs` if rollout nodes OOM. Start with 32 for g6e.4xlarge.

## Phase 7: Monitor Training

```bash
# Job status
JOB_ID=<from submit output>
aws batch describe-jobs --jobs $JOB_ID --query 'jobs[0].status'

# Stream logs (learner node)
aws logs tail /aws/batch/job --follow

# TensorBoard (from DCV workstation with EFS mounted)
tensorboard --logdir /mnt/efs/rl-training/results/ --host 0.0.0.0 --port 6006
```

**Ray dashboard:** Available on the main node at port 8265 (accessible via SSM port forwarding
to the Batch instance if needed).

## Phase 8: Evaluate Trained Checkpoint

After training completes, evaluate on the DCV workstation:

```bash
# On the DCV workstation (g6e.4xlarge with Isaac Sim container):
docker run --rm -it --gpus all --ipc=host --net=host \
  -v /mnt/efs:/workspaces \
  -v /mnt/efs/models:/models \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  --entrypoint "" \
  <rollout_ecr_uri>:latest bash

# Inside container:
python -u scripts/simulation/examples/eval_assemble_trocar.py \
  --enable_cameras \
  --task Isaac-Assemble-Trocar-G129-Dex3-Joint \
  --model_path /mnt/efs/rl-training/results/<run_timestamp>/checkpoint \
  --rl_ckpt \
  --num_episodes 100 \
  --max_steps 500
```

Expected: ~82% success rate on the full 4-stage trocar assembly (matching the i4h benchmark).

## Cleanup

```bash
cd training/gr00t/rl/infra

# Destroy stack
cdk destroy --force

# Clean retained resources
aws ecr delete-repository --repository-name gr00t-rl-learner --force
aws ecr delete-repository --repository-name gr00t-rl-rollout --force
# EFS and S3 bucket have RETAIN policy — delete manually if no longer needed
```
