---
name: deploy-rl-stack
description: >
  Guide for deploying the GR00T RL post-training stack on AWS Batch Multi-Node Parallel (MNP).
  Covers: prerequisites, CDK deployment (auto-builds images + stages EFS), job submission,
  monitoring, evaluation, and cleanup. Use this skill whenever someone wants to run RL training
  for GR00T on AWS — even if they just say "train with RL", "deploy the RL stack",
  "run PPO on the trocar task", or "scale up RL training".
---

# Deploy GR00T RL Post-Training on AWS Batch MNP

This skill deploys a multi-node Ray cluster via AWS Batch for GR00T RL post-training
using RLinf (PPO) on the Assemble Trocar task from i4h-workflows.

**Architecture:**
- **All nodes** (g6e.4xlarge): Homogeneous MNP cluster (Batch requires same instance type)
- **Node 0** (learner): Ray head + RLinf PPO learner, 1× L40S GPU
- **Nodes 1-4** (rollouts): Ray workers + Isaac Sim rollout environments, 1× L40S each
- **Artifacts:** Ray object store (trajectories), EFS (checkpoints/code/model), S3 (episode logs)

> For production 8-GPU FSDP training, use `--context compute_backend=sagemaker` (planned).

## Phase 1: Prerequisites

```bash
# AWS CLI configured
aws sts get-caller-identity --query '[Account, Arn]' --output text

# CDK available
cdk --version  # Requires >= 2.180.0

# Python venv for CDK
cd training/gr00t/rl/infra
python3 -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -r requirements.txt
```

Verify CDK bootstrap:
```bash
aws cloudformation describe-stacks --stack-name CDKToolkit \
  --query 'Stacks[0].StackStatus' --output text
```

**GPU quota:** Minimum 80 vCPUs for G6e On-Demand (5× g6e.4xlarge × 16 vCPUs).
```bash
aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA \
  --query 'Quota.Value' --output text
```

## Phase 2: Deploy Infrastructure

```bash
cd training/gr00t/rl/infra

# Synth first (validates without creating resources)
cdk synth --quiet

# Deploy — creates: VPC (or reuses), EFS, S3, ECR repos, Batch compute env,
# CodeBuild projects (auto-triggered), EFS staging (auto-triggered)
AWS_DEFAULT_REGION=us-west-2 cdk deploy --require-approval never
```

**Reusing Part 1 infrastructure:**
```bash
AWS_DEFAULT_REGION=us-west-2 cdk deploy --require-approval never \
  --context vpc_id=vpc-XXXXX \
  --context efs_id=fs-XXXXX \
  --context efs_sg_id=sg-XXXXX
```

**Using pre-built images (skips CodeBuild):**
```bash
AWS_DEFAULT_REGION=us-west-2 cdk deploy --require-approval never \
  --context learner_image_uri=<account>.dkr.ecr.<region>.amazonaws.com/gr00t-rl-learner:<tag> \
  --context rollout_image_uri=<account>.dkr.ecr.<region>.amazonaws.com/gr00t-rl-rollout:<tag>
```

Capture outputs:
```bash
STACK_OUTPUTS=$(aws cloudformation describe-stacks --stack-name GR00TRLBatchStack \
  --query 'Stacks[0].Outputs' --output json)
export JobQueueName=$(echo "$STACK_OUTPUTS" | jq -r '.[] | select(.OutputKey=="JobQueueName") | .OutputValue')
export JobDefinitionArn=$(echo "$STACK_OUTPUTS" | jq -r '.[] | select(.OutputKey=="JobDefinitionArn") | .OutputValue')
export ArtifactBucket=$(echo "$STACK_OUTPUTS" | jq -r '.[] | select(.OutputKey=="ArtifactBucket") | .OutputValue')
```

## Phase 3: Monitor Container Builds

The deploy auto-triggers three CodeBuild projects:
1. **GR00T-RL-Learner-Build** — Slim image (~15 min)
2. **GR00T-RL-Rollout-Build** — Isaac Sim base image (~20 min)
3. **GR00T-RL-Stage-EFS** — Clones repos + downloads model to EFS (~10 min)

```bash
# Monitor build status
aws codebuild batch-get-builds \
  --ids $(aws codebuild list-builds-for-project --project-name GR00T-RL-Learner-Build --query 'ids[0]' --output text) \
         $(aws codebuild list-builds-for-project --project-name GR00T-RL-Rollout-Build --query 'ids[0]' --output text) \
  --query 'builds[*].[projectName,buildStatus,currentPhase]' --output table

# EFS staging
aws codebuild batch-get-builds \
  --ids $(aws codebuild list-builds-for-project --project-name GR00T-RL-Stage-EFS --query 'ids[0]' --output text) \
  --query 'builds[0].[buildStatus,currentPhase]' --output text
```

Wait for all three to show `SUCCEEDED` before submitting jobs.

**If a build fails:** Check logs:
```bash
aws logs tail /aws/codebuild/GR00T-RL-Learner-Build --follow
aws logs tail /aws/codebuild/GR00T-RL-Rollout-Build --follow
aws logs tail /aws/codebuild/GR00T-RL-Stage-EFS --follow
```

## Phase 4: Submit Training Job

```bash
bash scripts/submit_training.sh \
  --num-nodes 5 \
  --model-path /mnt/efs/models/GR00T-N1.5-RL-Rheo-AssembleTrocar \
  --num-envs 64
```

Or submit directly:
```bash
aws batch submit-job \
  --job-name "gr00t-rl-training" \
  --job-queue "$JobQueueName" \
  --job-definition "$JobDefinitionArn" \
  --query 'jobId' --output text
```

**Low memory tip:** Reduce `--num-envs` to 32 if rollout nodes OOM.

## Phase 5: Monitor Training

```bash
JOB_ID=<from submit output>

# Job status
aws batch describe-jobs --jobs $JOB_ID --query 'jobs[0].status'

# Stream logs (main node / learner)
aws logs tail /aws/batch/job --follow

# Child node logs
aws batch describe-jobs --jobs "${JOB_ID}#1" \
  --query 'jobs[0].container.logStreamName' --output text
```

**TensorBoard** (from workstation with EFS mounted):
```bash
tensorboard --logdir /mnt/efs/rl-training/results/ --host 0.0.0.0 --port 6006
```

**Expected startup sequence:**
1. `Starting Ray head on port 6379...` (main node)
2. `Waiting for Ray head...` → `Ray head reachable. Joining cluster...` (child nodes)
3. `All 5 nodes connected.` (main node)
4. `Launching RLinf learner...` (main node)
5. Hydra config prints, FSDP actor initialization, training begins

## Phase 6: Evaluate Trained Checkpoint

After training completes, evaluate on a DCV workstation or g6e.4xlarge instance:

```bash
docker run --rm -it --gpus all --ipc=host \
  -v /mnt/efs:/workspaces \
  -v /mnt/efs/models:/models \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  --entrypoint "" --user root \
  <rollout_ecr_uri>:<tag> bash

# Inside container:
python -u /workspaces/workflows/rheo/scripts/simulation/examples/eval_assemble_trocar.py \
  --enable_cameras \
  --task Isaac-Assemble-Trocar-G129-Dex3-Joint \
  --model_path /mnt/efs/rl-training/results/<timestamp>/checkpoint \
  --rl_ckpt \
  --num_episodes 100 \
  --max_steps 500
```

Expected: ~82% success rate on full 4-stage trocar assembly.

## Cleanup

```bash
cd training/gr00t/rl/infra
cdk destroy --force

# Retained resources (manual):
aws ecr delete-repository --repository-name gr00t-rl-learner --force
aws ecr delete-repository --repository-name gr00t-rl-rollout --force
# EFS and S3 have RETAIN policy — delete if no longer needed
```

## Troubleshooting

### Job stuck in RUNNABLE
- Check vCPU quota: need ≥80 for 5× g6e.4xlarge
- Verify instance availability: `aws ec2 describe-instance-type-offerings --location-type availability-zone --filters "Name=instance-type,Values=g6e.4xlarge"`
- Check compute env: `aws batch describe-compute-environments --compute-environments GR00T-RL-ComputeEnv`

### Ray workers not connecting
- Security group allows all TCP self-referencing (verified in CDK)
- Check child node logs for import errors (missing Python packages)
- Verify EFS mount: child logs should show `EFS mounted successfully.`

### Training crashes on model load
- Ensure Isaac-GR00T is on EFS: `ls /mnt/efs/third_party/Isaac-GR00T/gr00t/`
- Ensure model checkpoint exists: `ls /mnt/efs/models/GR00T-N1.5-RL-Rheo-AssembleTrocar/config.json`
- Check transformers version is 4.x (not 5.x): `pip show transformers`

### Cached old images on Batch instances
If Batch reuses instances with stale images, terminate them:
```bash
aws ec2 terminate-instances --instance-ids \
  $(aws ec2 describe-instances \
    --filters "Name=tag:aws:batch:compute-environment,Values=*GR00T-RL*" \
              "Name=instance-state-name,Values=running" \
    --query 'Reservations[*].Instances[*].InstanceId' --output text)
```
Then resubmit — fresh instances will pull current images.

### OOM on rollout nodes
- Reduce `--num-envs` (try 32 or 16)
- Use `g6e.8xlarge` for more RAM (requires CDK stack update)
