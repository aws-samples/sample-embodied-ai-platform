---
name: deploy-batch-training
description: Deploy and run GR00T N1.5 PPO RL post-training on AWS Batch Multi-Node Parallel (homogeneous g6e.12xlarge, 4× L40S per node) backed by EFS. Covers CDK deploy, image build + EFS staging, job submission, monitoring, and teardown.
---

# Deploy GR00T RL Training on AWS Batch MNP (Homogeneous)

Deploy and run PPO training for GR00T N1.5 on AWS Batch Multi-Node Parallel with homogeneous g6e.12xlarge instances (4× L40S per node). This is the `compute_backend=batch-mnp` path — a simpler, homogeneous alternative to the heterogeneous EKS + KubeRay backend (see `deploy-eks-training`).

## When to Use

- Deploying the Batch MNP homogeneous training stack from scratch
- Redeploying after a teardown
- Running a new training job on Batch MNP
- Debugging a failed Batch training job

## Prerequisites

- An AWS account + a target region with **g6e capacity** (e.g. `us-east-1`, `us-west-2`, `us-east-2`). Keep VPC + EFS + S3 + ECR all in that ONE region.
- vCPU quota: enough "Running On-Demand G and VT instances" for your fleet (g6e.12xlarge = 48 vCPUs × 2 nodes = 96). Request the increase before deploying.
- ECR image built via CodeBuild (or a pre-built tag)
- CDK dependencies installed: `pip install -r training/gr00t/rl/infra/requirements.txt`
- Local CLIs: `aws`, `cdk`, `jq`

Set the region once via env and reuse it across every command below (`app.py` fails closed if no region env var is set):
```bash
export AWS_REGION=<region> AWS_DEFAULT_REGION=<region> CDK_DEFAULT_REGION=<region>
export CDK_DEFAULT_ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
```

## S3 / EFS Data Layout

Batch MNP mounts EFS natively at `/mnt/efs`. The EFS stage-data CodeBuild project
(`GR00T-RL-Stage-EFS`, Step 3) clones the pinned third-party repos + model + workflows
into EFS. These versions are pinned for compatibility — do NOT bump the RLinf pin:
```
/mnt/efs/
├── third_party/
│   ├── RLinf/          (commit 649e757)
│   ├── Isaac-GR00T/    (commit 4af2b62)
│   ├── IsaacLab/       (commit 941ebdf4a)
│   └── IsaacLab-Arena/ (commit dba099565)
├── models/
│   └── GR00T-N1.5-RL-Rheo-AssembleTrocar/
└── workflows/
    └── rheo/scripts/   # i4h-workflows tree (simulation/rl/rlinf_ext + task cfg overlay)
```

## Steps

### 1. Deploy CDK Stack

```bash
cd training/gr00t/rl/infra

cdk deploy \
  --context compute_backend=batch-mnp \
  --context num_rollout_nodes=1 \
  --require-approval never
```

This creates: VPC, EFS, S3 bucket, CodeBuild projects (image build + EFS staging), Batch compute environment, job queue, job definition. Takes ~10-15 minutes.

To use a pre-built image (skip the CodeBuild image build):
```bash
cdk deploy \
  --context compute_backend=batch-mnp \
  --context image_uri=<account>.dkr.ecr.<region>.amazonaws.com/gr00t-rl-unified:<tag> \
  --require-approval never
```

### 2. Build Container Image (if not using pre-built)

Trigger CodeBuild to build the unified image:
```bash
aws codebuild start-build \
  --project-name GR00T-RL-Unified-Build \
  --region <region>
```

Use unique tags to bypass the ECS Docker cache:
```bash
TAG="v$(date +%Y%m%d%H%M%S)"
# After the build completes, redeploy with the new tag
cdk deploy \
  --context compute_backend=batch-mnp \
  --context image_uri=<account>.dkr.ecr.<region>.amazonaws.com/gr00t-rl-unified:${TAG}
```

### 3. Stage EFS Data

Trigger CodeBuild to clone repos and download the model to EFS:
```bash
aws codebuild start-build \
  --project-name GR00T-RL-Stage-EFS \
  --region <region>
```

This stages the tree shown in **S3 / EFS Data Layout** above (~10 min incl. the model download).

### 4. Submit Training Job

```bash
aws batch submit-job \
  --job-name gr00t-rl-mnp-training \
  --job-queue GR00T-RL-JobQueue \
  --job-definition $(aws batch describe-job-definitions --region <region> \
    --query 'jobDefinitions[?status==`ACTIVE`] | sort_by(@, &revision) | [-1].jobDefinitionArn' \
    --output text) \
  --region <region>
```

### 5. Monitor Training

```bash
# Job status
aws batch describe-jobs --jobs <JOB_ID> --region <region> \
  --query 'jobs[0].{status:status,reason:statusReason}'

# CloudWatch logs (may lag 5-15 min)
aws logs filter-log-events --log-group-name /aws/batch/job \
  --filter-pattern "Rollout Epochs" \
  --start-time $(python3 -c "import time; print(int((time.time()-600)*1000))") \
  --region <region> --query 'events[-5:].message' --output text

# SSM into instances (when CloudWatch lags)
INSTANCE_ID=$(aws ec2 describe-instances --region <region> \
  --filters "Name=instance-state-name,Values=running" "Name=instance-type,Values=g6e.12xlarge" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)

aws ssm send-command --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["docker exec $(docker ps -q | head -1) nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"]' \
  --region <region>

# Check TensorBoard file size (>88 bytes = training data written)
aws ssm send-command --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["docker exec $(docker ps -q | head -1) find /mnt/efs/rl-training/results -name events* -exec ls -la {} \\;"]' \
  --region <region>

# Check Ray worker logs (where training details live)
aws ssm send-command --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["docker exec $(docker ps -q | head -1) find /tmp/ray/session_latest/logs -name worker-*.out -exec ls -la {} \\; | sort -k5 -rn | head -5"]' \
  --region <region>
```

### 6. Teardown

```bash
cdk destroy --context compute_backend=batch-mnp --force
```

Note: may fail on subnets if EFS mount targets exist. Delete mount targets first or retain the subnets:
```bash
aws cloudformation delete-stack --stack-name GR00TRLBatchStack --region <region> \
  --retain-resources VPCPrivateSubnet1Subnet8BCA10E0 VPCPrivateSubnet2SubnetCFCDAA7A VPCPublicSubnet1SubnetB4246D30
```

## Architecture

```
AWS Batch MNP (2× g6e.12xlarge)
├── Node 0 (Main Node, 4× L40S)
│   ├── Ray Head
│   ├── FSDP Actor (GPUs 0-3)
│   └── entrypoint.sh → train_embodied_agent.py
└── Node 1 (Child Node, 4× L40S)
    ├── Ray Worker
    ├── Isaac Sim EnvWorkers (8 envs per worker)
    └── MultiStepRolloutWorkers (GPUs 4-7)

Storage: EFS (mounted natively by Batch at /mnt/efs)
Networking: NCCL over TCP (no EFA, NCCL_IB_DISABLE=1)
```

## Timing Profile

| Phase | Duration |
|-------|----------|
| CDK deploy | ~10-15 min |
| Image build (CodeBuild) | ~20 min |
| EFS staging (CodeBuild) | ~10 min |
| Instance provisioning | ~3-5 min |
| Image pull + container start | ~2-3 min |
| Ray cluster formation | ~1 min |
| Isaac Sim init + scene creation | ~3 min |
| Rollout (8 epochs, 64 envs) | ~27 min |
| Training step (4 update epochs) | ~90 min |
| **Total per PPO iteration** | **~117 min (~2 hrs)** |
| Checkpoint save | At step 2 (after iter 2) |

## Key Config

| Parameter | Value | Notes |
|-----------|-------|-------|
| Instance type | g6e.12xlarge | 4× L40S, 384 GB RAM |
| Nodes | 2 | 1 learner + 1 rollout |
| cluster.num_nodes | 2 | Set in entrypoint |
| component_placement | actor: 0-3, env,rollout: 4-7 | Set in YAML (not CLI — comma in key breaks Hydra) |
| total_num_envs | 64 | From YAML default |
| micro_batch_size | 32 | Prevents VRAM OOM |
| gradient_checkpointing | True | Trades compute for memory |
| max_epochs | 1000 | Stop manually for validation |
| save_interval | 2 | Checkpoint every 2 iterations |
| TORCHDYNAMO_DISABLE | 1 | torch.compile deadlocks with Isaac Sim |
| NCCL_IB_DISABLE | 1 | No EFA, use TCP sockets |
| NCCL_SOCKET_IFNAME | eth0 | Explicit NIC for NCCL |
| RLINF_EXT_MODULE | rlinf_ext | Extension module for Ray actors |
| cpu_offload | False | NEVER enable — causes NCCL deadlock |

## Known Issues & Fixes (encountered during bring-up)

| # | Issue | Fix |
|---|-------|-----|
| 1 | `rlinf_ext` not propagating to Ray actors | `export RLINF_EXT_MODULE=rlinf_ext` in entrypoint |
| 2-8 | Missing Python deps (onnxruntime, pin-pink, lightwheel-sdk, etc.) | Added to Docker image |
| 9 | Isaac-GR00T version mismatch | Pin to commit `4af2b62` |
| 10 | GPU OOM on g6e.4xlarge | Switched to g6e.12xlarge |
| 11-13 | Hydra config issues | Correct script path, config-path, fsdp.yaml in searchpath |
| 14 | Only 1 node detected | Pass `cluster.num_nodes` override |
| 15 | Cached Docker image | Use unique version tags per deploy |
| 16 | Main node IP detection | Use `AWS_BATCH_JOB_NODE_INDEX` not `PRIVATE_IPV4_ADDRESS` |
| 17-18 | Ray cluster not forming | Start Ray explicitly in entrypoint, use full path |
| 19 | `rlinf_ext` not importable in actors | Add `simulation/rl` to PYTHONPATH |
| 20 | torch.compile hangs 8+ hours | `TORCHDYNAMO_DISABLE=1` |
| 21 | CUDA OOM during training step | `micro_batch_size=32` + `gradient_checkpointing=True` |
| 22 | Memory corruption after OOM | Separated component placement across nodes |
| 23 | Hydra can't parse comma in key | Set placement in YAML, not CLI |
| 24 | FSDP NCCL deadlock | Remove `cpu_offload=True` |
| 25 | System RAM OOM after 2 iterations | Set `RAY_memory_usage_threshold=0.99` or use a larger instance (see Known Limitation below) |

## Example Training Metrics (single validation iteration — illustrative)

| Metric | Value |
|--------|-------|
| env/success_once | 0.703 (trocar assembly success) |
| env/return | 3.66 |
| train/actor/policy_loss | -0.070 |
| train/actor/grad_norm | 6.71 |
| train/critic/value_loss | 0.068 |
| time/step | ~7023 sec (~117 min) |

## Monitoring Protocol

**IMPORTANT: Don't trust CloudWatch alone.** CloudWatch log delivery lags 5-15 minutes for Batch MNP jobs. Always SSM into instances to verify:

1. Poll job status every 5-10 min
2. If CloudWatch logs stop but the job is RUNNING → SSM in immediately
3. Check `/tmp/ray/session_*/logs/worker-*.out` for real-time progress
4. Check `nvidia-smi` for GPU utilization (0% = likely deadlocked)
5. Check the TensorBoard events file size (88 bytes = just header, no data)
6. Use unique image tags for EVERY iteration (bypass the Docker cache)

## Known Limitation — RAM OOM on long runs

On g6e.12xlarge the job runs ~2 iterations + saves a checkpoint, then can crash on
iteration 3: system RAM climbs to ~342 GB / 360 GB (95%) on the rollout node and
Ray's OOM monitor kills the FSDP actors. Mitigations:

1. `export RAY_memory_usage_threshold=0.99` in the entrypoint (quick)
2. Switch to a larger-RAM instance, e.g. g6e.48xlarge (768 GB RAM) — solves it permanently
3. Add `gc.collect()` between iterations
4. Reduce the FSDP actor count per node

For long training runs, the heterogeneous EKS backend (`deploy-eks-training`) has no such per-node RAM ceiling.

## Related Files

- CDK stack: `training/gr00t/rl/infra/mnp_batch_stack.py`
- App routing: `training/gr00t/rl/infra/app.py`
- Entrypoint: `training/gr00t/rl/docker/entrypoint.sh`
- Training config: `workflows/rheo/scripts/simulation/rl/rlinf_ext/config/isaaclab_ppo_gr00t_assemble_trocar.yaml`

## EFS Output Locations

| Path | Content |
|------|---------|
| `/mnt/efs/rl-training/results/isaaclab_ppo_gr00t_assemble_trocar/<timestamp>/tensorboard/` | TensorBoard events |
| `/mnt/efs/rl-training/results/isaaclab_ppo_gr00t_assemble_trocar/<timestamp>/*.pt` | Checkpoints |

## Comparison with EKS Backend

| | Batch MNP | EKS (KubeRay) |
|--|-----------|---------------|
| Deploy command | `--context compute_backend=batch-mnp` | `--context compute_backend=eks` |
| Instance topology | 2× g6e.12xlarge (homogeneous) | 1× g6e.48xlarge + N× g6e.8xlarge |
| GPUs | 8 (4+4) | 8 (learner) + 1 per rollout worker |
| Total envs | 64 | 128 (default 4 workers × 32) |
| Per-iteration time | ~2 hrs | ~4 hrs |
| Long-run RAM OOM | Yes (at iter 3, see Known Limitation) | No per-node RAM ceiling |
| Ray management | Manual in entrypoint | KubeRay operator |
| Monitoring | SSM + CloudWatch | kubectl logs/exec + CloudWatch |
| Cost/hr | ~$16 | ~$30 |
