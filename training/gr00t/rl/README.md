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

This EFS CodeBuild step is for the **batch-mnp / sagemaker** backends only — the EKS backend
uses FSx-Lustre backed by S3, so stage to S3 with `infra/stage-s3-eks.sh` (below) instead.

### Stage Training Data (EKS → S3/FSx)

The EKS backend reads its data from FSx-Lustre, which lazily imports from an S3 bucket via a
Data Repository Association. Staging that bucket is handled by the **`GR00T-RL-Stage-S3`
CodeBuild project**, which runs **automatically on `cdk deploy`** (via an `AwsCustomResource`
that triggers the build once). It clones the pinned third-party repos, **applies the RLinf
`_broadcast` patch**, downloads the model, stages the workflows, and uploads everything to
`$S3_DATA_BUCKET`. To re-run it any time:

```bash
aws codebuild start-build --project-name GR00T-RL-Stage-S3 --region <region>
```

Because staging auto-runs on deploy and FSx imports lazily via the DRA, no manual step is
required for a first deploy.

Under the hood CodeBuild runs `docker/buildspec-stage-s3.yml` → `infra/stage-s3-eks.sh
--execute --yes`. You can also run that script locally for dev/inspection — it is fail-closed
and dry-run by default (every `aws s3 sync` runs with `--dryrun` until you pass `--execute`):

```bash
cd training/gr00t/rl/infra
export S3_DATA_BUCKET=<your-s3-bucket> AWS_REGION=<region>   # same region as the bucket + FSx

# Dry-run (safe): clones/patches/downloads locally, prints what WOULD upload, writes $0 to S3
./stage-s3-eks.sh
# Real (writes to S3): add --execute (then type 'stage-s3-eks' to confirm)
./stage-s3-eks.sh --execute
```

You may deploy to any region with p5/p5e + g6e capacity; keep S3 + ECR + VPC + FSx all in that
ONE region (the FSx DRA requires same-region S3) and probe capacity first with `infra/capacity-probe.sh`.

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
- **Topology:** 1 head pod + N worker pods, all on `<rollout-instance>` (1× L40S each). Fleet size scales with the `num_rollout_workers` CDK context param — default is 1 (2 pods total, smoke config); benchmark eval at the yaml-default `total_num_envs=64` needs enough GPUs to stay inside a proven envelope on L40S-class hardware. Setting `num_rollout_workers=7` gives an 8-pod fleet at 8 envs/GPU across all 8 GPUs.
- **Invocation (smoke — matches the shipped smoke defaults):**

```bash
AWS_DEFAULT_REGION=us-east-2 cdk deploy GR00TRLEKSStack \
  --context compute_backend=eks \
  --context vpc_id=<your-vpc-id> \
  --context s3_data_bucket=<your-s3-bucket> \
  --context image_uri=<your-ecr-uri> \
  --context mode=eval \
  --context eval_ckpt=/mnt/fsx/<your-run-path>/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt
```

- **Invocation (benchmark eval at `total_num_envs=64`):**

```bash
AWS_DEFAULT_REGION=us-east-2 cdk deploy GR00TRLEKSStack \
  --context compute_backend=eks \
  --context vpc_id=<your-vpc-id> \
  --context s3_data_bucket=<your-s3-bucket> \
  --context image_uri=<your-ecr-uri> \
  --context mode=eval \
  --context num_rollout_workers=7 \
  --context eval_ckpt=/mnt/fsx/<your-run-path>/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt
```

Omit `--context eval_ckpt=...` when the model at `MODEL_PATH` is itself the RL-trained snapshot (no `.pt` overlay needed).

- **Prerequisites:** the rollout nodegroup can scale to `1 + num_rollout_workers` nodes total (head runs on the eval-learner NG, workers on the rollout NG). The training learner nodegroup can be at `desired=0` (no Capacity Block required).
- **Runtime:** benchmark eval (`num_rollout_workers=7`, `total_num_envs=64`, `eval_rollout_epoch=1`) runs ~15-20 min end-to-end per stage. Smoke eval (`num_rollout_workers=1`, `total_num_envs=8`, `eval_rollout_epoch=8`) also 64 episodes but on fewer GPUs, ~30-40 min.
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
- MODE=eval fails fast at pod startup if `eval_ckpt` points to a non-existent file. If `eval_ckpt` is omitted or empty, the pod runs base-model eval against the HF snapshot at `MODEL_PATH` (no RL overlay). Note: on the RLinf path (`eval_embodied_agent.py`), the model_path itself IS the payload — the class-selection to `GR00T_N1_5_ForRLActionPrediction` happens automatically, so no `.pt` overlay is needed when the snapshot at `MODEL_PATH` is already the RL-trained model.
- The eval topology is driven by Hydra overrides in `entrypoint-eks.sh` (`env.eval.total_num_envs=64, algorithm.eval_rollout_epoch=1, ++env.eval.ignore_terminations=True, ++env.eval.use_fixed_reset_state_ids=True, ++env.eval.max_episode_steps=256`) matching the yaml defaults. Component placement uses hardware-rank ranges (`env=0-N, rollout=0-N, actor=0-N` where `N = TOTAL_EXPECTED - 1`) to spawn one worker per Ray node — a scalar value like `env=1` would parse as "1 process on GPU rank 1" and pile every env onto a single GPU regardless of cluster size. To change env count, edit that override list.
- Video output size is roughly a few hundred MB per 64-episode run (256 frames per episode × 64 episodes).

### One-command multi-stage eval sweep (`eval-checkpoint.sh`)

`infra/eval-checkpoint.sh` runs the per-stage `success_stage` sweep (N=64, Wilson 95% CI) on a checkpoint and prints per-stage `eval/success_once` — the NVIDIA-comparable 4-number row. It is **self-contained on `--backend eks`**: ONE `--execute` sweeps all four stages autonomously (deploy once → per stage: patch `success_stage` → reform the RayCluster → wait for the metric → read → teardown). Dry-run by default; `--execute` plus a typed confirmation is required to spend.

```bash
cd training/gr00t/rl/infra
export AWS_REGION=us-east-2 AWS_DEFAULT_REGION=us-east-2 CDK_DEFAULT_REGION=us-east-2 \
       CDK_DEFAULT_ACCOUNT=<your-account> VPC_ID=<your-vpc-id> S3_DATA_BUCKET=<your-s3-bucket> \
       IMAGE_URI=<your-account>.dkr.ecr.us-east-2.amazonaws.com/<your-repo>:<tag>

# Dry-run (safe): prints the full plan, spends $0
./eval-checkpoint.sh --backend eks \
  --ckpt s3://<your-s3-bucket>/<your-run-path>/global_step_N/actor/model_state_dict/full_weights.pt \
  --n 64

# Real (PAID): add --execute (then type the confirmation string it prints)
./eval-checkpoint.sh --backend eks \
  --ckpt s3://<your-s3-bucket>/<your-run-path>/global_step_N/actor/model_state_dict/full_weights.pt \
  --n 64 --execute
```

**Prerequisite:** stage `infra/patch-success-stage.sh` to `s3://<your-s3-bucket>/scratch/step-a/patch-success-stage.sh` (the DRA maps it to `/mnt/fsx/scratch/step-a/`, where the script reads it per stage). It is committed here as the first-class copy of the runbook helper.

**Three gotchas the script handles for you:**

1. **`--ckpt` may be `s3://<your-s3-bucket>/<key>` OR the FSx path.** The head entrypoint loads the checkpoint as a LOCAL FSx path (`test -f`), NOT an s3 URI. The script auto-translates `s3://<your-s3-bucket>/<key>` → `/mnt/fsx/<key>` (the DRA map). Passing a raw `s3://` URI to `cdk deploy` directly (without the script) crashes the head with `EVAL_CKPT file not found`.
2. **`env.eval.total_num_envs` (N) MUST be divisible by `num_nodes` = `1 + num_rollout_workers`** (eval places one env process on every node). Otherwise RLinf's `validate_embodied_cfg` asserts and the head **crash-loops** (never emits the metric; looks like a Ray-formation hang but is not). For N=64 use a worker count from {1,3,7,15,31,63}; the script defaults to **7 workers (8 nodes)** and validates + lists valid counts if you change `--n`.
3. **Per-stage RayCluster reform is automatic.** The `only_eval` head auto-restarts after each stage and takes a NEW Ray GCS cluster-id that orphans the workers (they fail to re-register with a "GCS authentication error"). Between stages the workers must be recycled until all `num_nodes` re-register. The script's per-stage reform does this for you; driving a sweep by hand means repeating that step every stage.

**Cross-AZ capacity fallback:** the eval-learner (head) node group is pinned to the FSx AZ by default. If that AZ is g6e-capacity-dry, run the eval fleet in another AZ via `EKS_ROLLOUT_SUBNET_IDS=<other-AZ-subnet-id>` and `EKS_EVAL_LEARNER_SUBNET_IDS=<other-AZ-subnet-id>` (FSx stays put and is read cross-AZ; the static CSI PV has no topology affinity). Probe capacity first with `infra/capacity-probe.sh --subnet <subnet-id> --instance-type g6e.8xlarge --capacity 9`.

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

**Eval mode** (`--context mode=eval`) drops the training learner pod and uses a `(1 + num_rollout_workers)`-pod topology on `<rollout-instance>` (1× L40S each). Default is 2 pods (smoke); benchmark eval at the yaml-default `total_num_envs=64` typically uses `--context num_rollout_workers=7` (8 pods total, 8 envs/GPU). No p5.48xlarge / no Capacity Block required. The head pod runs `entrypoint-eks.sh → eval_embodied_agent.py` and writes MP4 rollouts to `${LOG_DIR}/video/eval/`. See the [Modes](#modes) section above for both smoke and benchmark invocations.

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
