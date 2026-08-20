# Deploy GR00T RL Training on EKS (Heterogeneous)

Deploy and run PPO training for GR00T N1.5 on EKS with KubeRay using heterogeneous instance types, backed by FSx for Lustre.

## When to Use

- Deploying the EKS heterogeneous training stack from scratch
- Redeploying after a teardown
- Running a new training job on EKS
- Debugging a failed EKS training deployment

## Prerequisites

- AWS account with us-east-2 region
- vCPU quota: 384+ for G instances (quota code `L-DB2E81BA`)
- ECR image: `<account>.dkr.ecr.us-east-2.amazonaws.com/gr00t-rl-unified:<tag>`
- S3 bucket in us-east-2 with staged training data (DRA-linked to FSx)
- VPC with NAT gateway (private subnets need egress)
- CDK dependencies: `pip install -r training/gr00t/rl/infra/requirements.txt`
- kubectl installed locally

## S3 Data Layout

The S3 bucket must contain:
```
s3://<bucket>/
├── third_party/
│   ├── RLinf/          (commit 649e757)
│   ├── Isaac-GR00T/    (commit 4af2b62)
│   └── IsaacLab/
├── models/
│   └── GR00T-N1.5-RL-Rheo-AssembleTrocar/
└── workflows/
    └── rheo/scripts/simulation/rl/rlinf_ext/
```

These versions are pinned for compatibility. Use the i4h-workflows repo as source of truth.

## Steps

### 1. Deploy CDK Stack

```bash
cd /home/ubuntu/Documents/fraolotu/i4h-training-infra/training/gr00t/rl/infra

AWS_REGION=us-east-2 AWS_DEFAULT_REGION=us-east-2 CDK_DEFAULT_REGION=us-east-2 \
  CDK_DEFAULT_ACCOUNT=<account-id> cdk deploy \
  --context compute_backend=eks \
  --context vpc_id=<vpc-id> \
  --context s3_data_bucket=<bucket-name> \
  --context image_uri=<ecr-image-uri> \
  --require-approval never
```

Optional context overrides:
- `--context learner_instance_type=g6e.48xlarge` (default)
- `--context rollout_instance_type=g6e.4xlarge` (default)
- `--context fsx_capacity_gib=1200` (default, minimum for PERSISTENT_2)
- `--context num_rollout_workers=4` (default)

Creates: EKS cluster, FSx for Lustre (PERSISTENT_2) + DRA, GPU node groups, KubeRay operator, NVIDIA device plugin, FSx CSI driver, RayCluster CR. Takes ~25-30 minutes.

### 2. Configure kubectl Access

```bash
# Use the role ARN from stack outputs (KubeconfigCommand)
aws eks update-kubeconfig --name gr00t-rl-eks --region us-east-2 \
  --role-arn arn:aws:iam::<account>:role/gr00t-rl-eks-admin-us-east-2
```

### 3. Post-Deploy: Create Entrypoint ConfigMap

The entrypoint is mounted via ConfigMap (not baked into the image):

```bash
kubectl create configmap entrypoint-eks \
  --from-file=entrypoint-eks.sh=/path/to/training/gr00t/rl/docker/entrypoint-eks.sh \
  -n training --dry-run=client -o yaml | kubectl apply -f -
```

Then delete pods to pick up the new ConfigMap (KubeRay recreates them):
```bash
kubectl delete pods -n training --all
```

**Important:** After head restarts, workers may get stuck (connected to old head IP). Delete workers once head is stable:
```bash
# Wait ~2 min for head to stabilize, then:
kubectl delete pods -n training -l ray-role=worker
```

### 4. Verify Cluster Health

```bash
# All 5 nodes Ready
kubectl get nodes

# RayCluster + all training pods Running
kubectl get pods -n training

# Verify FSx mount
kubectl exec <head-pod> -n training -- ls /mnt/fsx/third_party/
```

### 5. Monitor Training

```bash
# Head pod logs (training progress)
kubectl logs -n training <head-pod> -f

# Rollout progress (filter Gloo noise)
kubectl logs <head-pod> -n training | grep "Generating Rollout"

# GPU utilization
kubectl exec <head-pod> -n training -- nvidia-smi

# GPU metrics log (dmon, written every 30s)
kubectl exec <head-pod> -n training -- tail /mnt/fsx/rl-training/results/*/gpu_metrics/gpu_dmon.csv

# TensorBoard file size (>88 bytes = PPO iteration completed)
kubectl exec <head-pod> -n training -- \
  find /mnt/fsx/rl-training/results -name 'events*' -exec ls -la {} \;

# Checkpoints
kubectl exec <head-pod> -n training -- \
  find /mnt/fsx/rl-training/results -name '*.pt'
```

### 5.5 Per-Stage Eval (eval-checkpoint.sh) — the NVIDIA-comparable 4-number row

`training/gr00t/rl/infra/eval-checkpoint.sh` runs the per-stage `success_stage` sweep
(N=64, Wilson 95% CI) on a checkpoint and prints per-stage `eval/success_once` vs the
reference row. It is **self-contained on `--backend eks`**: ONE `--execute` sweeps all
four stages autonomously (deploy once → per stage: patch `success_stage` → reform the
RayCluster → wait for the metric → read → teardown). Dry-run by default; `--execute` +
a typed confirmation to spend.

```bash
cd training/gr00t/rl/infra
export AWS_REGION=us-east-2 AWS_DEFAULT_REGION=us-east-2 CDK_DEFAULT_REGION=us-east-2 \
       CDK_DEFAULT_ACCOUNT=<acct> VPC_ID=<vpc> S3_DATA_BUCKET=<bucket> \
       IMAGE_URI=<acct>.dkr.ecr.us-east-2.amazonaws.com/<repo>:<tag>
# Dry-run (safe): prints the full plan, spends $0
./eval-checkpoint.sh --backend eks --ckpt s3://<bucket>/<...>/global_step_N/actor/model_state_dict/full_weights.pt --n 64
# Real (PAID): add --execute (then type 'eval-checkpoint')
```

**Three gotchas the script now handles for you (learned the hard way, 2026-08-20):**

1. **`--ckpt` may be `s3://<S3_DATA_BUCKET>/<key>` OR the FSx path.** The head entrypoint
   loads it as a LOCAL FSx path (`test -f`), NOT an s3 URI. The script auto-translates
   `s3://<S3_DATA_BUCKET>/<key>` → `/mnt/fsx/<key>` (DRA map). Passing a raw s3:// URI to
   the deploy directly (without the script) crashes the head: `EVAL_CKPT file not found`.
2. **`env.eval.total_num_envs` (N) MUST be divisible by `num_nodes` = `1 + num_rollout_workers`**
   (the eval places an env process on every node). Otherwise RLinf's `validate_embodied_cfg`
   asserts and the head **crash-loops** (never emits the metric; looks like a Ray-formation
   hang but isn't). For N=64 use a worker count from {1,3,7,15,31,63}; the script defaults to
   **7 workers (8 nodes)** and validates + lists valid counts if you change `--n`.
3. **`only_eval` head auto-restarts after each stage**, taking a NEW Ray GCS cluster-id that
   **orphans the workers** (they can't re-register: "GCS authentication error"). Between
   stages you must delete the Ray pods and let KubeRay reform, recycling not-Ready workers
   until all `num_nodes` re-register. The script's `reform_raycluster_eks` does this
   automatically; if you ever drive a sweep by hand, that's the per-stage step.

**Capacity note:** the eval-learner (head) NG is pinned to the FSx AZ by default. If that AZ
is g6e-capacity-dry, run cross-AZ: `EKS_ROLLOUT_SUBNET_IDS=<other-AZ-subnet>
EKS_EVAL_LEARNER_SUBNET_IDS=<other-AZ-subnet>` (FSx stays put, read cross-AZ). Probe first
with `capacity-probe.sh --subnet <subnet> --instance-type g6e.8xlarge --capacity 9`.

### 6. Teardown

```bash
# Delete RayCluster first (avoids custom resource timeout)
kubectl delete raycluster gr00t-rl-training -n training

# Then CDK destroy
AWS_REGION=us-east-2 CDK_DEFAULT_REGION=us-east-2 cdk destroy GR00TRLEKSStack --force

# If CDK hangs on custom resources, force via API:
aws eks delete-cluster --name gr00t-rl-eks --region us-east-2
aws cloudformation delete-stack --stack-name GR00TRLEKSStack --region us-east-2
```

## Architecture

```
EKS Cluster (gr00t-rl-eks)
├── Head Pod (g6e.48xlarge, 8× L40S)
│   ├── Ray Head
│   ├── FSDP Actor (8 GPU shards)
│   └── entrypoint-eks.sh → train_embodied_agent.py
├── Worker Pod ×4 (g6e.4xlarge, 1× L40S each)
│   ├── Ray Worker
│   └── Isaac Sim EnvWorker + RolloutWorker (32 envs)
│
Storage: FSx for Lustre (PERSISTENT_2) ←→ S3 via DRA
         Mounted at /mnt/fsx via FSx CSI driver
```

## Timing Profile

| Phase | Duration |
|-------|----------|
| CDK deploy | ~25 min |
| Image pull (first time) | ~5-10 min |
| Isaac Sim init + scene creation | ~5 min |
| Rollout (8 epochs, 128 envs) | ~60 min |
| Training step (4 update epochs, batch 64) | ~2.5-3 hrs |
| **Total per PPO iteration** | **~4 hrs** |
| Checkpoint save | Every 2 iterations |

## Key Config

Training parameters are **env-var configurable** on the head pod:

| Env Variable | Default | Notes |
|-------------|---------|-------|
| MICRO_BATCH_SIZE | 64 | L40S-safe. Use 128 only on H100 (80GB VRAM). |
| GRADIENT_CHECKPOINTING | True | Must be True with batch 64+ on L40S. |
| ENVS_PER_WORKER | 32 | Environments per rollout worker pod |
| MAX_EPOCHS | 1000 | Set to 5 for quick validation |
| SAVE_INTERVAL | 2 | Checkpoint every N iterations |
| CONFIG_NAME | isaaclab_ppo_gr00t_assemble_trocar | Hydra config name |
| MODEL_PATH | /mnt/fsx/models/GR00T-N1.5-RL-Rheo-AssembleTrocar | Pre-trained model path |

## Pinned Versions (Compatibility)

| Dependency | Commit/Version | Notes |
|-----------|---------------|-------|
| RLinf | `649e757` | Does NOT require weight_syncer |
| Isaac-GR00T | `4af2b62` | Has `gr00t.experiment.data_config` |
| IsaacLab | Latest from i4h-workflows | |
| Ray | 2.9.0 | KubeRay cluster version |

**Critical:** RLinf `bc3d8aa`+ requires `weight_syncer` config. Isaac-GR00T `3df8b38` lacks `data_config`. Always use the versions pinned in the i4h-workflows repo.

## Known Issues

| Issue | Fix |
|-------|-----|
| Head crashes on first boot (DRA lazy-load) | Entrypoint waits for config file; head auto-restarts and works on 2nd boot |
| Workers stuck after head restart | Delete workers: `kubectl delete pods -n training -l ray-role=worker` |
| CUDA OOM with batch 128 on L40S | Use MICRO_BATCH_SIZE=64 + GRADIENT_CHECKPOINTING=True |
| `Eagle2_5_VLImageProcessorFast` not found | Transient on first boot; resolves on restart |
| P5/P5e capacity unavailable | Fall back to g6e instances |
| S3 bucket must be same region as FSx | DRA requires same-region S3 |
| CDK lookup uses wrong region | Set `AWS_REGION=us-east-2` explicitly (not just CDK_DEFAULT_REGION) |
| Eval head crash-loops: `total_num_envs must be divisible...` | N must be divisible by `num_nodes=1+num_rollout_workers`; for N=64 use 7 workers (8 nodes). See §5.5 gotcha 2 |
| Eval head crash: `EVAL_CKPT file not found: s3://...` | Pass the FSx path, not the s3:// URI (or use eval-checkpoint.sh which auto-translates). §5.5 gotcha 1 |
| Multi-stage eval reports stage-1's number for all stages | The head must be reformed between stages so it re-reads `success_stage`; use eval-checkpoint.sh (handles it). §5.5 gotcha 3 |
| g6e-dry in the FSx AZ | Run eval/rollout cross-AZ via `EKS_ROLLOUT_SUBNET_IDS`/`EKS_EVAL_LEARNER_SUBNET_IDS`; FSx read cross-AZ. §5.5 |

## Related Files

- CDK stack: `training/gr00t/rl/infra/eks_kuberay_stack.py`
- App routing: `training/gr00t/rl/infra/app.py`
- EKS entrypoint: `training/gr00t/rl/docker/entrypoint-eks.sh`
- Training config (on S3/FSx): `workflows/rheo/scripts/simulation/rl/rlinf_ext/config/isaaclab_ppo_gr00t_assemble_trocar.yaml`
