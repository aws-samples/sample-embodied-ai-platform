# Deploy GR00T RL Training on EKS (Heterogeneous)

Deploy and run PPO training for GR00T N1.5 on EKS with KubeRay using heterogeneous instance types, backed by FSx for Lustre.

## When to Use

- Deploying the EKS heterogeneous training stack from scratch
- Redeploying after a teardown
- Running a new training job on EKS
- Debugging a failed EKS training deployment

## Prerequisites

- AWS account with a target region that has p5/p5e + g6e capacity (e.g. us-east-2 — see "Choosing a region" below)
- vCPU quota: 384+ for G instances (quota code `L-DB2E81BA`)
- ECR image `<account>.dkr.ecr.<region>.amazonaws.com/gr00t-rl-unified@sha256:<digest>`, built + pushed by the `GR00T-RL-Pipeline` CodeBuild project (owned by `GR00TRLArtifactsStack`), driven by `infra/prepare-artifacts.sh` (Steps 1-2) — or by the manual `scripts/build_unified_and_push.sh` self-build (bring-your-own image). The workload deploys the wrapper-resolved `@sha256` digest, NOT the mutable `:latest` tag.
- S3 bucket in the SAME region as FSx with staged training data (DRA-linked to FSx) — staged by the same `GR00T-RL-Pipeline` project's stage-data arm, driven by `infra/prepare-artifacts.sh` (Step 2)
- VPC with NAT gateway (private subnets need egress) — ideally ≥2 private subnets in different AZs so you can chase g6e capacity
- **Runtime asset egress:** worker pods stream the trocar USD scene/props from the **public** NVIDIA Omniverse CDN (`omniverse-content-production.s3-us-west-2.amazonaws.com`) at run time — the NAT egress above covers it, no credentials. Those assets are CC-BY-NC-4.0 (NonCommercial). `lightwheel-sdk` (public PyPI, Apache-2.0) is baked into the image; no LightWheel account is required.
- CDK dependencies: `pip install -r training/gr00t/rl/infra/requirements.txt`
- kubectl + `jq` installed locally (`prepare-artifacts.sh` parses build JSON with `jq`)

### Choosing a region

You may deploy to ANY region that has p5/p5e (learner) + g6e (rollout/eval) capacity. Keep
S3 + ECR + VPC + FSx all in that ONE region — the FSx Data Repository Association (DRA)
requires the S3 bucket to be same-region. PROBE capacity first with `infra/capacity-probe.sh`
before deploying. Recommended regions where this was validated / capacity tends to exist:
`us-east-1`, `us-west-2`, `us-east-2`. Set the region once via env (`AWS_REGION` /
`CDK_DEFAULT_REGION` / `AWS_DEFAULT_REGION`) and reuse it across every command below —
`app.py` fails closed if no region env var is set (no hardcoded default).

## S3 Data Layout

The bucket is populated by the `GR00T-RL-Pipeline` stage-data arm (Step 2, driven by
`infra/prepare-artifacts.sh`) — you should not have to build this layout by hand. Under
the hood it runs `infra/stage-s3-eks.sh`, which produces exactly this tree (FSx-Lustre
lazily imports it via the DRA on first access):
```
s3://<bucket>/
├── third_party/
│   ├── RLinf/          (commit 649e757, _broadcast patch applied by stage-s3-eks.sh)
│   ├── Isaac-GR00T/    (commit 4af2b62)
│   ├── IsaacLab/       (commit 941ebdf4a)
│   └── IsaacLab-Arena/ (commit dba099565)
├── models/
│   └── GR00T-N1.5-RL-Rheo-AssembleTrocar/   (model rev b54e142)
└── workflows/
    └── rheo/scripts/            # COMPLETE i4h-workflows tree, cloned by stage-s3-eks.sh @ v0.5.0 (fb7727e)
        ├── simulation/          # tasks/assemble_trocar, assets/, rl/rlinf_ext/ (+ our tuned config overlay)
        ├── policy/
        ├── teleop_devices/      # sibling pkg the task cfg imports — MUST be present (else ModuleNotFoundError)
        ├── utils/
        └── config/
```

These versions are pinned for compatibility. Do NOT bump the RLinf pin (see Known Issues). The
`workflows/rheo/scripts` tree is staged **whole** from a pinned commit of i4h-workflows (only our
`simulation/rl/rlinf_ext/config` overlay is layered on top) — a hand-picked subset silently drops
interdependent sibling packages and breaks env creation at import time.

## Steps

The build/stage/deploy flow is **two stacks, three phases** (deploy `GR00TRLArtifactsStack` →
build image + stage data via the `GR00T-RL-Pipeline` project → deploy the consumer
`GR00TRLEKSStack`). There is **NO deployment auto-trigger**: a direct `cdk deploy GR00TRLEKSStack`
WITHOUT completing Steps 1-2 is **unsupported** — in fact the stack **fails synth** without an
explicit `image_uri` (there is deliberately no floating `:latest` fallback). Always drive the image
build + data staging through `prepare-artifacts.sh`, which gates the EKS deploy on verified
artifacts and hands `cdk deploy` the resolved `@sha256` digest.

### 0. Set the region env (once, reused by every step)

```bash
cd training/gr00t/rl/infra
export AWS_REGION=<region> AWS_DEFAULT_REGION=<region> CDK_DEFAULT_REGION=<region>
export CDK_DEFAULT_ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
```

### 1. Deploy the persistent artifacts stack (ECR + gated pipeline)

`GR00TRLArtifactsStack` owns the persistent `gr00t-rl-unified` ECR repo and the single
mode-switched **`GR00T-RL-Pipeline`** CodeBuild project. Deploy it ONCE (it persists across
EKS teardowns):

```bash
cdk deploy GR00TRLArtifactsStack \
  --context compute_backend=eks \
  --context s3_data_bucket=<bucket>
```

`GR00T-RL-Pipeline` is a single project driven by `STAGE_MODE ∈ {build-image, stage-data, all}`:
`build-image` builds + pushes the unified image; `stage-data` runs `infra/stage-s3-eks.sh`
(clones the pinned third-party repos — RLinf `649e757`, Isaac-GR00T `4af2b62`, IsaacLab
`941ebdf4a`, IsaacLab-Arena `dba099565` — APPLIES the `_broadcast` patch, downloads the RL
model, stages the workflows, uploads to `s3://$S3_DATA_BUCKET/{third_party,models,workflows}/`,
and writes the `_STAGING_COMPLETE` marker last on full success); `all` runs both.

### 2. Build image + stage data, then deploy EKS with the verified digest

`infra/prepare-artifacts.sh` drives + gates the pipeline: it kicks `GR00T-RL-Pipeline`,
polls to completion, VERIFIES the artifacts actually landed (the pushed ECR image digest
and/or the `s3://.../_STAGING_COMPLETE` marker), and only then deploys `GR00TRLEKSStack`
with the resolved `@sha256` digest as `image_uri`. It fails closed at every step:

```bash
cd training/gr00t/rl/infra
./prepare-artifacts.sh --region <region> --mode all --deploy --deploy-stack GR00TRLEKSStack -- \
    --context vpc_id=<vpc> --context s3_data_bucket=<bucket> --context mode=eval \
    --context rollout_instance_type=g6e.8xlarge --context num_rollout_workers=1 \
    --context eval_total_envs=8 --context rollout_subnet_ids=<subnet> \
    --context eval_learner_subnet_ids=<subnet> --context fsx_subnet_id=<subnet>
```

Everything after `--` is forwarded verbatim to `cdk deploy`. Do **not** pass `image_uri=` or
`compute_backend=` there — the wrapper owns them (it rejects them if you try). `--mode`
selects the pipeline arm: `all` (build + stage), `image` (build only), or `data` (re-stage
only; requires `--image-uri <...@sha256:...>` or a resolvable existing digest to deploy with).
Omit `--deploy` to stop after verify and just print the exact deploy line.

Optional EKS `--context` overrides (append after `--`):
- `--context learner_instance_type=g6e.48xlarge` (default)
- `--context rollout_instance_type=g6e.4xlarge` (default)
- `--context fsx_capacity_gib=1200` (default, minimum for PERSISTENT_2)
- `--context num_rollout_workers=4` (default)

The EKS deploy creates: EKS cluster, FSx for Lustre (PERSISTENT_2) + DRA, GPU node groups,
KubeRay operator, NVIDIA device plugin, FSx CSI driver, RayCluster CR. Takes ~25-30 minutes.

**Manual alternative (bring-your-own image):** `scripts/build_unified_and_push.sh --region <region>`
does a local `docker build`/`push` to the same `gr00t-rl-unified` repo and prints the
`--context image_uri=<repo>@sha256:<digest>` line to hand to `cdk deploy` (or to
`prepare-artifacts.sh --mode data --image-uri <...>`).

### 3. Configure kubectl Access

```bash
# Use the role ARN from stack outputs (KubeconfigCommand)
aws eks update-kubeconfig --name gr00t-rl-eks --region <region> \
  --role-arn arn:aws:iam::<account>:role/gr00t-rl-eks-admin-<region>
```

### 4. Post-Deploy: entrypoint ConfigMap (created by CDK — no manual step)

The `entrypoint-eks` ConfigMap is **created by the CDK stack** — `eks_kuberay_stack.py` reads
`docker/entrypoint-eks.sh` at synth time and materializes the ConfigMap, which is mounted into the
Ray pods. You do **NOT** create it by hand; a manual `kubectl create configmap` is neither needed
nor recommended (it drifts from the committed entrypoint).

To pick up an **edited** entrypoint, re-deploy the stack (clean path). For quick in-place iteration,
recycle the workers once the head is stable:
```bash
# After a head restart, workers may still target the old head IP. Once the head is stable (~2 min):
kubectl delete pods -n training -l ray-role=worker   # KubeRay recreates them against the new head
```

### 5. Verify Cluster Health

```bash
# All 5 nodes Ready
kubectl get nodes

# RayCluster + all training pods Running
kubectl get pods -n training

# Verify FSx mount
kubectl exec <head-pod> -n training -- ls /mnt/fsx/third_party/
```

### 6. Monitor Training

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

### 6.5 Per-Stage Eval (eval-checkpoint.sh) — the NVIDIA-comparable 4-number row

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

### 7. Teardown

```bash
# Delete RayCluster first (avoids custom resource timeout)
kubectl delete raycluster gr00t-rl-training -n training

# Then CDK destroy
AWS_REGION=<region> CDK_DEFAULT_REGION=<region> cdk destroy GR00TRLEKSStack --force

# If CDK hangs on custom resources, force via API:
aws eks delete-cluster --name gr00t-rl-eks --region <region>
aws cloudformation delete-stack --stack-name GR00TRLEKSStack --region <region>
```

**Leave `GR00TRLArtifactsStack` standing.** It persists across EKS teardowns (it owns the built
image + the pipeline), and its ECR repo `gr00t-rl-unified` is created with a **RETAIN** policy —
destroying the stack orphans the repo, and a later re-deploy then collides on the name (the Batch
stack also references `gr00t-rl-unified`). An idle ECR repo + an untriggered CodeBuild project cost
essentially nothing. If you truly must remove it, delete the repo explicitly afterward:
`aws ecr delete-repository --repository-name gr00t-rl-unified --force --region <region>`.

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
| Ray | 2.47.0 | RayCluster `rayVersion` (RLinf minimum; matches the image's `ray[default]==2.47.0`) |
| lightwheel-sdk | 1.0.3 | Public PyPI (Apache-2.0); baked into the image so IsaacLab-Arena's `object_library` import resolves. Trocar USD assets load from the public Omniverse CDN (CC-BY-NC-4.0), not the SDK's authenticated API |

**Critical:** RLinf `bc3d8aa`+ requires `weight_syncer` config. Isaac-GR00T `3df8b38` lacks `data_config`. Always use the versions pinned in the i4h-workflows repo.

**Do NOT bump the RLinf pin off `649e757`.** A bump to `be8d5c2` was verified to break
training three ways: a now-mandatory `weight_syncer` config, an N1.5 module rename, and the
deletion of `eval_embodied_agent.py`. The `_broadcast` deadlock (below) is fixed by applying
`patches/RLinf-649e7579-broadcast-raise.patch` at stage time — NOT by moving the pin.

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
| Eval head crash-loops: `total_num_envs must be divisible...` | N must be divisible by `num_nodes=1+num_rollout_workers`; for N=64 use 7 workers (8 nodes). See §6.5 gotcha 2 |
| Eval head crash: `EVAL_CKPT file not found: s3://...` | Pass the FSx path, not the s3:// URI (or use eval-checkpoint.sh which auto-translates). §6.5 gotcha 1 |
| Multi-stage eval reports stage-1's number for all stages | The head must be reformed between stages so it re-reads `success_stage`; use eval-checkpoint.sh (handles it). §6.5 gotcha 3 |
| g6e-dry in the FSx AZ | Run eval/rollout cross-AZ via `EKS_ROLLOUT_SUBNET_IDS`/`EKS_EVAL_LEARNER_SUBNET_IDS`; FSx read cross-AZ. §6.5 |
| Silent ~3h hang; `ValueError: Unsupported object type` / `invalid load key` after a Gloo broadcast failure (`_broadcast` deadlock) | `stage-s3-eks.sh` applies `patches/RLinf-649e7579-broadcast-raise.patch` (RLinf swallows the broadcast exception and reads uninitialized buffers as pickle); the patch re-raises so training dies loudly and `auto-recover` restarts from the latest checkpoint. Keep RLinf pinned at `649e757` — do NOT bump the pin to fix this |
| Head pod restarts / can't find `/mnt/fsx/third_party` right after a first deploy | Data may not have landed yet. `prepare-artifacts.sh` gates the EKS deploy on staging (Step 2), so this only bites if you bypassed the wrapper. Confirm staging converged (the `GR00T-RL-Pipeline` stage-data arm, ~15-20 min incl. the ~5.5 GB model): `aws s3 ls s3://<bucket>/_STAGING_COMPLETE` (the marker is written last, only on full success). FSx imports lazily via the DRA; KubeRay self-heals the head until data lands |
| Changed the patch / pins / workflows and need to re-stage | Re-run the stage-data arm: `./prepare-artifacts.sh --region <region> --mode data` (or `aws codebuild start-build --project-name GR00T-RL-Pipeline --environment-variables-override name=STAGE_MODE,value=stage-data,type=PLAINTEXT --region <region>`); `aws s3 sync --delete` converges the bucket to the new tree |
| CodeBuild staging fails with S3 `AccessDenied` | If the data bucket uses a customer-managed KMS key, grant the `GR00T-RL-Pipeline` role `kms:Encrypt`/`kms:Decrypt`/`kms:GenerateDataKey` on that key (the bucket `grant_read_write` alone does not cover a CMK), or use SSE-S3 |
| **Before public release:** EKS is pinned to Kubernetes 1.31 | k8s 1.31 is in AWS EKS **extended support** (ends 2026-11-26). Bump `eks.KubernetesVersion` in `infra/eks_kuberay_stack.py` off `V1_31` to a standard-support version (and re-validate KubeRay/addon compatibility) before a public release |
| **Before public release:** workload should pin the image by digest | The EKS workload must run the wrapper-resolved `@sha256:<digest>` (what `prepare-artifacts.sh` verifies + hands to `cdk deploy`), NOT the mutable `:latest` tag — `:latest` can drift to an unverified image. Confirm the deployed `image_uri` is a `@sha256` reference, not a tag |

## Related Files

- CDK stack: `training/gr00t/rl/infra/eks_kuberay_stack.py`
- App routing: `training/gr00t/rl/infra/app.py`
- EKS entrypoint: `training/gr00t/rl/docker/entrypoint-eks.sh`
- Training config (on S3/FSx): `workflows/rheo/scripts/simulation/rl/rlinf_ext/config/isaaclab_ppo_gr00t_assemble_trocar.yaml`
