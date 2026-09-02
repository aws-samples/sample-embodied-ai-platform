---
name: deploy-eks-training
description: Deploy and run GR00T N1.5 PPO RL post-training on EKS + KubeRay (heterogeneous g6e learner + rollout fleet, FSx for Lustre backed by S3). Covers the two-stack build/stage/deploy flow, train and standalone-eval modes, monitoring, and teardown.
---

# Deploy GR00T RL Training on EKS (Heterogeneous)

Deploy and run PPO training for GR00T N1.5 on EKS with KubeRay using heterogeneous instance types, backed by FSx for Lustre.

## When to Use

- Deploying the EKS heterogeneous training stack from scratch
- Redeploying after a teardown
- Running a new training job on EKS
- Debugging a failed EKS training deployment

## Prerequisites

- AWS account with a target region that has **g6e capacity** (e.g. us-east-2 — see "Choosing a region" below). p5/p5e is **OPTIONAL**: the learner defaults to an on-demand `g6e.48xlarge` (8× L40S) and only needs a Capacity Block / p5 if you pass `--context capacity_reservation_id=<cr-id>`.
- vCPU quota: enough "Running On-Demand G and VT instances" (quota code `L-DB2E81BA`) for your fleet — e.g. a benchmark eval on 8× g6e.8xlarge = 256 vCPUs; a training learner (g6e.48xlarge = 192 vCPU) + rollout fleet is larger. Request the increase before deploying.
- ECR image `<account>.dkr.ecr.<region>.amazonaws.com/gr00t-rl-unified@sha256:<digest>`, built + pushed by the `GR00T-RL-Pipeline` CodeBuild project (owned by `GR00TRLArtifactsStack`), driven by `infra/prepare-artifacts.sh` (Steps 1-2) — or by the manual `scripts/build_unified_and_push.sh` self-build (bring-your-own image). The workload deploys the wrapper-resolved `@sha256` digest, NOT the mutable `:latest` tag.
- S3 bucket in the SAME region as FSx with staged training data (DRA-linked to FSx) — staged by the same `GR00T-RL-Pipeline` project's stage-data arm, driven by `infra/prepare-artifacts.sh` (Step 2)
- VPC with NAT gateway (private subnets need egress) — ideally ≥2 private subnets in different AZs so you can chase g6e capacity. `vpc_id` (bring your own VPC) is the primary path. **Fresh account with no VPC?** Deploy the standalone `GR00TRLNetworkStack` once to create an EKS-ready VPC (≥2 AZs, private+public subnets, NAT egress, `kubernetes.io/role/*` subnet tags): `cdk deploy GR00TRLNetworkStack --context compute_backend=network --context s3_data_bucket=<bucket>` (default CIDR `10.73.0.0/16`; override with `--context vpc_cidr`). Read the `VpcId` / `PrivateSubnetIds` / per-AZ `PrivateSubnetId{N}`+`PrivateSubnetAz{N}` outputs, then pass `--context vpc_id=<VpcId>` to the EKS deploy (feed a `PrivateSubnetIdN` to `fsx_subnet_id`/`rollout_subnet_ids`/`eval_learner_subnet_ids` to pin capacity by AZ). Caveats: it **creates a NAT gateway (a running charge)** and the VPC is **`RemovalPolicy.RETAIN`** — an EKS teardown never deletes it, so remove the VPC by hand when fully done.
- **Runtime asset egress** (NAT egress above covers all; no credentials required in the validated smoke, but availability is a real dependency): (a) `omniverse-content-production.s3-us-west-2.amazonaws.com` — the trocar scene/props USD (CC-BY-NC-4.0, NonCommercial); (b) `api.lightwheel.net` + `api-s3-assets.lightwheel.net` — Arena's *generic* object library calls the LightWheel SDK at import time (transitive via the trocar task's teleop import), returned HTTP 200 anonymously. `lightwheel-sdk` (public PyPI, Apache-2.0) is baked into the image; no LightWheel account is required.
- CDK dependencies: `pip install -r training/gr00t/rl/infra/requirements.txt`
- kubectl + `jq` installed locally (`prepare-artifacts.sh` parses build JSON with `jq`)

### Choosing a region

You may deploy to ANY region that has **g6e capacity** (learner default, rollout, and eval all
run on g6e). p5/p5e is **OPTIONAL** and only matters if you pass `--context
capacity_reservation_id=<cr-id>` to run the learner on reserved p5 (H100) capacity. Keep
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
  --context artifacts_only=true \
  --context s3_data_bucket=<bucket>
```

`artifacts_only=true` is **required** in this phase: it tells `app.py` to synth ONLY the
artifacts stack. Without it (and without an `image_uri`), the app **fails closed** — the EKS
stack is a pure image consumer and refuses to synth against a nonexistent/`:latest` image.

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
- `--context rollout_instance_type=g6e.8xlarge` (default — the rollout worker pod requests 24 vCPU / 200 GiB, which does NOT fit g6e.4xlarge's 16 vCPU / 128 GiB, so pods stay `Pending`; g6e.8xlarge = 32 vCPU / 256 GiB)
- `--context fsx_capacity_gib=1200` (default, minimum for PERSISTENT_2)
- `--context num_rollout_workers=4` (default)
- `--context eval_total_envs=N` (eval only — overrides the yaml default of 64; MUST be divisible by `num_nodes = 1 + num_rollout_workers`)

The EKS deploy creates: EKS cluster, FSx for Lustre (PERSISTENT_2) + DRA, GPU node groups,
KubeRay operator, NVIDIA device plugin, FSx CSI driver, RayCluster CR. Takes ~25-30 minutes.

**Manual alternative (bring-your-own image):** run from `training/gr00t/rl` (NOT `infra/` — the
script's `docker build` context is `docker/` relative to that dir):
```bash
cd training/gr00t/rl        # if you were in infra/, `cd ..`
scripts/build_unified_and_push.sh --region <region>
```
It does a local `docker build`/`push` to the same `gr00t-rl-unified` repo and prints the
`--context image_uri=<repo>@sha256:<digest>` line to hand to `cdk deploy` (or to
`prepare-artifacts.sh --mode data --image-uri <...>`).

### 2.5 Canonical end-to-end run (validated): smoke-eval → train → eval → teardown

The cheapest way to prove the whole path before a real run. Each step re-deploys the **same**
`GR00TRLEKSStack` with different context (mode-switch in place); the image + data were built/staged
in Steps 1-2. Configure kubectl (Step 3) first so you can watch logs and locate the checkpoint.
Set `IMG=<account>.dkr.ecr.<region>.amazonaws.com/gr00t-rl-unified@sha256:<digest>`.

```bash
cd training/gr00t/rl/infra

# Common contexts for EVERY deploy + the teardown — INCLUDING the subnet pins, so a
# redeploy never relocates FSx or churns node groups. Use the SAME subnets as Step 2.
CTX=(--context compute_backend=eks
     --context vpc_id=<vpc> --context s3_data_bucket=<bucket> --context image_uri=$IMG
     --context rollout_instance_type=g6e.8xlarge
     --context fsx_subnet_id=<subnet> --context rollout_subnet_ids=<subnet> --context eval_learner_subnet_ids=<subnet>)

# (a) SMOKE-EVAL the base model — 2-pod fleet, 8 envs (8 episodes; eval_rollout_epoch is hardcoded 1). ~$9/hr.
cdk deploy GR00TRLEKSStack "${CTX[@]}" \
  --context mode=eval --context num_rollout_workers=1 --context eval_total_envs=8 --force

# (b) TRAIN a cost-bounded 2-step run. save_interval=2 → a checkpoint at global_step_2, then stop.
#     On-demand g6e.48xlarge learner (no Capacity Block needed).
cdk deploy GR00TRLEKSStack "${CTX[@]}" \
  --context mode=train --context max_epochs=2 --context envs_per_worker=8 --force
kubectl logs -n training -l ray.io/node-type=head -f    # watch for global_step_2 + checkpoint save

# (c) LOCATE the checkpoint (the actor .pt FILE, not the dir):
kubectl exec -n training <head-pod> -- \
  find /mnt/fsx/rl-training/results -path '*/global_step_2/actor/model_state_dict/full_weights.pt'

# (d) EVAL that checkpoint — pass the FULL FSx .pt path from (c):
cdk deploy GR00TRLEKSStack "${CTX[@]}" \
  --context mode=eval --context num_rollout_workers=1 --context eval_total_envs=8 \
  --context eval_ckpt=<the full_weights.pt path from step c> --force

# (e) TEARDOWN — delete the RayCluster, then destroy with the SAME CTX (subnets included).
kubectl delete raycluster gr00t-rl-training -n training --ignore-not-found
cdk destroy GR00TRLEKSStack "${CTX[@]}" --force    # see Step 7 for the LAST-RESORT hang path
```

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
# Nodes Ready. Expect `1 + num_rollout_workers` GPU nodes: the smoke config
# (num_rollout_workers=1) is 2 nodes; train default (num_rollout_workers=4) is
# 1 learner + 4 rollout = 5 nodes.
kubectl get nodes

# RayCluster + all training pods Running (head + `num_rollout_workers` worker pods)
kubectl get pods -n training

# Verify FSx mount
kubectl exec <head-pod> -n training -- ls /mnt/fsx/third_party/
```

### 6. Monitor Training

```bash
# Head pod logs (training progress) — live, but EPHEMERAL (gone once the pod/cluster is torn down)
kubectl logs -n training <head-pod> -f

# PERSISTENT logs in CloudWatch (on by default via the amazon-cloudwatch-observability add-on;
# survives teardown, unlike kubectl logs). Log group /aws/containerinsights/gr00t-rl-eks/application:
aws logs tail /aws/containerinsights/gr00t-rl-eks/application --region <region> --follow
aws logs tail /aws/containerinsights/gr00t-rl-eks/application --region <region> \
  --filter-pattern "success_once" --since 2h        # just the eval metric
# Disable the add-on at deploy with --context enable_cloudwatch_logs=false (30-day retention when on).
# This is EKS POD logging — separate from control-plane logging (which wouldn't capture pod stdout).

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
       IMAGE_URI=<acct>.dkr.ecr.us-east-2.amazonaws.com/<repo>@sha256:<digest>
# Dry-run (safe): prints the full plan, spends $0
./eval-checkpoint.sh --backend eks --ckpt s3://<bucket>/<...>/global_step_N/actor/model_state_dict/full_weights.pt --n 64
# Real (PAID): add --execute (then type 'eval-checkpoint')
```

**Three gotchas the script now handles for you:**

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
kubectl delete raycluster gr00t-rl-training -n training --ignore-not-found

# Then CDK destroy. `cdk destroy` SYNTHS the app first, so the EKS backend's required
# contexts must be present (compute_backend, vpc_id, s3_data_bucket, and an image_uri —
# the app fails closed without one; there is no :latest fallback).
AWS_REGION=<region> CDK_DEFAULT_REGION=<region> cdk destroy GR00TRLEKSStack \
  --context compute_backend=eks --context vpc_id=<vpc> \
  --context s3_data_bucket=<bucket> --context image_uri=<the digest you deployed with> --force

# LAST RESORT ONLY — if `cdk destroy` hangs on the kubectl custom resources. The raw
# `delete-cluster` alone ORPHANS the CloudFormation stack, so you MUST follow it with
# `delete-stack` to clean up (do not stop after the first line):
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
├── Worker Pod ×4 (g6e.8xlarge, 1× L40S each)
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
| Training step (4 update epochs, mbs 32) | ~2.5-3 hrs |
| **Total per PPO iteration** | **~4 hrs** |
| Checkpoint save | Every 2 iterations |

## Key Config

Training parameters are head-pod env vars. The ones marked **`--context`** are exposed
as CDK context params (set at deploy, e.g. `--context max_epochs=5`); the rest are the
entrypoint's built-in defaults (change them by editing the entrypoint/manifest).

| Env Variable | Default | Notes |
|-------------|---------|-------|
| MICRO_BATCH_SIZE | 32 | L40S-safe value (mbs 64 AND 128 both OOM the 44GB L40S — 2026-06-15 benchmark). 128 (no grad-checkpoint) needs H100/p5 80GB. Sync entrypoint default is already 32. |
| GRADIENT_CHECKPOINTING | True | Must be True with batch 64+ on L40S. |
| ENVS_PER_WORKER | 32 | **`--context envs_per_worker`**. Environments per rollout worker pod (`total_num_envs = num_rollout_workers × this`). |
| MAX_EPOCHS | 1000 | **`--context max_epochs`**. Bounds the run to N global_steps — set low (e.g. 2-5) for a cost-bounded plumbing/validation run. Pair with SAVE_INTERVAL so a short run still writes an eval-able checkpoint. |
| SAVE_INTERVAL | 2 | Checkpoint every N iterations |
| CONFIG_NAME | isaaclab_ppo_gr00t_assemble_trocar | Hydra config name |
| MODEL_PATH | /mnt/fsx/models/GR00T-N1.5-RL-Rheo-AssembleTrocar | Pre-trained model path |

## Pinned Versions (Compatibility)

| Dependency | Commit/Version | Notes |
|-----------|---------------|-------|
| RLinf | `649e757` | Does NOT require weight_syncer |
| Isaac-GR00T | `4af2b62` | Has `gr00t.experiment.data_config` |
| IsaacLab | `941ebdf4a` (`941ebdf4ad1fbf89018777012bdfa4b5944c758f`) | Pinned in `infra/stage-s3-eks.sh` and `docker/Dockerfile.unified` — NOT a floating "latest" |
| IsaacLab-Arena | `dba099565` | Pinned in `infra/stage-s3-eks.sh` and `docker/Dockerfile.unified` |
| i4h-workflows | `fb7727e` (tag v0.5.0) | Full `rheo/scripts` tree cloned at stage time |
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
| Workers stuck after head restart (orphaned on old Ray GCS id) | **Fixed in the manifest**: the worker container now runs KubeRay's `ray start` **with `--block`** (`eval $KUBERAY_GEN_RAY_START_CMD`), so it self-heals — when the head restarts and re-keys the GCS, the worker's `ray start` exits and KubeRay recreates it against the current head. (Previously the manifest stripped `--block` + `sleep infinity`, which orphaned the worker and required a manual `kubectl delete pods -n training -l ray.io/node-type=worker`. That manual recycle is still a valid fallback if you see `1/N nodes connected` persist.) |
| CUDA OOM (learner) with batch 64 or 128 on L40S | Use MICRO_BATCH_SIZE=32 + GRADIENT_CHECKPOINTING=True (only 32 fits the 44GB L40S; 64/128 need H100/p5) |
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
| Changed the patch / pins / workflows and need to re-stage | If you edited files **locally**, first **redeploy `GR00TRLArtifactsStack`** — the `GR00T-RL-Pipeline` project builds from an **immutable source asset bundled at synth time**, so re-running the pipeline without a redeploy re-stages the OLD source. Then re-run the stage-data arm: `./prepare-artifacts.sh --region <region> --mode data` (or `aws codebuild start-build --project-name GR00T-RL-Pipeline --environment-variables-override name=STAGE_MODE,value=stage-data,type=PLAINTEXT --region <region>`); `aws s3 sync --delete` converges the bucket to the new tree |
| CodeBuild staging fails with S3 `AccessDenied` | If the data bucket uses a customer-managed KMS key, grant the `GR00T-RL-Pipeline` role `kms:Encrypt`/`kms:Decrypt`/`kms:GenerateDataKey` on that key (the bucket `grant_read_write` alone does not cover a CMK), or use SSE-S3 |
| **Before public release:** EKS is pinned to Kubernetes 1.31 | k8s 1.31 is in AWS EKS **extended support** (ends 2026-11-26). Bump `eks.KubernetesVersion` in `infra/eks_kuberay_stack.py` off `V1_31` to a standard-support version (and re-validate KubeRay/addon compatibility) before a public release |
| **Before public release:** workload should pin the image by digest | The EKS workload must run the wrapper-resolved `@sha256:<digest>` (what `prepare-artifacts.sh` verifies + hands to `cdk deploy`), NOT the mutable `:latest` tag — `:latest` can drift to an unverified image. Confirm the deployed `image_uri` is a `@sha256` reference, not a tag |

## Related Files

- CDK stack: `training/gr00t/rl/infra/eks_kuberay_stack.py`
- App routing: `training/gr00t/rl/infra/app.py`
- EKS entrypoint: `training/gr00t/rl/docker/entrypoint-eks.sh`
- Training config (on S3/FSx): `workflows/rheo/scripts/simulation/rl/rlinf_ext/config/isaaclab_ppo_gr00t_assemble_trocar.yaml`
