# GR00T RL Post-Training on AWS

Reinforcement learning post-training for NVIDIA GR00T N1.5 on the Assemble Trocar surgical task using PPO (via RLinf). **EKS + KubeRay is the recommended, validated backend for real runs**; AWS Batch MNP is a simpler secondary option for quick experiments.

## Compute Backends

| Backend | Command | Instances | Best For |
|---------|---------|-----------|----------|
| **EKS + KubeRay** (heterogeneous) — *recommended* | `--context compute_backend=eks` | 1× g6e.48xlarge (8× L40S) + 4× g6e.8xlarge (1× L40S each), configurable | Real runs: better GPU utilization, no RAM OOM |
| **AWS Batch MNP** (homogeneous) | `--context compute_backend=batch-mnp` | 5× g6e.12xlarge (1 learner + 4 rollout, 4× L40S each; default `num_rollout_nodes=4`) | Simple setup — quick experiments only (~$52/hr on-demand) |

> **Why g6e.8xlarge (not g6e.4xlarge) for EKS rollout:** each rollout worker pod requests
> 24 vCPU / 200 GiB, which does **not** fit `g6e.4xlarge` (16 vCPU / 128 GiB → pods stay
> `Pending`). `g6e.8xlarge` (32 vCPU / 256 GiB) is the smallest single-L40S g6e that schedules it.
> Cost figures are approximate us-east-1 on-demand (g6e.8xlarge ≈ $4.53/hr, g6e.48xlarge ≈ $30.13/hr,
> g6e.12xlarge ≈ $10.49/hr) — **scale the fleet to 0 / tear it down immediately after each run**
> (see Teardown).

**EKS + KubeRay is validated end-to-end: 8 clean PPO iterations across ~37h unattended, with checkpoints saved.** The Batch MNP path RAM-OOMs after ~2 PPO iterations (the head node exhausts system RAM), so use EKS for any real run and reserve Batch for short smoke tests.

## Quick Start

### Prerequisites

This assumes an **existing VPC with NAT egress** (see the VPC bullet below) and the **listed GPU
quotas** already granted — it is not a from-scratch account bootstrap (creating the VPC/subnets/NAT
and the IAM permissions to deploy CDK is out of scope here). Keep S3 + ECR + VPC + FSx all in **one**
region — the FSx DRA requires same-region S3:

- **CLIs:** `aws` (v2), `cdk` (**`npm install -g aws-cdk@latest`** — a stale global CDK CLI can be
  older than the `aws-cdk-lib` that `pip` installs and fail `cdk synth` with a *"Cloud assembly schema
  version mismatch"*; keep the CLI current), `docker` (only for the manual self-build),
  and `jq` (`prepare-artifacts.sh` parses build JSON with it). Python 3.10+ with the CDK deps:
  `pip install -r infra/requirements.txt`.
- **GPU quota:** enough **G-instance vCPUs** for your fleet (e.g. a benchmark eval on 8× g6e.8xlarge
  = 256 vCPUs; a training learner + rollout fleet is larger). Request the *"Running On-Demand G and
  VT instances"* quota (code `L-DB2E81BA`) in your region **before** deploying — new accounts start
  well below what a real run needs, and increases can take hours to days.
- **VPC:** a VPC with **≥2 private subnets in different AZs**, each with **NAT gateway egress**
  (nodes must reach the EKS API, ECR, PyPI/GitHub, and the public Omniverse asset CDN — see the
  asset note below). Pick AZ(s) that actually have g6e capacity; **probe first** with
  `infra/capacity-probe.sh --subnet <subnet-id> --instance-type g6e.8xlarge --capacity <n>`.
- **S3 data bucket (same region):** `aws s3 mb s3://<your-bucket> --region <region>`. It's the
  DRA source FSx-Lustre lazily imports from; `stage-s3-eks.sh` populates it.
- **CDK bootstrap (once per account/region):**
  `cdk bootstrap aws://$(aws sts get-caller-identity --query Account --output text)/<region>`.
- **Runtime egress for assets** (nodes need outbound HTTPS to all of these; no credentials were
  required in the validated smoke, but their availability is a real dependency):
  - `omniverse-content-production.s3-us-west-2.amazonaws.com` — the trocar scene/props USD
    (`workflows/.../simulation/assets/assets.py` `ASSET_PATH`). These "Powered by LightWheel" assets
    are **CC-BY-NC-4.0 (NonCommercial)** — fine for research/eval, not for commercial redeployment.
  - `api.lightwheel.net` + `api-s3-assets.lightwheel.net` — IsaacLab-Arena's *generic* object library
    (Microwave/CoffeeMachine) calls the LightWheel SDK at import time (reached transitively because the
    trocar task imports teleop). It returned HTTP 200 anonymously in the smoke, so no LightWheel account
    is needed, but the hosted API IS contacted. `lightwheel-sdk` (public PyPI, Apache-2.0) is baked into
    the image.

### Deploy (EKS — recommended, validated)

The EKS build/stage/deploy flow is **two stacks, three phases**: (1) deploy the persistent
`GR00TRLArtifactsStack` (ECR repo + the `GR00T-RL-Pipeline` CodeBuild project), (2) build the
image + stage the data through that pipeline, (3) deploy the consumer `GR00TRLEKSStack` with the
resolved image digest. There is **NO deployment auto-trigger**: a direct `cdk deploy
GR00TRLEKSStack` WITHOUT completing phases 1-2 is **unsupported** — in fact the stack now
**fails synth** if you don't pass an explicit `image_uri` (there is deliberately no floating
`:latest` fallback). The image build + data staging are driven through `prepare-artifacts.sh`,
which gates the EKS deploy on verified artifacts and deploys the resolved `@sha256` digest.

```bash
# 0. Region env (reused by every step)
cd training/gr00t/rl/infra
export AWS_REGION=<region> AWS_DEFAULT_REGION=<region> CDK_DEFAULT_REGION=<region>
export CDK_DEFAULT_ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"

# 1. Persistent artifacts stack: gr00t-rl-unified ECR repo + the gated GR00T-RL-Pipeline
#    CodeBuild project. Deploy ONCE (persists across EKS teardowns). artifacts_only=true
#    is REQUIRED in this phase — it tells the app to synth ONLY the artifacts stack
#    (the EKS stack has no verified image yet and fails closed without one).
cdk deploy GR00TRLArtifactsStack --context compute_backend=eks --context artifacts_only=true --context s3_data_bucket=<bucket>

# 2. Build image + stage data, poll, verify, then deploy EKS with the verified digest.
./prepare-artifacts.sh --region <region> --mode all --deploy --deploy-stack GR00TRLEKSStack -- \
    --context vpc_id=<vpc> --context s3_data_bucket=<bucket> --context mode=eval \
    --context rollout_instance_type=g6e.8xlarge --context num_rollout_workers=1 \
    --context eval_total_envs=8 --context rollout_subnet_ids=<subnet> \
    --context eval_learner_subnet_ids=<subnet> --context fsx_subnet_id=<subnet>
```

`GR00T-RL-Pipeline` is the ONE mode-switched CodeBuild project (`STAGE_MODE ∈ {build-image,
stage-data, all}`) owned by `GR00TRLArtifactsStack`. `prepare-artifacts.sh` kicks it, polls
to completion, verifies the ECR digest and/or the `s3://.../_STAGING_COMPLETE` marker landed,
and only then runs `cdk deploy GR00TRLEKSStack` with `image_uri=<...@sha256:digest>`. Everything
after `--` is forwarded verbatim to `cdk deploy` (do not pass `image_uri=`/`compute_backend=` —
the wrapper owns them). `--mode data` re-stages only; append optional
`--context learner_instance_type=` / `rollout_instance_type=` after `--`. Omit `--deploy` to
stop after verify and just print the exact deploy line. The EKS backend does **not** use EFS —
storage is FSx for Lustre backed by S3 (see Architecture). This matches the `deploy-eks-training`
skill.

**Manual alternative (bring-your-own image):** run from the `training/gr00t/rl` directory (NOT
`infra/` — the script's `docker build` context is `docker/` relative to that dir):
```bash
cd training/gr00t/rl        # if you were in infra/, `cd ..`
scripts/build_unified_and_push.sh --region <region>
```
It does a local `docker build`/`push` to the same `gr00t-rl-unified` repo and prints the
`--context image_uri=<repo>@sha256:<digest>` line to hand to `cdk deploy` (or to
`prepare-artifacts.sh --mode data --image-uri <...>`).

```bash
# Batch MNP (homogeneous, simpler — quick experiments only)
AWS_REGION=<region> AWS_DEFAULT_REGION=<region> cdk deploy --context compute_backend=batch-mnp
```

### Canonical end-to-end run (validated): smoke-eval → train → eval → teardown

The cheapest way to prove the whole path works before committing to a real run. Each step
re-deploys the **same** `GR00TRLEKSStack` with different context (mode-switch in place); the
image + data were built/staged in Quick Start phases 1-2. Set `IMG=<account>.dkr.ecr.<region>.amazonaws.com/gr00t-rl-unified@sha256:<digest>`
(the digest `prepare-artifacts.sh` resolved) and reuse the region env from step 0.

```bash
cd training/gr00t/rl/infra

# (a) SMOKE-EVAL the base model — 2-pod fleet, 8 envs (8 episodes). Cheapest sanity check
#     of the eval path (~2× g6e.8xlarge ≈ $9/hr). No eval_ckpt => base-model eval.
cdk deploy GR00TRLEKSStack --context compute_backend=eks \
  --context vpc_id=<vpc> --context s3_data_bucket=<bucket> --context image_uri=$IMG \
  --context mode=eval --context rollout_instance_type=g6e.8xlarge \
  --context num_rollout_workers=1 --context eval_total_envs=8 --force

# (b) TRAIN a cost-bounded 2-step run. save_interval=2 → a checkpoint at global_step_2, then stop.
#     envs_per_worker=8 shrinks per-step cost; on-demand g6e.48xlarge learner (no Capacity Block).
cdk deploy GR00TRLEKSStack --context compute_backend=eks \
  --context vpc_id=<vpc> --context s3_data_bucket=<bucket> --context image_uri=$IMG \
  --context mode=train --context max_epochs=2 --context envs_per_worker=8 \
  --context rollout_instance_type=g6e.8xlarge --force
# watch for "global_step_2" + the checkpoint save in the head logs:
kubectl logs -n training -l ray.io/node-type=head -f

# (c) LOCATE the checkpoint the train run wrote (the actor .pt FILE, not the dir):
kubectl exec -n training <head-pod> -- \
  find /mnt/fsx/rl-training/results -path '*/global_step_2/actor/model_state_dict/full_weights.pt'
#   => /mnt/fsx/rl-training/results/<config>_eks_train/<timestamp>/.../global_step_2/actor/model_state_dict/full_weights.pt

# (d) EVAL that checkpoint — pass the FULL FSx .pt path from (c):
cdk deploy GR00TRLEKSStack --context compute_backend=eks \
  --context vpc_id=<vpc> --context s3_data_bucket=<bucket> --context image_uri=$IMG \
  --context mode=eval --context rollout_instance_type=g6e.8xlarge \
  --context num_rollout_workers=1 --context eval_total_envs=8 \
  --context eval_ckpt=<the full_weights.pt path from step c> --force

# (e) TEARDOWN — stop all GPU spend (see the Teardown section for the full sequence):
kubectl delete raycluster gr00t-rl-training -n training --ignore-not-found
cdk destroy GR00TRLEKSStack --context compute_backend=eks \
  --context vpc_id=<vpc> --context s3_data_bucket=<bucket> --context image_uri=$IMG --force
```

### Stage Training Data (EFS)

After deploying, trigger the CodeBuild project to stage code + model on EFS:

```bash
aws codebuild start-build --project-name GR00T-RL-Stage-EFS --region <region>
```

This stages:
- RLinf framework (pinned commit)
- Isaac-GR00T model code (commit `4af2b622`)
- IsaacLab + IsaacLab-Arena
- GR00T N1.5 pre-trained checkpoint
- Training workflows (rlinf_ext, configs, task definition)

This EFS CodeBuild step is for the **batch-mnp / sagemaker** backends only — the EKS backend
uses FSx-Lustre backed by S3, staged by the `GR00T-RL-Pipeline` stage-data arm (see below).

### Stage Training Data (EKS → S3/FSx)

The EKS backend reads its data from FSx-Lustre, which lazily imports from an S3 bucket via a
Data Repository Association. Staging that bucket is the **stage-data arm of the single
`GR00T-RL-Pipeline` CodeBuild project** (owned by `GR00TRLArtifactsStack`), driven + gated by
`infra/prepare-artifacts.sh` in Step 2 of the Deploy flow above — there is **no** auto-trigger.
The stage-data arm runs `infra/stage-s3-eks.sh`: it clones the pinned third-party repos,
**applies the RLinf `_broadcast` patch**, downloads the model, stages the workflows, uploads
everything to `$S3_DATA_BUCKET`, and writes the `s3://$S3_DATA_BUCKET/_STAGING_COMPLETE` marker
last on full success.

To re-stage any time (e.g. after changing a pin, patch, or workflow):

```bash
# gated wrapper (verifies the marker, then can deploy with the verified image):
./prepare-artifacts.sh --region <region> --mode data
# or the pipeline directly:
aws codebuild start-build --project-name GR00T-RL-Pipeline \
  --environment-variables-override name=STAGE_MODE,value=stage-data,type=PLAINTEXT \
  --region <region>
```

`infra/stage-s3-eks.sh` is the fail-closed engine CodeBuild runs (committed here for local dev /
inspection; you never need to run it by hand). You may deploy to any region with p5/p5e + g6e
capacity; keep S3 + ECR + VPC + FSx all in that ONE region (the FSx DRA requires same-region S3)
and probe capacity first with `infra/capacity-probe.sh`.

### Run Training

**Batch MNP:**
```bash
aws batch submit-job \
  --job-name gr00t-rl-training \
  --job-queue GR00T-RL-JobQueue \
  --job-definition <job-definition-arn> \
  --region <region>
```

**EKS:** Training starts automatically when the RayCluster pods are created by CDK deploy. Monitor with:
```bash
# Use the role ARN from the KubeconfigCommand stack output
aws eks update-kubeconfig --name gr00t-rl-eks --region <region> \
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

# TensorBoard (EKS). FSx (/mnt/fsx) is mounted ONLY inside the Ray pods — NOT on your laptop —
# so `tensorboard --logdir /mnt/fsx/...` will not work locally. Instead pull the run's TB dir
# from the DRA-exported S3 copy and serve it locally:
aws s3 sync s3://<your-s3-bucket>/rl-training/results/<run>/tensorboard ./tb && tensorboard --logdir ./tb
# Or serve it in-pod and port-forward:
#   kubectl exec -n training <head-pod> -- tensorboard --logdir /mnt/fsx/rl-training/results/ --host 0.0.0.0 --port 6006 &
#   kubectl port-forward -n training <head-pod> 6006:6006
# (Batch MNP path stores results on EFS at /mnt/efs/rl-training/results/ instead)

# Centralized, PERSISTENT pod logs in CloudWatch (EKS — on by default).
# The amazon-cloudwatch-observability add-on ships all container stdout/stderr here; the
# log group survives cluster teardown (unlike `kubectl logs`), so you can review a run after.
aws logs tail /aws/containerinsights/gr00t-rl-eks/application --region <region> --follow
# Filter to just the Ray head (training/eval progress, incl. eval/success_once):
aws logs tail /aws/containerinsights/gr00t-rl-eks/application --region <region> \
  --filter-pattern "success_once" --since 2h
```

> **Centralized logging (EKS):** the CloudWatch Observability add-on is installed **by
> default** and streams pod logs to CloudWatch Container Insights (log group
> `/aws/containerinsights/gr00t-rl-eks/application`, 30-day retention). These persist after
> `cdk destroy`. It adds a small per-node agent + CloudWatch ingest/storage cost — disable with
> `--context enable_cloudwatch_logs=false` if you'd rather rely on `kubectl logs` only. (This
> is EKS **pod** logging; it is separate from EKS *control-plane* logging, which we don't enable
> as it wouldn't capture the training/eval stdout.)

### Teardown

**EKS (routine — stops all GPU spend):**
```bash
export AWS_REGION=<region> AWS_DEFAULT_REGION=<region> CDK_DEFAULT_REGION=<region>

# 1. Delete the RayCluster first so KubeRay drains pods before the cluster goes
#    (avoids a custom-resource deletion hang):
kubectl delete raycluster gr00t-rl-training -n training --ignore-not-found

# 2. Destroy the consumer stack (tears down node groups → terminates all g6e,
#    plus FSx, the DRA, and the EKS cluster). This is the correct path.
#    NOTE: `cdk destroy` still SYNTHS the app first, so the EKS backend's required
#    contexts must be present (compute_backend, vpc_id, s3_data_bucket, and an
#    image_uri — the app fails closed without one; there is no :latest fallback).
cdk destroy GR00TRLEKSStack --context compute_backend=eks \
  --context vpc_id=<your-vpc-id> --context s3_data_bucket=<your-s3-bucket> \
  --context image_uri=<the digest you deployed with> --force

# 3. LAST RESORT ONLY — if `cdk destroy` hangs on the kubectl custom resources.
#    The raw `delete-cluster` on its own would ORPHAN the CloudFormation stack, so
#    you MUST follow it with `delete-stack` to clean up (do not stop after line 1):
aws eks delete-cluster --name gr00t-rl-eks --region <region>
aws cloudformation delete-stack --stack-name GR00TRLEKSStack --region <region>

# 4. VERIFY zero GPU spend (should print nothing):
aws ec2 describe-instances --region <region> \
  --filters "Name=instance-state-name,Values=running,pending" \
  --query 'Reservations[].Instances[?starts_with(InstanceType,`g6e`)].[InstanceId,InstanceType]' \
  --output text
```

**Batch MNP:**
```bash
AWS_DEFAULT_REGION=<region> cdk destroy GR00TRLBatchStack --context compute_backend=batch-mnp --force
```

**Full cleanup (only if you want to remove EVERYTHING, including the retained artifacts):**
```bash
# The artifacts stack synths only when the app is told to (compute_backend=eks) AND is allowed to
# skip the EKS backend — pass artifacts_only=true + s3_data_bucket, or the app fails closed
# (no image_uri → no :latest fallback).
cdk destroy GR00TRLArtifactsStack --context compute_backend=eks \
  --context artifacts_only=true --context s3_data_bucket=<your-s3-bucket> --force
aws ecr delete-repository --repository-name gr00t-rl-unified --force --region <region>
aws s3 rb s3://<your-s3-bucket> --force --region <region>   # deletes all staged data
```

> **Do NOT destroy `GR00TRLArtifactsStack` as part of routine teardown.** It is designed to persist
> across EKS teardowns (it owns the built image + the pipeline). Its ECR repo `gr00t-rl-unified` is
> created with a **RETAIN** policy, so destroying the stack leaves the repo behind; a later re-deploy
> then **collides** on the existing repo name (the Batch stack also references `gr00t-rl-unified`).
> If you truly must remove it, delete the repo explicitly afterward
> (`aws ecr delete-repository --repository-name gr00t-rl-unified --force --region <region>`) before
> re-deploying. For normal cost control, just tear down the EKS stack and leave the artifacts stack
> (an idle ECR repo + an untriggered CodeBuild project cost essentially nothing).

## Modes

The EKS + KubeRay backend supports two runtime modes via a `mode` CDK context param. Unset (or `mode=train`) preserves today's behavior; `mode=eval` runs standalone evaluation on a saved checkpoint.

> **The `cdk deploy GR00TRLEKSStack …` commands in this section are the *advanced redeploy* path** —
> they assume you have **already** completed phases 1-2 (artifacts stack deployed, image built +
> data staged) and are re-deploying the consumer stack against an existing, **verified** image.
> Pass `image_uri` as a **pinned digest** (`…/gr00t-rl-unified@sha256:<digest>`), not a floating
> tag — the stack fails synth without an explicit `image_uri` and never falls back to `:latest`.
> For a first deploy (or any time the image/data changed), use `prepare-artifacts.sh` (Quick Start
> phase 2) instead: it builds, verifies, resolves the digest, and runs this deploy for you.

### MODE=train (default)

- **Purpose:** PPO training on the Assemble Trocar task.
- **Topology:** 1 head pod on `<learner-instance>` (8× L40S or 8× H100) + N worker pods on `<rollout-instance>` (1× L40S each). **A Capacity Block / p5 is OPTIONAL:** by default (no `capacity_reservation_id`) the learner node group is an **on-demand `g6e.48xlarge`** (8× L40S) — no reservation required. Supply `--context capacity_reservation_id=<cr-id>` only if you want to run the learner on reserved p5 (H100) capacity.
- **Invocation:**

```bash
cd training/gr00t/rl/infra
AWS_DEFAULT_REGION=<region> cdk deploy GR00TRLEKSStack \
  --context compute_backend=eks \
  --context vpc_id=<your-vpc-id> \
  --context s3_data_bucket=<your-s3-bucket> \
  --context image_uri=<account>.dkr.ecr.<region>.amazonaws.com/gr00t-rl-unified@sha256:<digest>
  # optional: --context capacity_reservation_id=<cr-id>   (run the learner on reserved p5/H100 capacity)
```

- **Output:** checkpoints at `${LOG_DIR}/checkpoints/global_step_N/`, TensorBoard at `${LOG_DIR}/tensorboard/` under FSx.
- **Cost-bounded / plumbing run:** add `--context max_epochs=N` to stop after N global_steps (default is the entrypoint's 1000). Because `save_interval=2` writes a checkpoint every 2 steps, e.g. `--context max_epochs=2` produces an eval-able `global_step_2/` checkpoint and stops — useful to validate the train path end-to-end before committing to a full run. Shrink `--context num_rollout_workers` / `--context envs_per_worker` to lower per-step cost (per-step time scales with `total_num_envs × rollout_epoch`).

### MODE=eval

- **Purpose:** standalone evaluation of a saved RL checkpoint; reports `eval/success_once`. No learner GPU required. **MP4 rollout videos are OFF by default** (see the video note under Output).
- **Topology:** 1 head pod + N worker pods, all on `<rollout-instance>` (1× L40S each). Fleet size scales with the `num_rollout_workers` CDK context param — default is 1 (2 pods total, smoke config); benchmark eval at the yaml-default `total_num_envs=64` needs enough GPUs to stay inside a proven envelope on L40S-class hardware. Setting `num_rollout_workers=7` gives an 8-pod fleet at 8 envs/GPU across all 8 GPUs.
- **Invocation (smoke — 2-pod fleet, 8 envs):**

```bash
AWS_DEFAULT_REGION=<region> cdk deploy GR00TRLEKSStack \
  --context compute_backend=eks \
  --context vpc_id=<your-vpc-id> \
  --context s3_data_bucket=<your-s3-bucket> \
  --context image_uri=<account>.dkr.ecr.<region>.amazonaws.com/gr00t-rl-unified@sha256:<digest> \
  --context mode=eval \
  --context rollout_instance_type=g6e.8xlarge \
  --context num_rollout_workers=1 \
  --context eval_total_envs=8 \
  --context eval_ckpt=/mnt/fsx/<your-run-path>/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt
```

  `eval_total_envs` (`--context eval_total_envs=N`) MUST be divisible by `num_nodes = 1 + num_rollout_workers`
  (eval places one env process per node). The smoke above is a valid topology: 8 envs ÷ 2 nodes = 4.
  Omit `eval_total_envs` to fall through to the yaml default (64).

- **Invocation (benchmark eval at `total_num_envs=64`):**

```bash
AWS_DEFAULT_REGION=<region> cdk deploy GR00TRLEKSStack \
  --context compute_backend=eks \
  --context vpc_id=<your-vpc-id> \
  --context s3_data_bucket=<your-s3-bucket> \
  --context image_uri=<account>.dkr.ecr.<region>.amazonaws.com/gr00t-rl-unified@sha256:<digest> \
  --context mode=eval \
  --context rollout_instance_type=g6e.8xlarge \
  --context num_rollout_workers=7 \
  --context eval_ckpt=/mnt/fsx/<your-run-path>/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt
```

  Here `total_num_envs` falls through to the yaml default (64), and 64 ÷ (1 + 7) = 8 envs/node — a valid
  divisible topology across all 8 GPUs.

Omit `--context eval_ckpt=...` when the model at `MODEL_PATH` is itself the RL-trained snapshot (no `.pt` overlay needed).

- **Prerequisites:** the rollout nodegroup can scale to `1 + num_rollout_workers` nodes total (head runs on the eval-learner NG, workers on the rollout NG). The training learner nodegroup can be at `desired=0` (no Capacity Block required).
- **Runtime:** `eval_rollout_epoch` is hardcoded to **1** in `entrypoint-eks.sh`, so an eval runs exactly `total_num_envs` episodes. Benchmark eval (`num_rollout_workers=7`, `total_num_envs=64`) = 64 episodes, ~15-20 min end-to-end per stage. Smoke eval (`num_rollout_workers=1`, `total_num_envs=8`) = **8 episodes** (not 64), a quick plumbing check, ~15-25 min including cluster spin-up.
- **Output:** `eval/success_once` in the head-pod logs (always). **MP4 videos are OFF by default:** the entrypoint passes `env.eval.video_cfg.save_video=${SAVE_VIDEO:-False}`, and `SAVE_VIDEO` is a **raw container env var** — it is **not** exposed as a CDK `--context` param. To capture videos you must set `SAVE_VIDEO=true` in the head pod's env (edit the entrypoint/manifest and redeploy). When enabled, MP4s land at `${LOG_DIR}/video/eval/` on FSx (auto-exported to `s3://<your-s3-bucket>/rl-training/results/…` via the FSx Data Repository Association).
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
export AWS_REGION=<region> AWS_DEFAULT_REGION=<region> CDK_DEFAULT_REGION=<region> \
       CDK_DEFAULT_ACCOUNT=<your-account> VPC_ID=<your-vpc-id> S3_DATA_BUCKET=<your-s3-bucket> \
       IMAGE_URI=<your-account>.dkr.ecr.<region>.amazonaws.com/<your-repo>:<tag>

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
AWS Batch MNP Job (g6e.12xlarge, 4× L40S each; default 1 learner + 4 rollout = 5 nodes)
├── Node 0 (Learner):  Ray Head + FSDP Actor (GPUs 0-3)
└── Node 1..N (Rollout): Ray Worker + Isaac Sim Envs (GPUs 0-3)   # N = num_rollout_nodes (default 4)

Storage: EFS mounted natively at /mnt/efs
Network: NCCL over TCP (no EFA)
```

### EKS + KubeRay (Heterogeneous)

```
EKS Cluster (gr00t-rl-eks)
├── Head Pod (g6e.48xlarge, 8× L40S)
│   ├── Ray Head + FSDP Actor (all 8 GPUs)
│   └── entrypoint-eks.sh → train_embodied_agent.py
└── Worker Pods ×4 (g6e.8xlarge, 1× L40S each)
    ├── Ray Workers
    └── Isaac Sim EnvWorker + RolloutWorker (32 envs each)

Storage: FSx for Lustre (PERSISTENT_2) via the FSx CSI driver at /mnt/fsx,
         backed by S3 through a Data Repository Association (DRA)
Operators: KubeRay, NVIDIA device plugin
```

**Eval mode** (`--context mode=eval`) drops the training learner pod and uses a `(1 + num_rollout_workers)`-pod topology on `<rollout-instance>` (1× L40S each). Default is 2 pods (smoke); benchmark eval at the yaml-default `total_num_envs=64` typically uses `--context num_rollout_workers=7` (8 pods total, 8 envs/GPU). No p5.48xlarge / no Capacity Block required. The head pod runs `entrypoint-eks.sh → eval_embodied_agent.py` and writes MP4 rollouts to `${LOG_DIR}/video/eval/`. See the [Modes](#modes) section above for both smoke and benchmark invocations.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (Proximal Policy Optimization) |
| Model | GR00T N1.5 (3B params, per the NVIDIA model card) |
| FSDP | Fully Sharded Data Parallel across actor GPUs |
| micro_batch_size | **32 on L40S-class hardware (L40S-safe — mbs 64 AND 128 both OOM the 44 GB L40S, per the 2026-06-15 benchmark)**; 128 (no gradient checkpointing) only on H100/p5 (80 GB VRAM). Configurable via `MICRO_BATCH_SIZE` env var (entrypoint sync default is already 32) |
| gradient_checkpointing | True (must stay True at these batch sizes) |
| Rollout epochs | 8 per iteration |
| Update epochs | 4 per iteration |
| Save interval | Every 2 iterations |
| Max epochs | 1000 |

## Training Outputs

On the EKS path, results are saved to FSx for Lustre at `/mnt/fsx/rl-training/results/...`
(auto-exported to S3 via the DRA). (On the Batch MNP path they live on EFS at
`/mnt/efs/rl-training/results/...` instead.)

```
# LOG_DIR carries provenance in the path: <config>_<backend>_<mode>/<timestamp>
/mnt/fsx/rl-training/results/<config>_eks_train/<timestamp>/
├── tensorboard/events.out.tfevents.*   # Training metrics
└── gr00t_assemble_trocar/
    └── checkpoints/global_step_N/
        └── actor/model_state_dict/full_weights.pt  # actor checkpoint (~5.5 GB) — this is the eval_ckpt .pt FILE
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
│   ├── buildspec-stage-efs.yml  # CodeBuild: stage code + model to EFS (Batch)
│   ├── buildspec-pipeline.yml   # CodeBuild: GR00T-RL-Pipeline (build-image|stage-data|all)
│   └── requirements-unified.txt # Python dependencies
├── infra/
│   ├── app.py                   # CDK app (routes compute_backend)
│   ├── mnp_batch_stack.py       # Batch MNP CDK stack
│   ├── eks_kuberay_stack.py     # EKS + KubeRay CDK stack
│   ├── artifacts_stack.py       # GR00TRLArtifactsStack: persistent ECR + GR00T-RL-Pipeline
│   ├── prepare-artifacts.sh     # Gated build+stage wrapper (kicks pipeline, verifies, deploys)
│   ├── stage-s3-eks.sh          # Staging engine the pipeline's stage-data arm runs (EKS → S3/FSx)
│   ├── eval-checkpoint.sh       # One-command multi-stage eval sweep (EKS)
│   ├── capacity-probe.sh        # Probe GPU capacity in a subnet before deploy
│   ├── patch-success-stage.sh   # Per-stage success_stage patch (eval sweep helper)
│   └── requirements.txt         # CDK Python dependencies
├── patches/
│   └── RLinf-649e7579-broadcast-raise.patch  # RLinf _broadcast re-raise (applied at stage time)
├── scripts/
│   ├── build_unified_and_push.sh  # Manual self-build of the unified image (bring-your-own)
│   └── submit_training.sh       # Job submission helper
└── workflows/                  # ONLY our custom RL config overlay (NOT the full tree)
    └── simulation/rl/rlinf_ext/config/
        ├── isaaclab_ppo_gr00t_assemble_trocar.yaml  # tuned PPO hyperparams (overrides upstream)
        ├── training_backend/fsdp.yaml               # our FSDP backend config (not in upstream)
        └── env/, model/                             # (identical to upstream, kept for a self-contained overlay)
```

> The complete i4h-workflows `rheo/scripts` tree (`simulation/`, `policy/`, `teleop_devices/`,
> `utils/`, `config/` — interdependent on `PYTHONPATH`) is **cloned at stage time** by
> `stage-s3-eks.sh` from a **pinned commit** of `isaac-for-healthcare/i4h-workflows` (v0.5.0), and
> only the `config/` overlay above is layered on top. The repo intentionally does **not** vendor a
> subset of that tree — a hand-picked subset silently drops interdependent siblings and breaks env
> creation at import time (e.g. `ModuleNotFoundError: teleop_devices`).

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| CUDA OOM during training step | gradient_checkpointing=False with large batch | Keep GRADIENT_CHECKPOINTING=True (default) when micro_batch_size=128 |
| FSDP NCCL deadlock | cpu_offload enabled | Never set cpu_offload=True |
| torch.compile hangs forever | Incompatible with Isaac Sim multi-process | TORCHDYNAMO_DISABLE=1 |
| Ray kills workers (Batch) | System RAM >95% after 2 iterations | Use EKS backend (768 GB RAM) or set RAY_memory_usage_threshold=0.99 |
| Pods can't find `/mnt/fsx/third_party` on first deploy (EKS) | Data not staged yet — `prepare-artifacts.sh` gates the deploy on the `GR00T-RL-Pipeline` stage-data arm (~15-20 min), so this only bites if you bypassed the wrapper; FSx imports lazily via the DRA | Confirm the marker: `aws s3 ls s3://<your-s3-bucket>/_STAGING_COMPLETE` (KubeRay self-heals the head until data lands) |
| Region mismatch / CDK lookup uses wrong region (EKS) | Only a partial region env var set | Set `AWS_REGION` (and `CDK_DEFAULT_REGION`/`AWS_DEFAULT_REGION`) explicitly; keep S3 + ECR + VPC + FSx in the SAME region (the FSx DRA requires same-region S3) |
| `cdk synth`/`deploy` fails: *"Cloud assembly schema version mismatch … You need at least CLI version X"* | The global `cdk` CLI is older than the `aws-cdk-lib` `pip` installed (unbounded `>=`, so pip pulls the newest lib) | Upgrade the CLI: `npm install -g aws-cdk@latest` (AWS ships the CLI and library together, so the latest CLI reads the latest lib). Not a code issue — the stacks synth fine once the versions are compatible |
| Pods Pending (EKS) | Insufficient GPU quota or node not Ready | Check `kubectl describe node` and service quotas |
| `ray: command not found` (EKS) | Ray binary not on PATH | PATH env var includes /isaac-sim/kit/python/bin |
