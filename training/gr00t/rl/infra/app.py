#!/usr/bin/env python3
"""CDK app for GR00T RL post-training infrastructure.

Deploy (Batch MNP - default):
  cd training/gr00t/rl/infra
  cdk deploy --context compute_backend=batch-mnp --context num_rollout_nodes=4

Deploy (SageMaker heterogeneous):
  cdk deploy --context compute_backend=sagemaker --context num_rollout_nodes=4

Deploy (EKS + KubeRay):
  cdk deploy --context compute_backend=eks \\
    --context vpc_id=<your-vpc-id> \\
    --context s3_data_bucket=<your-s3-bucket> \\
    --context image_uri=<your-account>.dkr.ecr.<region>.amazonaws.com/<your-repo>:<tag>

Deploy (EKS + KubeRay - eval on a saved checkpoint):
  cdk deploy --context compute_backend=eks \\
    --context vpc_id=<your-vpc-id> \\
    --context s3_data_bucket=<your-s3-bucket> \\
    --context image_uri=<your-account>.dkr.ecr.<region>.amazonaws.com/<your-repo>:<tag> \\
    --context mode=eval \\
    --context eval_ckpt=/mnt/fsx/rl-training/results/<run>/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt

Deploy (EKS + KubeRay - eval of a base/SFT model on FSx):
  cdk deploy --context compute_backend=eks \\
    --context vpc_id=<your-vpc-id> \\
    --context s3_data_bucket=<your-s3-bucket> \\
    --context image_uri=<your-account>.dkr.ecr.<region>.amazonaws.com/<your-repo>:<tag> \\
    --context mode=eval \\
    --context model_path=/mnt/fsx/models/<your-model-dir>

Context parameters:
  vpc_id                - Existing VPC ID (creates new if omitted for batch-mnp)
  efs_id                - Existing EFS file system ID (batch-mnp/sagemaker only)
  efs_sg_id             - EFS security group ID (batch-mnp/sagemaker only)
  s3_data_bucket        - S3 bucket name with staged training data (EKS only, DRA-linked to FSx)
  fsx_capacity_gib      - FSx for Lustre capacity in GiB (EKS only, default: 1200)
  image_uri             - Pre-built ECR URI for the unified image (required for EKS). Normally the wrapper-resolved @sha256 digest built by the GR00T-RL-Pipeline CodeBuild project (GR00TRLArtifactsStack) and handed over by infra/prepare-artifacts.sh; or a bring-your-own image from scripts/build_unified_and_push.sh. There is no in-stack CodeBuild auto-trigger.
  num_rollout_nodes     - Number of rollout child nodes for batch-mnp/sagemaker (default: 4)
  num_rollout_workers   - Number of rollout worker pods for eks (default: 4)
  learner_instance_type - EC2 instance type for learner node group (default: g6e.48xlarge)
  rollout_instance_type - EC2 instance type for rollout node group (default: g6e.8xlarge — the rollout worker pod requests 24 vCPU / 200 GiB, which does NOT fit g6e.4xlarge's 16 vCPU / 128 GiB)
  compute_backend       - "batch-mnp" (default), "sagemaker", or "eks"
  max_epochs            - EKS train only. Bound the run to N global_steps (default: entrypoint's 1000). Pair with save_interval for a cost-bounded run that still writes an eval-able checkpoint.
  "mode"                - "train" (default) or "eval" — routes the EKS backend to training or standalone eval
  "eval_ckpt"           - Full path to actor checkpoint (.pt) for mode=eval; ignored for mode=train
  "model_path"          - Full FSx-visible path (mount root /mnt/fsx) to the model dir. When omitted, the entrypoint's default (RL model) is used. Enables SFT/RL model swap without editing the entrypoint.
  "eval_total_envs"     - Optional integer. When set, appends EVAL_TOTAL_ENVS=<val> to the eval head-pod env, which entrypoint-eks.sh reads to override env.eval.total_num_envs. Omit to fall through to the yaml default (64). MUST be divisible by num_nodes = 1 + num_rollout_workers (eval places one env process per node).
  "eval_actor_gbs"      - Optional integer. When set, appends EVAL_ACTOR_GBS=<val> to the eval head-pod env, which entrypoint-eks.sh reads to override actor.global_batch_size. Omit to fall through to the yaml default (2048). Lets you satisfy the RLinf validator (gbs % (mbs * world_size) == 0) at unusual node counts — e.g. EVAL_ACTOR_GBS=1280 with mbs=128 unlocks world_size=10 for a 10-node topology. Safe in eval mode because actor.global_batch_size is consumed only by the FSDP trainer, which eval_embodied_agent.py never spawns.
  "task_description"    - Optional string. When set, appends TASK_DESCRIPTION=<val> to the eval head-pod env, which entrypoint-eks.sh reads to override env.eval.init_params.task_description. Omit to fall through to the yaml default ("install trocar from box"). Useful when comparing against a baseline that used a different prompt (e.g. "assemble trocar from tray"). Additive-only; unset preserves the shipped byte-identical behavior.
  "eval_inject_noise"   - Optional string. When set to "true", appends EVAL_INJECT_NOISE=true to the eval head-pod env. On the FSx-side, the patched RLinf gr00t_action_model.py reads this env var and, when true, uses the train-mode noise formula (flow_sde + noise_level yaml default) at eval time instead of the deterministic x_t_std=0 branch. Diagnostic knob for probing whether eval-time sampling temperature affects results. Requires the corresponding FSx-side patch to have any effect. Additive-only; when unset the patched Python code preserves original eval-mode behavior byte-identically.
  "noise_level"         - Optional string (float). When set, appends NOISE_LEVEL=<val> to the eval head-pod env, and entrypoint-eks.sh emits ++actor.rl_head_config.noise_level=<val> as a Hydra override. Used only when eval_inject_noise=true. Yaml default is 0.3 (flow_sde); sweep values (e.g. {0.1, 0.6, 1.0}) to characterize the noise-vs-success curve. Additive-only; unset preserves the shipped yaml default.
  "save_video"          - Optional string (EKS eval only). Set to "true" to write MP4 rollout videos to ${LOG_DIR}/video/eval/ (exported to S3 via DRA). Default off — the entrypoint forces SAVE_VIDEO=False unless this is set.
  "kuberay_version"     - Optional string (EKS only). KubeRay operator Helm chart version. Default "1.1.0" is the validated version — a deploy without this flag synthesizes exactly the frozen 1.1.0 stack. Pass --context kuberay_version=1.2.0 for async node-recovery (which needs KubeRay >= 1.2.0). Omit the flag to fall back to 1.1.0; no in-place mutation of the validated stack.
  "rollout_subnet_ids"  - Optional string (EKS only). Default-off capacity-resilient rollout knob. Unset keeps the RolloutNodes node group on the single-AZ FSx subnet (byte-identical synth). Set to one or more comma-separated private subnet IDs to place the rollout fleet in a different AZ (chasing g6e capacity) while the learner and FSx stay in the FSx AZ. Applies to the RolloutNodes NG ONLY — the learner NG stays on the FSx subnet, and the eval-learner NG stays on the FSx subnet unless eval_learner_subnet_ids is also set. Reversible by omitting the flag; subnet IDs are supplied at deploy via --context and never committed.
  "eval_learner_subnet_ids" - Optional string (EKS only). Default-off capacity-resilient EVAL knob. Unset keeps the EvalLearnerNodes NG (which runs the eval head pod) on the single-AZ FSx subnet (byte-identical synth). Set to comma-separated private subnet IDs to place the eval-learner in another AZ when the FSx AZ is g6e-capacity-dry; FSx stays put and is read cross-AZ (the static CSI PV has no topology affinity). Pair with rollout_subnet_ids=<same subnet> so the eval head + rollout workers co-locate intra-AZ. Reversible by omitting; subnet IDs supplied at deploy via --context, never committed.
  "fsx_subnet_id"       - Optional string (EKS only). Pins the single-AZ FSx-Lustre filesystem (and the CB/on-demand learner NG that co-locates with it) to a SPECIFIC private subnet instead of the first PRIVATE_WITH_EGRESS subnet (index 0). Set it to the subnet in the AZ that actually holds g6e/H100 capacity so FSx and the learner land together. Unset keeps the historical select_subnets(...).subnets[0] default (byte-identical synth). Subnet ID supplied at deploy via --context, never committed.
"""
import os
import sys
import aws_cdk as cdk
from mnp_batch_stack import RLBatchMNPStack
from eks_kuberay_stack import EKSKubeRayStack
from artifacts_stack import GR00TRLArtifactsStack
from network_stack import GR00TRLNetworkStack

app = cdk.App()

# Region portability: derive the region from the environment with NO baked default.
# A hardcoded default (previously "us-west-2") silently misdeploys when the operator
# sets only a partial region env (e.g. AWS_REGION but not CDK_DEFAULT_REGION) while the
# rest of the bash tooling/docs default elsewhere. Fail closed instead: require one of
# the region env vars to be exported to the intended target region.
#
# Additionally REJECT disagreement: if two of the three region env vars are set to
# DIFFERENT non-empty values, that is an operator error (partial re-export) that
# would otherwise silently resolve to whichever wins the precedence order below —
# and misdeploy to a region where the S3/VPC/FSx don't live. Fail closed instead.
_region_sources = {
    "CDK_DEFAULT_REGION": os.environ.get("CDK_DEFAULT_REGION"),
    "AWS_REGION": os.environ.get("AWS_REGION"),
    "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION"),
}
_distinct_regions = {v for v in _region_sources.values() if v}
if len(_distinct_regions) > 1:
    _detail = ", ".join(
        f"{k}={v!r}" for k, v in _region_sources.items() if v
    )
    raise SystemExit(
        "Conflicting AWS region env vars — refusing to guess. "
        f"Set them all to the SAME region (or unset the extras). Saw: {_detail}. "
        "Keep S3 + ECR + VPC + FSx all in that ONE region (the FSx DRA requires same-region S3)."
    )
_region = (
    os.environ.get("CDK_DEFAULT_REGION")
    or os.environ.get("AWS_REGION")
    or os.environ.get("AWS_DEFAULT_REGION")
)
if not _region:
    raise SystemExit(
        "No AWS region set. Export CDK_DEFAULT_REGION (or AWS_REGION / AWS_DEFAULT_REGION) "
        "to your target region before deploying, e.g. `export CDK_DEFAULT_REGION=us-east-2`. "
        "Keep S3 + ECR + VPC + FSx all in that ONE region (the FSx DRA requires same-region S3)."
    )

env = cdk.Environment(
    account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=_region,
)

compute_backend = app.node.try_get_context("compute_backend") or "batch-mnp"

# Shared artifacts stack (persistent ECR repo `gr00t-rl-unified` + the mode-switched
# GR00T-RL-Pipeline CodeBuild project). Used by the EKS-family backends, which are
# pure consumers of the built image + staged bucket. Deploy this FIRST, then drive it
# with infra/prepare-artifacts.sh, then deploy the backend stack with the resolved
# image_uri. Instantiated as its own stack so it OWNS those resources persistently
# (independent of any single backend deploy).
if compute_backend == "eks":
    GR00TRLArtifactsStack(
        app,
        "GR00TRLArtifactsStack",
        s3_data_bucket=app.node.try_get_context("s3_data_bucket"),
        image_tag=app.node.try_get_context("image_tag") or "latest",
        env=env,
    )

# Opt-in cold-start network bootstrap (Phase 15). A SEPARATE, PERSISTENT stack that
# creates an EKS-ready VPC (>=2 AZs, private+public subnets, NAT egress, EKS subnet
# tags) and OUTPUTS its VpcId + subnet IDs. Deploy it ONCE on a fresh account with no
# VPC, then pass --context vpc_id=<VpcId> into the normal EKS deploy. It is NOT the
# default and does NOT touch the vpc_id path. Synthesizes on its own — needs neither
# image_uri nor vpc_id. Gate: compute_backend=network OR create_network=true.
_create_network = str(
    app.node.try_get_context("create_network") or ""
).lower() in ("1", "true", "yes")

if compute_backend == "network" or _create_network:
    stack = GR00TRLNetworkStack(
        app,
        "GR00TRLNetworkStack",
        vpc_cidr=app.node.try_get_context("vpc_cidr") or "10.73.0.0/16",
        max_azs=int(app.node.try_get_context("network_max_azs") or 2),
        nat_gateways=int(app.node.try_get_context("network_nat_gateways") or 1),
        env=env,
    )
elif compute_backend in ("batch-mnp", "sagemaker"):
    stack = RLBatchMNPStack(
        app,
        "GR00TRLBatchStack",
        vpc_id=app.node.try_get_context("vpc_id"),
        efs_id=app.node.try_get_context("efs_id"),
        efs_sg_id=app.node.try_get_context("efs_sg_id"),
        image_uri=app.node.try_get_context("image_uri"),
        num_rollout_nodes=int(app.node.try_get_context("num_rollout_nodes") or 4),
        env=env,
    )
elif compute_backend == "eks" and not app.node.try_get_context("image_uri"):
    # The EKS stack is a PURE consumer of a pre-built image — it requires an explicit,
    # verified image_uri (digest preferred) and never fabricates a floating :latest.
    # With no image_uri we do NOT silently skip: that could be mistaken for a complete
    # deployment (esp. `cdk deploy --all`). Phase-1 (deploy ONLY the artifacts stack,
    # before any image digest exists) must OPT IN explicitly with artifacts_only=true;
    # anything else fails closed with guidance.
    _artifacts_only = str(
        app.node.try_get_context("artifacts_only") or ""
    ).lower() in ("1", "true", "yes")
    if _artifacts_only:
        print(
            "[app] artifacts_only=true — synthesizing GR00TRLArtifactsStack only "
            "(phase-1: build the image + stage data before the EKS backend exists). "
            "GR00TRLEKSStack is intentionally skipped.",
            file=sys.stderr,
        )
    else:
        raise SystemExit(
            "compute_backend=eks requires ONE of:\n"
            "  --context image_uri=<acct>.dkr.ecr.<region>.amazonaws.com/"
            "gr00t-rl-unified@sha256:<digest>   (deploy the EKS backend), OR\n"
            "  --context artifacts_only=true   (phase-1: deploy GR00TRLArtifactsStack "
            "only, before the image exists).\n"
            "Refusing to synth ambiguously (no ':latest' fallback). Normally you run "
            "infra/prepare-artifacts.sh, which builds+verifies the image and deploys "
            "the EKS stack with the resolved digest."
        )
elif compute_backend == "eks":
    stack = EKSKubeRayStack(
        app,
        "GR00TRLEKSStack",
        vpc_id=app.node.try_get_context("vpc_id"),
        s3_data_bucket=app.node.try_get_context("s3_data_bucket"),
        image_uri=app.node.try_get_context("image_uri"),
        num_rollout_workers=int(
            app.node.try_get_context("num_rollout_workers") or 4
        ),
        fsx_capacity_gib=int(
            app.node.try_get_context("fsx_capacity_gib") or 1200
        ),
        learner_instance_type=(
            app.node.try_get_context("learner_instance_type") or "g6e.48xlarge"
        ),
        rollout_instance_type=(
            app.node.try_get_context("rollout_instance_type") or "g6e.8xlarge"
        ),
        capacity_reservation_id=app.node.try_get_context("capacity_reservation_id"),
        mode=app.node.try_get_context("mode") or "train",
        eval_ckpt=app.node.try_get_context("eval_ckpt"),
        model_path=app.node.try_get_context("model_path"),
        # None => train head pod omits VAL_CHECK_INTERVAL => historical behavior.
        val_check_interval=app.node.try_get_context("val_check_interval"),
        # None => entrypoint default ENVS_PER_WORKER=32 (total_num_envs =
        # num_rollout_workers * 32). Lower it to shrink the co-located rollout
        # L40S GPU footprint (32/GPU OOMs the 46 GiB L40S at the Eagle lm_head).
        envs_per_worker=app.node.try_get_context("envs_per_worker"),
        # None => entrypoint default MAX_EPOCHS=1000. Set to N to bound a train
        # run to N global_steps (cost-bounded / deterministic stop) — pairs with
        # save_interval so a short run still writes an eval-able checkpoint.
        max_epochs=app.node.try_get_context("max_epochs"),
        eval_total_envs=app.node.try_get_context("eval_total_envs"),
        eval_actor_gbs=app.node.try_get_context("eval_actor_gbs"),
        task_description=app.node.try_get_context("task_description"),
        eval_inject_noise=app.node.try_get_context("eval_inject_noise"),
        noise_level=app.node.try_get_context("noise_level"),
        # None => entrypoint default SAVE_VIDEO=False (eval videos off). Set
        # --context save_video=true to write MP4 rollouts to ${LOG_DIR}/video/eval/.
        save_video=app.node.try_get_context("save_video"),
        kuberay_version=app.node.try_get_context("kuberay_version") or "1.1.0",
        rollout_subnet_ids=app.node.try_get_context("rollout_subnet_ids"),
        eval_learner_subnet_ids=app.node.try_get_context("eval_learner_subnet_ids"),
        fsx_subnet_id=app.node.try_get_context("fsx_subnet_id"),
        # Container Insights pod logs to CloudWatch: ON by default; disable with
        # --context enable_cloudwatch_logs=false. Any of false/0/no disables.
        enable_cloudwatch_logs=(
            str(app.node.try_get_context("enable_cloudwatch_logs")).lower()
            not in ("false", "0", "no")
        ),
        env=env,
    )
else:
    raise ValueError(
        f"Unknown compute_backend: {compute_backend}. "
        "Supported: 'batch-mnp', 'sagemaker', 'eks'"
    )

app.synth()
