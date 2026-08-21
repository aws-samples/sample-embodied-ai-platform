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
  rollout_instance_type - EC2 instance type for rollout node group (default: g6e.4xlarge)
  compute_backend       - "batch-mnp" (default), "sagemaker", or "eks"
  "mode"                - "train" (default) or "eval" — routes the EKS backend to training or standalone eval
  "eval_ckpt"           - Full path to actor checkpoint (.pt) for mode=eval; ignored for mode=train
  "model_path"          - Full FSx-visible path (mount root /mnt/fsx) to the model dir. When omitted, the entrypoint's default (RL model) is used. Enables SFT/RL model swap without editing the entrypoint.
  "eval_total_envs"     - Optional integer. When set, appends EVAL_TOTAL_ENVS=<val> to the eval head-pod env, which entrypoint-eks.sh reads to override env.eval.total_num_envs. Omit to fall through to the yaml default (64). Used by Plan 07.1.1-06 for the N=100 diagnostic sweep.
  "eval_actor_gbs"      - Optional integer. When set, appends EVAL_ACTOR_GBS=<val> to the eval head-pod env, which entrypoint-eks.sh reads to override actor.global_batch_size. Omit to fall through to the yaml default (2048). Used by Plan 07.1.1-07 for the N=100 diagnostic sweep: EVAL_ACTOR_GBS=1280 with mbs=128 unlocks world_size=10 required for 10-node topology (satisfies RLinf validator gbs % (mbs * world_size) == 0). Safe in eval mode because actor.global_batch_size is consumed only by the FSDP trainer, which eval_embodied_agent.py never spawns.
  "task_description"    - Optional string. When set, appends TASK_DESCRIPTION=<val> to the eval head-pod env, which entrypoint-eks.sh reads to override env.eval.init_params.task_description. Omit to fall through to the yaml default ("install trocar from box"). Used by Plan 07.1.1-08 to diagnose whether NVIDIA's 83% SFT Stage 1 baseline was measured with a different task description (e.g., "assemble trocar from tray" — the SFT dataset's canonical annotation). Additive-only; D-09 reversibility preserved when unset.
  "eval_inject_noise"   - Optional string. When set to "true", appends EVAL_INJECT_NOISE=true to the eval head-pod env. On the FSx-side, the patched RLinf gr00t_action_model.py reads this env var and, when true, uses the train-mode noise formula (flow_sde + noise_level yaml default) at eval time instead of the deterministic x_t_std=0 branch. Used by Plan 07.1.1-13 to test whether NVIDIA's 83% SFT Stage 1 was measured on a code state where temperature_eval actually applied at eval time. Requires the corresponding FSx-side patch (Plan 13 Task 1) to have any effect. Additive-only; D-09 reversibility preserved when unset — the patched Python code preserves original eval-mode behavior byte-identically when EVAL_INJECT_NOISE is unset.
  "noise_level"         - Optional string (float). When set, appends NOISE_LEVEL=<val> to the eval head-pod env, and entrypoint-eks.sh emits ++actor.rl_head_config.noise_level=<val> as a Hydra override. Used only when eval_inject_noise=true. Yaml default is 0.3 (flow_sde). Plan 07.1.1-13 Stage B sweeps {0.1, 0.6, 1.0} to characterize the noise-vs-success curve. Additive-only; unset preserves shipped yaml default.
  "kuberay_version"     - Optional string (EKS only). KubeRay operator Helm chart version. Default "1.1.0" keeps the validated eks stack byte-identical (A9-3) — a deploy without this flag synthesizes exactly today's frozen 1.1.0 stack. Pass --context kuberay_version=1.2.0 for the async deploy only (KubeRay 1.2.0 is needed for async node-recovery per PRD §8). Reversibility (D-06): omit the flag to fall back to the 1.1.0 default; no in-place mutation of the validated stack.
  "rollout_subnet_ids"  - Optional string (EKS only). Default-off capacity-resilient rollout knob (Phase 12). Unset keeps the RolloutNodes node group on the single-AZ FSx subnet (byte-identical synth to today). Set to one or more comma-separated private subnet IDs to place the rollout fleet cross-AZ (e.g. us-east-2b) while the learner (pinned to the 2a Capacity Block) and FSx stay in us-east-2a. Applies to the RolloutNodes NG ONLY — the (CB) learner NG stays on the FSx subnet, and the eval-learner NG stays on the FSx subnet unless eval_learner_subnet_ids is also set. Reversible by omitting the flag; subnet IDs are supplied at deploy via --context and never committed.
  "eval_learner_subnet_ids" - Optional string (EKS only). Default-off capacity-resilient EVAL knob (Phase 13). Unset keeps the EvalLearnerNodes NG (which runs the eval head pod) on the single-AZ FSx subnet (byte-identical synth). Set to comma-separated private subnet IDs to place the eval-learner in another AZ (e.g. us-east-2b) when the FSx AZ is g6e-capacity-dry; FSx stays in us-east-2a and is read cross-AZ (the static CSI PV has no topology affinity). Pair with rollout_subnet_ids=<same subnet> so the eval head + rollout workers co-locate intra-AZ. Reversible by omitting; subnet IDs supplied at deploy via --context, never committed.
  "fsx_subnet_id"       - Optional string (EKS only). Pins the single-AZ FSx-Lustre filesystem (and the CB/on-demand learner NG that co-locates with it) to a SPECIFIC private subnet instead of the first PRIVATE_WITH_EGRESS subnet (index 0). Set it to the subnet in the AZ that actually holds g6e/H100 capacity so FSx and the learner land together. Unset keeps the historical select_subnets(...).subnets[0] default (byte-identical synth). Subnet ID supplied at deploy via --context, never committed.
"""
import os
import sys
import aws_cdk as cdk
from mnp_batch_stack import RLBatchMNPStack
from eks_kuberay_stack import EKSKubeRayStack
from artifacts_stack import GR00TRLArtifactsStack

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
if compute_backend in ("eks", "hyperpod-eks"):
    GR00TRLArtifactsStack(
        app,
        "GR00TRLArtifactsStack",
        s3_data_bucket=app.node.try_get_context("s3_data_bucket"),
        image_tag=app.node.try_get_context("image_tag") or "latest",
        env=env,
    )

if compute_backend in ("batch-mnp", "sagemaker"):
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
    # When no image_uri is in context we SKIP instantiating it, so that phase-1
    # `cdk deploy GR00TRLArtifactsStack --context compute_backend=eks ...` (build the
    # image; no digest exists yet) can synth+deploy on its own. To deploy the EKS
    # backend, use infra/prepare-artifacts.sh (builds+verifies the image, then deploys
    # with --context image_uri=<resolved @sha256 digest>) or pass image_uri yourself.
    print(
        "[app] compute_backend=eks but no image_uri context — skipping GR00TRLEKSStack "
        "(pure image consumer). Expected when deploying GR00TRLArtifactsStack first. "
        "Deploy the EKS backend via infra/prepare-artifacts.sh or pass "
        "--context image_uri=<...@sha256:digest>.",
        file=sys.stderr,
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
            app.node.try_get_context("rollout_instance_type") or "g6e.4xlarge"
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
        eval_total_envs=app.node.try_get_context("eval_total_envs"),
        eval_actor_gbs=app.node.try_get_context("eval_actor_gbs"),
        task_description=app.node.try_get_context("task_description"),
        eval_inject_noise=app.node.try_get_context("eval_inject_noise"),
        noise_level=app.node.try_get_context("noise_level"),
        kuberay_version=app.node.try_get_context("kuberay_version") or "1.1.0",
        rollout_subnet_ids=app.node.try_get_context("rollout_subnet_ids"),
        eval_learner_subnet_ids=app.node.try_get_context("eval_learner_subnet_ids"),
        fsx_subnet_id=app.node.try_get_context("fsx_subnet_id"),
        env=env,
    )
else:
    raise ValueError(
        f"Unknown compute_backend: {compute_backend}. "
        "Supported: 'batch-mnp', 'sagemaker', 'eks'"
    )

app.synth()
