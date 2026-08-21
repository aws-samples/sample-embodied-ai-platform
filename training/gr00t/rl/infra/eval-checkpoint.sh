#!/usr/bin/env bash
#
# eval-checkpoint.sh — self-serve checkpoint-eval stop/go runbook for the SFT->RL run.
#
# Given a checkpoint (an S3 URI / FSx path to an actor .pt, OR a base/SFT model dir),
# this runs the eval-path 4-stage success_stage sweep (MODE=eval, g6e-only, N=64),
# reads per-stage eval/success_once from FSx-persisted TensorBoard events, computes
# Wilson 95% confidence intervals, and prints a per-stage table with a PASS / CONTINUE
# verdict against the same-apparatus reference 100 / 93.75 / 85.9 / 78.1 — NVIDIA's own RL
# checkpoint re-measured on THIS N=64 eval path (inside the Wilson 95% CI of NVIDIA's
# *posted* row 100 / 92 / 85 / 82, which was measured on 100 scenes). This IS the "the agent
# monitors" mechanism made reproducible with standard tooling + a committed script.
#
# ─────────────────────────────────────────────────────────────────────────────
#  SAFETY MODEL — READ THIS FIRST
# ─────────────────────────────────────────────────────────────────────────────
# By DEFAULT this script runs in DRY-RUN mode: it PRINTS every mutating / AWS /
# cdk / kubectl / deploy command it *would* run, prefixed with "DRY-RUN >>", and
# executes NOTHING. No cluster is provisioned, no GPU is burned, no dollar is spent.
#
# To actually run the (PAID) eval sweep you must pass --execute AND type a
# confirmation string. Every AWS/CDK/kubectl/deploy-mutating line is routed through
# the run()/run_sh() guard, so it is impossible to spend anything without --execute.
#
#   Dry-run (default, safe):   ./eval-checkpoint.sh --ckpt s3://<bucket>/<...>/full_weights.pt
#   Real (PAID) eval:          ./eval-checkpoint.sh --ckpt s3://... --execute
#   Syntax check only:         bash -n ./eval-checkpoint.sh
#   Usage:                     ./eval-checkpoint.sh --help
#
# LIFECYCLE OWNERSHIP (SKILL Known Hiccup #14 — the paid-for lesson):
#   A crashed or hung eval (e.g. the Isaac Sim `omni.usd::newStage` fatal) leaves the
#   RayCluster head pod `Running`, so an unattended eval burns g6e INDEFINITELY. The
#   --execute path therefore OWNS the eval cluster's lifecycle end to end: a bash
#   `trap ... EXIT` ALWAYS tears the eval cluster down / scales it to 0 on EVERY exit
#   path (normal completion, non-zero failure, and interrupt), and a MAX_RUNTIME
#   hard-deadline guard force-tears-down and exits non-zero if any sweep overruns.
#   No eval cluster may outlive this script.
#
# PUBLIC-MIRROR RULE:
#   This script is destined for the public mirror. It contains NO internal IDs.
#   Account / VPC / FSx / bucket / capacity-reservation ids are ALL supplied
#   via env or args. Nothing internal is baked in.
#
# ─────────────────────────────────────────────────────────────────────────────
#  BACKEND: EKS heterogeneous stack (GR00TRLEKSStack)
# ─────────────────────────────────────────────────────────────────────────────
# This runbook targets the plain-EKS heterogeneous stack (GR00TRLEKSStack). It
# PROVISIONS by invoking `cdk deploy GR00TRLEKSStack --context compute_backend=eks
# --context mode=eval ...` directly (per the deploy-eks-training SKILL), and its
# capacity is EKS *managed nodegroups*, not a SageMaker cluster — so TEARDOWN scales
# the eval-learner + rollout nodegroups to 0 via `aws eks update-nodegroup-config ...
# desiredSize=0`. vpc_id / s3_data_bucket / image_uri are read from env exactly as
# the eks SKILL shows.
#
# ─────────────────────────────────────────────────────────────────────────────
#  CAVEAT: ONE STACK CANNOT BE BOTH train AND eval AT ONCE
# ─────────────────────────────────────────────────────────────────────────────
# A single GR00TRLEKSStack bakes the head pod's MODE (train vs eval) at DEPLOY time
# (the RayCluster head-pod shape + env are selected by is_eval at synth). You cannot
# run one GR00TRLEKSStack in train and eval mode simultaneously. Therefore
# eval-checkpoint.sh is an END-OF-RUN / SEPARATE-WINDOW tool: run it after a training
# run (or in its own deploy window), not against a live training stack.
#
# For IN-FLIGHT eval DURING a training run, do NOT use this script — use the
# `val_check_interval=N` knob instead: it streams the aggregate eval metric into the
# SAME TensorBoard as the training run, no second stack required. The two are
# complementary, not interchangeable:
#   * eval-checkpoint.sh  -> the per-stage success_stage CDF: the exact
#                            NVIDIA-comparable 4-number breakdown (stages 1/2/3/4).
#   * val_check_interval=N -> the in-flight AGGREGATE eval number, live in TensorBoard.
#
# ─────────────────────────────────────────────────────────────────────────────
#  DESIGN NOTES / SELF-CONTAINED STAGE MACHINE  (validated on the 2026-08-20 run)
# ─────────────────────────────────────────────────────────────────────────────
# This backend drives the ENTIRE per-stage sweep in ONE `--execute` invocation with NO
# human/agent in the loop (public-mirror requirement). It deploys the stack + forms the
# RayCluster ONCE, then for each requested stage: patches success_stage on FSx -> REFORMS
# the RayCluster (reform_raycluster_eks) so a fresh head re-reads the stage -> waits for a
# FRESH eval/success_once in a new FSx LOG_DIR -> reads it. Key realities it handles:
# R1. RLinf's only_eval head runs once then EXITS; KubeRay restarts it with a NEW Ray GCS
#     cluster-id that ORPHANS workers ("GCS authentication error", can't re-register). The
#     reform deletes the Ray pods and polls `ray status`, periodically recycling not-Ready
#     workers until num_nodes re-register against the stabilized head. Without this, stages
#     2-4 would silently report stage 1's number.
# R2. total_num_envs (N) MUST be divisible by num_nodes = 1 + EKS_NUM_ROLLOUT_WORKERS
#     (env is placed on every node); else the eval head asserts + crash-loops. Validated
#     up-front with a "use one of these worker counts" hint (e.g. N=64 => 1,3,7,15,31,63).
# R3. EVAL_CKPT/MODEL_PATH must be the DRA FSx PATH, not an s3:// URI (entrypoint does a
#     local test -f). An s3://<S3_DATA_BUCKET>/<key> arg is auto-translated to ${FSX_MOUNT}/<key>.
# R4. FSx-side ops (patch/read) run on the UNIFIED image with FSx mounted — auto-discovered
#     from a Running RayCluster worker (no separate helper needed); pin FSX_HELPER_POD only
#     if you maintain a dedicated unified-image helper.
# R5. Teardown scales NGs to 0 OUTSIDE CloudFormation; a later identical `cdk deploy` won't
#     un-drift them (no template change) — restore_eval_ng_desired() re-asserts desired sizes.
# SAFETY: every stage is deadline-guarded (REFORM_TIMEOUT + per-stage MAX_RUNTIME) and a
# FRESHNESS ASSERTION fails closed (won't report a metric unless a NEW LOG_DIR appeared).
#
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# =============================================================================
#  Configuration (override via env vars before invoking — NO internal defaults)
# =============================================================================
# Region / account: no baked default — the account must be supplied by the operator.
REGION="${AWS_REGION:-}"        # REQUIRED — export AWS_REGION (no baked default; fail closed below)
ACCOUNT_ID="${CDK_DEFAULT_ACCOUNT:-}"

# Required CDK/deploy context, all parameterized (no safe defaults — operator supplies):
VPC_ID="${VPC_ID:-}"
S3_DATA_BUCKET="${S3_DATA_BUCKET:-}"
IMAGE_URI="${IMAGE_URI:-}"

# Resource names (these are stack resource names, not secrets — same as the deploy script).
FSX_MOUNT="${FSX_MOUNT:-/mnt/fsx}"
CONFIG_NAME="${CONFIG_NAME:-isaaclab_ppo_gr00t_assemble_trocar}"
COMPUTE_BACKEND="eks"
K8S_NAMESPACE="${K8S_NAMESPACE:-training}"
# FSx-side ops (patch success_stage, read tfevents) need the UNIFIED image
# (bash+python3+tensorboard) with FSx mounted. Default EMPTY => the script auto-discovers
# a Running RayCluster worker pod (which IS the unified image, FSx mounted) so it is
# SELF-CONTAINED with no separate helper to pre-create. Pin FSX_HELPER_POD only if you
# maintain a dedicated unified-image helper pod (then it is used as-is, default container).
FSX_HELPER_POD="${FSX_HELPER_POD:-}"
# Per-stage RayCluster reform tuning — the script drives stage transitions itself
# (patch success_stage -> reform head+workers -> read metric), no human/agent in the loop.
REFORM_TIMEOUT="${REFORM_TIMEOUT:-900}"      # max seconds to reach num_nodes Ray nodes/stage
REFORM_RECYCLE_EVERY="${REFORM_RECYCLE_EVERY:-120}"  # re-recycle not-joined workers every N s
RAY_HEAD_CONTAINER="${RAY_HEAD_CONTAINER:-ray-head}"
RAY_WORKER_CONTAINER="${RAY_WORKER_CONTAINER:-ray-worker}"

# EKS resource names (stack resource names, not secrets — same as the deploy-eks-training
# SKILL). The eval-learner + rollout managed nodegroups are created by GR00TRLEKSStack;
# CDK auto-generates their physical names, so by default teardown DISCOVERS them
# (list-nodegroups + substring match). Pin them explicitly via EKS_EVAL_LEARNER_NG /
# EKS_ROLLOUT_NG to skip discovery.
EKS_STACK_NAME="${EKS_STACK_NAME:-GR00TRLEKSStack}"
EKS_CLUSTER_NAME="${EKS_CLUSTER_NAME:-gr00t-rl-eks}"
EKS_EVAL_LEARNER_NG="${EKS_EVAL_LEARNER_NG:-}"
EKS_ROLLOUT_NG="${EKS_ROLLOUT_NG:-}"

# EKS eval TOPOLOGY (the silent-nodegroup-churn fix — Phase 12).
# The eval `cdk deploy` (launch_eval_eks) MUST pin the rollout/eval-learner
# instance type and worker count. If omitted, CDK falls through to app.py DEFAULTS
# (rollout_instance_type=g6e.4xlarge, num_rollout_workers=4) and CloudFormation
# RECREATES the EvalLearnerNodes + RolloutNodes managed nodegroups at the wrong,
# too-small size. The eval head pod requests cpu:24/mem:100Gi (eks_kuberay_stack.py
# eval_head_pod) which CANNOT fit a g6e.4xlarge (16 vCPU/128 GiB) → head pod stays
# Pending → the eval hangs and burns g6e until the MAX_RUNTIME guard force-tears-down.
# BOTH the EvalLearnerNodes and RolloutNodes NGs are sized by rollout_instance_type
# (a single knob pins both); g6e.8xlarge (32 vCPU/256 GiB) satisfies the head request.
EKS_ROLLOUT_INSTANCE_TYPE="${EKS_ROLLOUT_INSTANCE_TYPE:-g6e.8xlarge}"
# Default 7 rollout workers => 8 nodes (7 rollout + 1 eval-learner). 8 divides the default
# N=64 (env divisibility, see below) and gives 8 envs/node. If you change --n, pick a
# worker count so (1 + workers) divides N (the script validates and lists valid counts).
EKS_NUM_ROLLOUT_WORKERS="${EKS_NUM_ROLLOUT_WORKERS:-7}"
# Eval is g6e-only: we deliberately DO NOT pass capacity_reservation_id, so the
# CB-backed p5 LearnerNodes NG stays absent from the eval template (a passed CR
# would spin up an idle p5 at desired=1 that the eval never uses — the eval head
# runs on the g6e EvalLearnerNodes). Set CAPACITY_RESERVATION_ID (+ optionally
# EKS_LEARNER_INSTANCE_TYPE) ONLY if you intend to co-provision / preserve the
# CB-backed training learner NG during the eval (wasteful for a pure eval).
CAPACITY_RESERVATION_ID="${CAPACITY_RESERVATION_ID:-}"
EKS_LEARNER_INSTANCE_TYPE="${EKS_LEARNER_INSTANCE_TYPE:-p5.48xlarge}"
# Other STRUCTURAL context keys the eval deploy must not silently revert (Codex
# red-team). These default to the SAME app.py defaults, so an unset override is a
# byte-identical no-op for a stack deployed at defaults — but if the training stack
# was deployed with a non-default value (e.g. kuberay 1.2.0 for the async path, or a
# cross-AZ rollout subnet), you MUST set the matching env var here or the eval
# `cdk deploy` will churn/downgrade that resource. rollout_subnet_ids is passed
# ONLY when set (unset == app.py default == the single FSx-AZ subnet).
EKS_FSX_CAPACITY_GIB="${EKS_FSX_CAPACITY_GIB:-1200}"
EKS_KUBERAY_VERSION="${EKS_KUBERAY_VERSION:-1.1.0}"
EKS_ROLLOUT_SUBNET_IDS="${EKS_ROLLOUT_SUBNET_IDS:-}"
# Capacity-resilient EVAL knob (Phase 13): when the FSx AZ (us-east-2a) is g6e-dry,
# set BOTH EKS_EVAL_LEARNER_SUBNET_IDS and EKS_ROLLOUT_SUBNET_IDS to a same other-AZ
# private subnet (e.g. us-east-2b) so the eval head + rollout workers co-locate there;
# FSx stays in 2a and is read cross-AZ. Unset == app.py default == the FSx-AZ subnet.
EKS_EVAL_LEARNER_SUBNET_IDS="${EKS_EVAL_LEARNER_SUBNET_IDS:-}"

# The reused FSx success_stage patch, referenced-not-copied (D-15). SCRIPT_DIR anchors
# the infra dir (used as the cwd for `cdk deploy`).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# patch-success-stage.sh lives on the FSx mount inside the fsx-helper pod (sha256-verified
# copy of the git-tracked source). Override PATCH_SCRIPT for a different mount layout.
PATCH_SCRIPT="${PATCH_SCRIPT:-${FSX_MOUNT}/scratch/step-a/patch-success-stage.sh}"

# =============================================================================
#  Defaults for the sweep (overridable via args)
# =============================================================================
CKPT=""                         # --ckpt  : s3:// URI or FSx path to an actor .pt  (=> EVAL_CKPT)
MODEL_PATH_ARG=""               # --model-path : FSx model dir for a base/SFT eval  (=> MODEL_PATH)
N="64"                          # --n     : envs per stage (NVIDIA WeChat-confirmed N=64)
REF_ROW="100,93.75,85.9,78.1"   # --ref : same-apparatus ref = NVIDIA's RL ckpt on this N=64 path (posted headline: 100/92/85/82, 100 scenes)
STAGES="1 2 3 4"                # --stages: which success_stage values to sweep
# MAX_RUNTIME: hard deadline PER STAGE sweep (seconds). Default 3h. If an eval overruns
# (e.g. a crashed head pod stuck Running), the guard force-tears-down and exits non-zero.
MAX_RUNTIME="${MAX_RUNTIME:-10800}"
EXECUTE=0

# =============================================================================
#  Arg parsing
# =============================================================================
usage() {
  # Print the top-of-file doc block, then the explicit arg synopsis.
  grep -E '^# ' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'USAGE'

eval-checkpoint.sh — usage
  --backend eks              Compute backend (default and only: eks). Provisions via
                             `cdk deploy GR00TRLEKSStack --context mode=eval ...` and
                             tears down with `aws eks update-nodegroup-config ...
                             desiredSize=0`. CAVEAT: one GR00TRLEKSStack cannot be
                             train AND eval at once (head-pod mode is baked at deploy);
                             this is an end-of-run / separate-window tool. For in-flight
                             eval during a training run use `val_check_interval=N`
                             (streams the aggregate eval into the same TensorBoard).
  --ckpt <s3-uri|fsx-path>   Actor .pt checkpoint to eval (passed as EVAL_CKPT).
  --model-path <fsx-dir>     Base/SFT model dir to eval (passed as MODEL_PATH). Mutually
                             exclusive with --ckpt; use one or the other.
  --n <N>                    Envs per stage (default 64).
  --ref "100,93.75,85.9,78.1"  Reference row (default = NVIDIA's RL ckpt on this N=64 path; posted headline 100/92/85/82).
  --stages "1 2 3 4"         success_stage values to sweep (default "1 2 3 4").
  --max-runtime <sec>        Hard per-stage deadline; force-teardown on overrun (default 10800).
  --execute                  Actually run the PAID g6e eval (default = dry-run).
  -h | --help                Show this help and exit (zero AWS calls).

TOPOLOGY env vars (silent-NG-churn fix — override before invoking):
  EKS_ROLLOUT_INSTANCE_TYPE  Rollout + eval-learner instance type (default g6e.8xlarge).
                             MUST be >= g6e.8xlarge: the eval head pod requests
                             cpu:24/mem:100Gi, which a g6e.4xlarge (the app.py
                             default) cannot fit → head Pending → eval hangs.
  EKS_NUM_ROLLOUT_WORKERS    Rollout worker count (default 8).
  CAPACITY_RESERVATION_ID    (optional) Set ONLY to co-provision/preserve the
                             CB-backed p5 learner NG during eval; unset = g6e-only
                             (no idle p5). EKS_LEARNER_INSTANCE_TYPE pairs with it.
USAGE
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)     COMPUTE_BACKEND="${2:-}"; shift 2 ;;
    --ckpt)        CKPT="${2:-}"; shift 2 ;;
    --model-path)  MODEL_PATH_ARG="${2:-}"; shift 2 ;;
    --n)           N="${2:-}"; shift 2 ;;
    --ref)         REF_ROW="${2:-}"; shift 2 ;;
    --stages)      STAGES="${2:-}"; shift 2 ;;
    --max-runtime) MAX_RUNTIME="${2:-}"; shift 2 ;;
    --execute)     EXECUTE=1; shift ;;
    -h|--help)     usage ;;
    *) echo "Unknown argument: $1 (use --help)"; exit 2 ;;
  esac
done

# =============================================================================
#  Echo helpers + run() guard
# =============================================================================
say()  { echo ""; echo "==> $*"; }
ok()   { echo "    [OK]   $*"; }
warn() { echo "    [WARN] $*"; }
die()  { echo ""; echo "    [FATAL] $*" >&2; exit 1; }

# run(): the single choke point for every mutating / AWS / cdk / kubectl command.
#   - Dry-run (default): prints the command prefixed "DRY-RUN >>", does NOT execute.
#   - --execute: prints it prefixed "RUN >>" and executes it.
run() {
  if [[ "$EXECUTE" -eq 1 ]]; then
    echo "    RUN >> $*"
    "$@"
  else
    echo "    DRY-RUN >> $*"
  fi
}

# run_sh(): same guard for a shell string (env-prefixed invocations, pipes, redirects).
run_sh() {
  if [[ "$EXECUTE" -eq 1 ]]; then
    echo "    RUN >> $1"
    bash -c "$1"
  else
    echo "    DRY-RUN >> $1"
  fi
}

# run_capture(): guarded read whose stdout we need in --execute (e.g. TB metric read).
#   - Dry-run: prints "DRY-RUN >>" and returns empty (caller uses a placeholder).
#   - --execute: prints "RUN >>", executes, and echoes captured stdout to our stdout.
run_capture() {
  if [[ "$EXECUTE" -eq 1 ]]; then
    echo "    RUN >> $1" >&2
    bash -c "$1"
  else
    echo "    DRY-RUN >> $1" >&2
  fi
}

# run_capture_argv(): array-safe capture (Codex red-team). Same contract as run_capture
# but takes an argv ARRAY, executed directly with "$@" — NO `bash -c` reparse. Required
# whenever an argument contains characters the shell would eat, e.g. a `python3 -c`
# program whose source has its own double quotes (`os.environ.get("EC_LOG_DIR", "")`).
run_capture_argv() {
  if [[ "$EXECUTE" -eq 1 ]]; then
    echo "    RUN >> $*" >&2
    "$@"
  else
    echo "    DRY-RUN >> $*" >&2
  fi
}

# =============================================================================
#  Input validation
# =============================================================================
case "$COMPUTE_BACKEND" in
  eks) ;;
  *) die "--backend must be 'eks' (the only supported backend; got '$COMPUTE_BACKEND')." ;;
esac
if [[ -n "$CKPT" && -n "$MODEL_PATH_ARG" ]]; then
  die "Pass EITHER --ckpt OR --model-path, not both."
fi
if [[ -z "$CKPT" && -z "$MODEL_PATH_ARG" ]]; then
  die "Provide a checkpoint: --ckpt <s3-uri|fsx-path> OR --model-path <fsx-dir>."
fi
[[ "$N" =~ ^[0-9]+$ && "$N" -gt 0 ]] || die "--n must be a positive integer (got '$N')."
[[ "$MAX_RUNTIME" =~ ^[0-9]+$ && "$MAX_RUNTIME" -gt 0 ]] || die "--max-runtime must be a positive integer seconds (got '$MAX_RUNTIME')."
[[ -n "$REGION" ]] || die "no region — export AWS_REGION before invoking (no baked default)."

# STAGES validation: non-empty AND every token in 1..4 (an empty/whitespace --stages
# would otherwise sweep zero stages and print a bogus "PASS"). Enforced for dry-run too.
_num_stages=$(echo "$STAGES" | wc -w)
[[ "$_num_stages" -ge 1 ]] || die "--stages must name at least one stage in 1..4 (got '${STAGES}')."
for _s in $STAGES; do case "$_s" in 1|2|3|4) ;; *) die "--stages entries must each be in 1..4 (got '$_s')." ;; esac; done

# INJECTION CLOSE (Codex red-team): every env-overridable identifier that is interpolated
# into a run_sh()/run_capture() `bash -c` string (the NG discovery/teardown loops and the
# FSx-read helpers) must be free of shell metacharacters. The eks `cdk deploy` itself is
# already array-safe (launch_eval_eks), but these names still flow through a shell string.
_safe_id='^[A-Za-z0-9._/-]+$'
for _pair in "AWS_REGION:${REGION}" "EKS_CLUSTER_NAME:${EKS_CLUSTER_NAME}" \
             "EKS_STACK_NAME:${EKS_STACK_NAME}" "K8S_NAMESPACE:${K8S_NAMESPACE}" \
             "CONFIG_NAME:${CONFIG_NAME}" \
             "FSX_MOUNT:${FSX_MOUNT}"; do
  [[ "${_pair#*:}" =~ $_safe_id ]] || die "${_pair%%:*} contains unsafe characters ('${_pair#*:}') — must match ${_safe_id} (it is interpolated into a shell command)."
done
[[ -z "$ACCOUNT_ID" || "$ACCOUNT_ID" =~ ^[0-9]+$ ]] || die "CDK_DEFAULT_ACCOUNT must be numeric (got '$ACCOUNT_ID')."
# Optional identifiers — validate only when set (empty is allowed: FSX_HELPER_POD unset
# means auto-discover a worker; the NG names unset means discover-by-substring).
[[ -z "$FSX_HELPER_POD" || "$FSX_HELPER_POD" =~ $_safe_id ]] || die "FSX_HELPER_POD contains unsafe characters ('$FSX_HELPER_POD')."
[[ -z "$EKS_EVAL_LEARNER_NG" || "$EKS_EVAL_LEARNER_NG" =~ $_safe_id ]] || die "EKS_EVAL_LEARNER_NG contains unsafe characters ('$EKS_EVAL_LEARNER_NG')."
[[ -z "$EKS_ROLLOUT_NG" || "$EKS_ROLLOUT_NG" =~ $_safe_id ]] || die "EKS_ROLLOUT_NG contains unsafe characters ('$EKS_ROLLOUT_NG')."

# Validate eks topology overrides. The injection class is closed structurally by the
# array-safe launch_eval_eks (no bash -c string flattening); these checks additionally
# reject values that would provision the wrong/too-small/too-expensive fleet. Enforced
# for both dry-run and execute so a bad value can never reach `cdk deploy`.
# Allowlist: rollout runs Isaac Sim (needs RTX/L40S => g6e) AND the eval head pod
# requests cpu:24/mem:100Gi, so only g6e.8xlarge+ fit. A bare format regex is not
# enough (it accepted t3.micro) — pin to the head-capable L40S sizes.
case "$EKS_ROLLOUT_INSTANCE_TYPE" in
  g6e.8xlarge|g6e.12xlarge|g6e.16xlarge|g6e.24xlarge|g6e.48xlarge) ;;
  *) die "EKS_ROLLOUT_INSTANCE_TYPE must be a head-capable L40S type g6e.{8,12,16,24,48}xlarge (got '$EKS_ROLLOUT_INSTANCE_TYPE'): Isaac Sim needs RTX/L40S and the eval head needs >= g6e.8xlarge (cpu:24/mem:100Gi)." ;;
esac
# Worker count: positive AND ceilinged (a fat-fingered 100 would cost a fortune).
[[ "$EKS_NUM_ROLLOUT_WORKERS" =~ ^[0-9]+$ && "$EKS_NUM_ROLLOUT_WORKERS" -ge 1 && "$EKS_NUM_ROLLOUT_WORKERS" -le 16 ]] \
  || die "EKS_NUM_ROLLOUT_WORKERS must be an integer in 1..16 (got '$EKS_NUM_ROLLOUT_WORKERS'). Raise the in-script ceiling only deliberately — this is a paid g6e fleet."
[[ "$EKS_FSX_CAPACITY_GIB" =~ ^[0-9]+$ && "$EKS_FSX_CAPACITY_GIB" -gt 0 ]] \
  || die "EKS_FSX_CAPACITY_GIB must be a positive integer (got '$EKS_FSX_CAPACITY_GIB')."
[[ "$EKS_KUBERAY_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] \
  || die "EKS_KUBERAY_VERSION must look like 1.1.0 (got '$EKS_KUBERAY_VERSION')."
if [[ -n "$CAPACITY_RESERVATION_ID" ]]; then
  [[ "$CAPACITY_RESERVATION_ID" =~ ^cr-[0-9a-f]+$ ]] \
    || die "CAPACITY_RESERVATION_ID must look like cr-0123abcd... (got '$CAPACITY_RESERVATION_ID')."
  [[ "$EKS_LEARNER_INSTANCE_TYPE" =~ ^[a-z][a-z0-9]*\.[a-z0-9]+$ ]] \
    || die "EKS_LEARNER_INSTANCE_TYPE must be an instance type (got '$EKS_LEARNER_INSTANCE_TYPE')."
fi
if [[ -n "$EKS_ROLLOUT_SUBNET_IDS" ]]; then
  [[ "$EKS_ROLLOUT_SUBNET_IDS" =~ ^subnet-[0-9a-f]+(,subnet-[0-9a-f]+)*$ ]] \
    || die "EKS_ROLLOUT_SUBNET_IDS must be a comma-separated list of subnet-ids (got '$EKS_ROLLOUT_SUBNET_IDS')."
fi
if [[ -n "$EKS_EVAL_LEARNER_SUBNET_IDS" ]]; then
  [[ "$EKS_EVAL_LEARNER_SUBNET_IDS" =~ ^subnet-[0-9a-f]+(,subnet-[0-9a-f]+)*$ ]] \
    || die "EKS_EVAL_LEARNER_SUBNET_IDS must be a comma-separated list of subnet-ids (got '$EKS_EVAL_LEARNER_SUBNET_IDS')."
fi
# Reform knobs (Codex red-team): RAY_*_CONTAINER flow into kubectl exec / bash -c strings
# (validate as safe names); REFORM_TIMEOUT/REFORM_RECYCLE_EVERY enter Bash arithmetic
# (validate as plain positive integers so a value like 'x[$(cmd)]' can't inject).
[[ "$RAY_HEAD_CONTAINER" =~ ^[A-Za-z0-9._-]+$ ]] || die "RAY_HEAD_CONTAINER must be a container name (got '$RAY_HEAD_CONTAINER')."
[[ "$RAY_WORKER_CONTAINER" =~ ^[A-Za-z0-9._-]+$ ]] || die "RAY_WORKER_CONTAINER must be a container name (got '$RAY_WORKER_CONTAINER')."
[[ "$REFORM_TIMEOUT" =~ ^[0-9]+$ && "$REFORM_TIMEOUT" -gt 0 ]] || die "REFORM_TIMEOUT must be a positive integer (got '$REFORM_TIMEOUT')."
[[ "$REFORM_RECYCLE_EVERY" =~ ^[0-9]+$ && "$REFORM_RECYCLE_EVERY" -gt 0 ]] || die "REFORM_RECYCLE_EVERY must be a positive integer (got '$REFORM_RECYCLE_EVERY')."
# EVAL CONFIG DIVISIBILITY (RLinf validate_embodied_cfg): total_num_envs (N) MUST be
# divisible by env_world_size, which equals the eval node count num_nodes = 1 head/
# eval-learner + num_rollout_workers (the entrypoint places env on every node, 0..num_nodes-1).
# If not, the eval head asserts + crash-loops (never emits eval/success_once). Validate
# here and, if it fails, name the worker counts that DO divide N so the fix is obvious.
EKS_NUM_NODES=$(( EKS_NUM_ROLLOUT_WORKERS + 1 ))
if (( N % EKS_NUM_NODES != 0 )); then
  _valid_workers="$(python3 -c "n=$N; print(', '.join(str(d-1) for d in range(2,n+1) if n%d==0 and d-1>=1))" 2>/dev/null || echo '?')"
  die "eval requires N (total_num_envs=${N}) divisible by num_nodes (1 + EKS_NUM_ROLLOUT_WORKERS = ${EKS_NUM_NODES}); ${N} % ${EKS_NUM_NODES} = $(( N % EKS_NUM_NODES )). Set EKS_NUM_ROLLOUT_WORKERS to one of: ${_valid_workers} (each yields num_nodes | ${N})."
fi

# Parse the reference row (comma-separated percentages) into an array.
IFS=',' read -r -a REF_ARR <<< "$REF_ROW"

# =============================================================================
#  Banner
# =============================================================================
PAYLOAD="${CKPT:-$MODEL_PATH_ARG}"
PAYLOAD_KIND="EVAL_CKPT (actor .pt)"
[[ -n "$MODEL_PATH_ARG" ]] && PAYLOAD_KIND="MODEL_PATH (base/SFT model dir)"

# What the banner shows for the eval capacity target (EKS stack).
CLUSTER_DISPLAY="${EKS_CLUSTER_NAME} (stack ${EKS_STACK_NAME})"
# Node count = num_rollout_workers rollout + 1 eval-learner (both rollout_instance_type).
EKS_TOTAL_G6E=$(( EKS_NUM_ROLLOUT_WORKERS + 1 ))
EKS_TOPO_DISPLAY="${EKS_TOTAL_G6E}x ${EKS_ROLLOUT_INSTANCE_TYPE} (=${EKS_NUM_ROLLOUT_WORKERS} rollout +1 eval-learner)"
if [[ -n "$CAPACITY_RESERVATION_ID" ]]; then
  EKS_TOPO_DISPLAY="${EKS_TOPO_DISPLAY}; +1 CB learner ${EKS_LEARNER_INSTANCE_TYPE} (CR set)"
else
  EKS_TOPO_DISPLAY="${EKS_TOPO_DISPLAY}; no p5 (g6e-only eval)"
fi

echo "============================================================================"
echo " eval-checkpoint.sh — checkpoint-eval stop/go runbook (SFT->RL)"
echo " Payload:  ${PAYLOAD}   [${PAYLOAD_KIND}]"
echo " N/stage:  ${N}        Stages: ${STAGES}"
echo " Ref row:  ${REF_ROW}   (PASS iff we meet-or-exceed every swept stage)"
echo " Cluster:  ${CLUSTER_DISPLAY}   Region: ${REGION}   Backend: ${COMPUTE_BACKEND}"
[[ -n "$EKS_TOPO_DISPLAY" ]] && echo " Topology: ${EKS_TOPO_DISPLAY}"
echo " Max runtime/stage: ${MAX_RUNTIME}s (hard-deadline force-teardown guard)"
if [[ "$EXECUTE" -eq 1 ]]; then
  echo " MODE: *** EXECUTE — THIS PROVISIONS A PAID g6e EVAL CLUSTER AND COSTS MONEY ***"
else
  echo " MODE: DRY-RUN (mutating/AWS commands are PRINTED only; nothing runs, \$0 spend)"
  echo "       Pass --execute to actually run the paid eval."
fi
echo "============================================================================"

if [[ "$EXECUTE" -eq 1 ]]; then
  read -r -p "Type 'eval-checkpoint' to confirm a REAL (PAID) eval sweep: " _confirm
  [[ "$_confirm" == "eval-checkpoint" ]] || die "Confirmation mismatch — aborting."
  # Execute-mode preflight on the parameterized context (fail-fast).
  [[ -n "$ACCOUNT_ID" ]]     || die "CDK_DEFAULT_ACCOUNT is empty. Export it (no baked default in this mirror script)."
  [[ -n "$VPC_ID" ]]         || die "VPC_ID is empty. Export VPC_ID=<vpc-id>."
  [[ -n "$S3_DATA_BUCKET" ]] || die "S3_DATA_BUCKET is empty. Export S3_DATA_BUCKET=<bucket>."
  [[ -n "$IMAGE_URI" ]]      || die "IMAGE_URI is empty. Export IMAGE_URI=<account>.dkr.ecr.${REGION}.amazonaws.com/<repo>:<tag>."
fi

# =============================================================================
#  Eval-cluster lifecycle guard — teardown + max-runtime deadline (SKILL #14)
# =============================================================================
TEARDOWN_DONE=0

# teardown_eval_eks(): eks teardown. The eval capacity is EKS *managed nodegroups*
# (eval-learner + rollout), NOT a SageMaker cluster — so we scale them to 0 via
# `aws eks update-nodegroup-config ... desiredSize=0` (per the deploy-eks-training
# SKILL). This is the SKILL #14 idle-burn guard: a crashed/hung eval head pod cannot
# keep g6e nodes alive once their nodegroups are at desired=0. The EKS cluster + FSx +
# CFN stack are RETAINED (fast re-run). Idempotent + dry-run-visible.
teardown_eval_eks() {
  echo "    (eks: scale eval-learner + rollout managed nodegroups to desired=0; EKS cluster + stack retained)"
  echo "    (eks capacity is EKS managed nodegroups; SKILL #14 idle-burn guard)"
  if [[ -n "$EKS_EVAL_LEARNER_NG" && -n "$EKS_ROLLOUT_NG" ]]; then
    run aws eks update-nodegroup-config \
      --cluster-name "$EKS_CLUSTER_NAME" --region "$REGION" \
      --nodegroup-name "$EKS_EVAL_LEARNER_NG" --scaling-config minSize=0,desiredSize=0
    run aws eks update-nodegroup-config \
      --cluster-name "$EKS_CLUSTER_NAME" --region "$REGION" \
      --nodegroup-name "$EKS_ROLLOUT_NG" --scaling-config minSize=0,desiredSize=0
  else
    echo "    (nodegroup names not pinned via EKS_EVAL_LEARNER_NG/EKS_ROLLOUT_NG — discover + scale each)"
    run_sh "for ng in \$(aws eks list-nodegroups --cluster-name '${EKS_CLUSTER_NAME}' --region '${REGION}' --query \"nodegroups[?contains(@, 'NodegroupEva') || contains(@, 'NodegroupRol')]\" --output text); do aws eks update-nodegroup-config --cluster-name '${EKS_CLUSTER_NAME}' --region '${REGION}' --nodegroup-name \"\$ng\" --scaling-config minSize=0,desiredSize=0; done"
  fi
}

teardown_eval_cluster() {
  # Idempotent: only tear down once, even if called from both the sweep end and the trap.
  [[ "$TEARDOWN_DONE" -eq 1 ]] && return 0
  TEARDOWN_DONE=1
  say "TEARDOWN — releasing the g6e eval capacity (idempotent; runs on EVERY exit path)"
  # Scale managed nodegroups to 0 (not a SageMaker cluster op).
  teardown_eval_eks
  ok "Teardown issued — no eval capacity outlives this script."
}

# on_exit(): the trap. Fires on normal completion, non-zero failure, AND interrupt.
# It ALWAYS tears the eval cluster down so a crashed/hung eval cannot burn g6e (SKILL #14).
on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    warn "Non-zero/interrupted exit (rc=${rc}) — forcing eval-cluster teardown."
  fi
  teardown_eval_cluster
  exit "$rc"
}
trap on_exit EXIT INT TERM

# Arm the max-runtime hard deadline (announced now; enforced per-stage in wait_for_eval).
SWEEP_START_EPOCH="$(date +%s)"
say "Armed max-runtime hard-deadline guard: ${MAX_RUNTIME}s per stage"
echo "    A stage that overruns this deadline (e.g. Isaac Sim newStage crash leaves the"
echo "    head pod Running) triggers a force-teardown + non-zero exit via the trap."

# check_deadline(): abort (non-zero) if a stage sweep overruns MAX_RUNTIME. The trap
# then tears the cluster down. In dry-run this never trips (no real waiting).
check_deadline() {
  local stage_start="$1" now elapsed
  now="$(date +%s)"
  elapsed=$(( now - stage_start ))
  if [[ "$elapsed" -ge "$MAX_RUNTIME" ]]; then
    die "MAX_RUNTIME (${MAX_RUNTIME}s) exceeded for the current stage — force-teardown via trap."
  fi
}

# =============================================================================
#  STEP 1 — Resolve / verify the checkpoint (read-only, routed through run())
# =============================================================================
say "STEP 1 — Resolve checkpoint"
# s3_to_fsx(): translate an s3://<S3_DATA_BUCKET>/<key> URI to its DRA-mounted FSx path
# ${FSX_MOUNT}/<key>. The entrypoint loads EVAL_CKPT / MODEL_PATH as a LOCAL FSx path
# (it does `test -f`/`test -d`), NOT an s3 URI — passing a raw s3:// URI crashes the head
# ("EVAL_CKPT file not found: s3://..."). The S3 data bucket is DRA-linked to ${FSX_MOUNT},
# so the object at s3://<bucket>/<key> is visible on the mount at ${FSX_MOUNT}/<key>.
# Echoes the FSx path on success; dies (execute) / warns+passthrough (dry-run w/o bucket).
s3_to_fsx() {
  local uri="$1"
  if [[ -n "$S3_DATA_BUCKET" && "$uri" == "s3://${S3_DATA_BUCKET}/"* ]]; then
    echo "${FSX_MOUNT}/${uri#s3://${S3_DATA_BUCKET}/}"
  elif [[ -z "$S3_DATA_BUCKET" ]]; then
    warn "S3_DATA_BUCKET unset — cannot translate '${uri}' to its FSx path; set it before --execute." >&2
    echo "$uri"
  else
    die "'${uri}' is not under the DRA-linked S3_DATA_BUCKET (${S3_DATA_BUCKET}); it won't be visible on ${FSX_MOUNT}. Pass an FSx path, or a checkpoint under s3://${S3_DATA_BUCKET}/."
  fi
}

if [[ -n "$CKPT" ]]; then
  case "$CKPT" in
    s3://*)
      echo "    s3:// checkpoint — verify it exists on S3 (DRA-linked to FSx)."
      run aws s3 ls "$CKPT" --region "$REGION"
      # Entrypoint needs a LOCAL FSx path, not an s3:// URI — translate via the DRA map.
      EVAL_CKPT_VAL="$(s3_to_fsx "$CKPT")"
      echo "    entrypoint EVAL_CKPT (FSx path via DRA): ${EVAL_CKPT_VAL}"
      ;;
    /*)
      # DEFERRED existence check: STEP 1 runs BEFORE the cluster is up (no pod to exec on).
      # The head entrypoint validates EVAL_CKPT (test -f) at deploy and fails clearly if
      # missing, and the per-stage freshness assertion fails closed — so we do NOT exec here.
      echo "    FSx-path checkpoint — validated by the head entrypoint at deploy: ${CKPT}"
      EVAL_CKPT_VAL="$CKPT"
      ;;
    *) die "--ckpt must be an s3:// URI or an absolute FSx path (got '$CKPT')." ;;
  esac
  MODEL_PATH_VAL=""
else
  case "$MODEL_PATH_ARG" in
    s3://*)
      echo "    s3:// model dir — verify it exists on S3 (DRA-linked to FSx)."
      run aws s3 ls "$MODEL_PATH_ARG" --region "$REGION"
      MODEL_PATH_VAL="$(s3_to_fsx "$MODEL_PATH_ARG")"
      echo "    entrypoint MODEL_PATH (FSx path via DRA): ${MODEL_PATH_VAL}"
      ;;
    /*)
      # DEFERRED existence check (see --ckpt above): no pod exists before the deploy.
      echo "    Base/SFT model dir — validated by the head entrypoint at deploy: ${MODEL_PATH_ARG}"
      MODEL_PATH_VAL="$MODEL_PATH_ARG"
      ;;
    *) die "--model-path must be an s3:// URI or an absolute FSx path (got '$MODEL_PATH_ARG')." ;;
  esac
  EVAL_CKPT_VAL=""
fi

# =============================================================================
#  Per-stage helpers
# =============================================================================
# eval_results_parent(): where the eval-mode entrypoint writes LOG_DIRs on FSx
#   (entrypoint-eks.sh: .../results/<config>_<backend>_eval/<timestamp>/).
EVAL_RESULTS_PARENT="${FSX_MOUNT}/rl-training/results/${CONFIG_NAME}_${COMPUTE_BACKEND}_eval"

# resolve_fsx_pod(): set FSX_POD / FSX_CT to a pod that can run FSx-side ops (unified
# image: bash+python3+tensorboard, FSx mounted). Pinned FSX_HELPER_POD wins (default
# container); else auto-discover a Running RayCluster worker (container ray-worker) so the
# script is self-contained. In dry-run (no live pods) FSX_POD is a printable placeholder.
FSX_POD=""; FSX_CT=""
resolve_fsx_pod() {
  if [[ -n "$FSX_HELPER_POD" ]]; then FSX_POD="$FSX_HELPER_POD"; FSX_CT=""; return 0; fi
  if [[ "$EXECUTE" -eq 1 ]]; then
    # Poll for a Running RayCluster worker (right after a deploy/reform they may still be
    # starting). Deadline-guarded so we never hang if the cluster never forms.
    local waited=0
    while true; do
      FSX_POD="$(kubectl get pods -n "$K8S_NAMESPACE" -l ray-role=worker \
        -o jsonpath='{range .items[?(@.status.phase=="Running")]}{.metadata.name}{"\n"}{end}' 2>/dev/null | head -1)"
      [[ -n "$FSX_POD" ]] && { FSX_CT="$RAY_WORKER_CONTAINER"; return 0; }
      (( waited >= REFORM_TIMEOUT )) && die "no Running RayCluster worker pod for FSx ops after ${REFORM_TIMEOUT}s. Cluster never formed? Pin FSX_HELPER_POD to a unified-image helper if you maintain one."
      sleep 15; waited=$(( waited + 15 ))
    done
  else
    FSX_POD="<fsx-pod>"; FSX_CT="$RAY_WORKER_CONTAINER"
  fi
}

# ensure_backup(): the patch script's `set N` requires an .orig BACKUP to exist. Create it
# once (idempotent — tolerate "already exists"). Self-contained: no pre-step by the operator.
BACKUP_DONE=0
ensure_backup() {
  [[ "$BACKUP_DONE" -eq 1 ]] && return 0
  BACKUP_DONE=1
  resolve_fsx_pod
  echo "    Ensure success_stage BACKUP exists (idempotent; needed by patch-success-stage.sh 'set')"
  if [[ "$EXECUTE" -eq 1 ]]; then
    echo "    RUN >> kubectl exec ${FSX_POD} -- bash ${PATCH_SCRIPT} backup (tolerating already-exists)"
    kubectl exec -n "$K8S_NAMESPACE" "$FSX_POD" ${FSX_CT:+-c "$FSX_CT"} -- bash "$PATCH_SCRIPT" backup 2>&1 | tail -2 || true
  else
    echo "    DRY-RUN >> kubectl exec ${FSX_POD} -- bash ${PATCH_SCRIPT} backup"
  fi
}

# patch_stage(): mutate success_stage=<n> on the FSx task cfg via the reused
# patch-success-stage.sh (D-15 reference-not-copy), on the resolved FSx pod. The FRESH
# head re-reads this at env-build time after the per-stage reform. Ensures backup first.
patch_stage() {
  local stage="$1"
  ensure_backup
  resolve_fsx_pod
  echo "    Patch FSx success_stage=${stage} (references patch-success-stage.sh; not copied)"
  run kubectl exec -n "$K8S_NAMESPACE" "$FSX_POD" ${FSX_CT:+-c "$FSX_CT"} -- bash "$PATCH_SCRIPT" set "$stage"
}

# launch_eval_eks(): eks provision. There is NO deploy script on this path — it deploys
# GR00TRLEKSStack directly via `cdk deploy --context compute_backend=eks --context
# mode=eval ...` (per the deploy-eks-training SKILL). vpc_id / s3_data_bucket /
# image_uri come from env; eval_ckpt (or model_path) + eval_total_envs thread the payload
# through app.py -> the head-pod eval env. ARRAY-SAFE (Codex red-team): the env + context
# tokens are passed to run() as an argv ARRAY (`run env "${envs[@]}" cdk ... "${ctx[@]}"`),
# NOT flattened into a bash -c string — so a value containing spaces/;/$() (e.g. a hostile
# VPC_ID) can never be re-parsed as shell. Each --context and its value are SEPARATE tokens.
# Runs from SCRIPT_DIR (the infra dir); dry-run prints the argv and makes ZERO cdk/aws calls.
launch_eval_eks() {
  local stage="$1"
  local envs=(
    "AWS_REGION=${REGION}" "AWS_DEFAULT_REGION=${REGION}" "CDK_DEFAULT_REGION=${REGION}"
    "CDK_DEFAULT_ACCOUNT=${ACCOUNT_ID}"
  )
  local ctx=(
    --context "compute_backend=eks"
    --context "mode=eval"
    --context "vpc_id=${VPC_ID}"
    --context "s3_data_bucket=${S3_DATA_BUCKET}"
    --context "image_uri=${IMAGE_URI}"
    --context "eval_total_envs=${N}"
    # TOPOLOGY PIN (silent-NG-churn fix): pin the rollout/eval-learner instance
    # type + worker count so CDK does NOT revert them to app.py defaults
    # (g6e.4xlarge / 4) and recreate the NGs too small for the head pod.
    --context "rollout_instance_type=${EKS_ROLLOUT_INSTANCE_TYPE}"
    --context "num_rollout_workers=${EKS_NUM_ROLLOUT_WORKERS}"
    # STRUCTURAL PIN: keep the other resource-shaping keys explicit so the eval
    # deploy can't silently revert FSx capacity or downgrade the KubeRay operator.
    --context "fsx_capacity_gib=${EKS_FSX_CAPACITY_GIB}"
    --context "kuberay_version=${EKS_KUBERAY_VERSION}"
  )
  # rollout_subnet_ids / eval_learner_subnet_ids: pass ONLY when set (unset == app.py
  # default == FSx-AZ subnet). Set both to the same other-AZ subnet for a cross-AZ eval.
  if [[ -n "$EKS_ROLLOUT_SUBNET_IDS" ]]; then
    ctx+=( --context "rollout_subnet_ids=${EKS_ROLLOUT_SUBNET_IDS}" )
  fi
  if [[ -n "$EKS_EVAL_LEARNER_SUBNET_IDS" ]]; then
    ctx+=( --context "eval_learner_subnet_ids=${EKS_EVAL_LEARNER_SUBNET_IDS}" )
  fi
  # Only co-provision the CB-backed p5 learner NG when a CR is explicitly supplied
  # (a pure eval leaves it absent — the eval head runs on the g6e EvalLearnerNodes).
  if [[ -n "$CAPACITY_RESERVATION_ID" ]]; then
    ctx+=( --context "capacity_reservation_id=${CAPACITY_RESERVATION_ID}" )
    ctx+=( --context "learner_instance_type=${EKS_LEARNER_INSTANCE_TYPE}" )
  fi
  if [[ -n "$EVAL_CKPT_VAL" ]]; then
    ctx+=( --context "eval_ckpt=${EVAL_CKPT_VAL}" )
  else
    ctx+=( --context "model_path=${MODEL_PATH_VAL}" )
  fi
  echo "    Launch MODE=eval for stage ${stage} via cdk deploy ${EKS_STACK_NAME} (eks backend;"
  echo "    on-demand g6e eval-learner NG desired=1 + rollout NG; head pod routes by is_eval)"
  # Array-safe: cd in a subshell, then run() executes env+cdk as a proper argv array.
  ( cd "$SCRIPT_DIR" && run env "${envs[@]}" cdk deploy "$EKS_STACK_NAME" "${ctx[@]}" --require-approval never )
  # CFN scale-to-zero DRIFT guard (Codex red-team): a prior teardown scaled these
  # NGs to desired=0 OUTSIDE CloudFormation. If the eval template's DesiredSize
  # equals the last-deployed template's (e.g. eval-learner desired=1 both times),
  # CFN sees no property change and LEAVES the NG at the drifted 0 → the eval head
  # pod stays Pending and the eval hangs. So after deploy we EXPLICITLY restore the
  # runtime desired sizes (idempotent; a no-op when CFN already brought them up).
  restore_eval_ng_desired
}

# restore_eval_ng_desired(): force the eval-learner NG to desired=1 and the rollout
# NG to desired=EKS_NUM_ROLLOUT_WORKERS at RUNTIME (independent of CFN drift). Uses
# the pinned NG names when supplied, else discovers by the same substring match as
# teardown. Routed through run()/run_sh() so dry-run prints and executes nothing.
restore_eval_ng_desired() {
  echo "    Restore eval NG desired sizes (drift guard): eval-learner=1, rollout=${EKS_NUM_ROLLOUT_WORKERS}"
  if [[ -n "$EKS_EVAL_LEARNER_NG" && -n "$EKS_ROLLOUT_NG" ]]; then
    run aws eks update-nodegroup-config \
      --cluster-name "$EKS_CLUSTER_NAME" --region "$REGION" \
      --nodegroup-name "$EKS_EVAL_LEARNER_NG" --scaling-config minSize=0,maxSize=1,desiredSize=1
    run aws eks update-nodegroup-config \
      --cluster-name "$EKS_CLUSTER_NAME" --region "$REGION" \
      --nodegroup-name "$EKS_ROLLOUT_NG" --scaling-config "minSize=${EKS_NUM_ROLLOUT_WORKERS},maxSize=${EKS_NUM_ROLLOUT_WORKERS},desiredSize=${EKS_NUM_ROLLOUT_WORKERS}"
  else
    echo "    (NG names not pinned via EKS_EVAL_LEARNER_NG/EKS_ROLLOUT_NG — discover + restore each)"
    run_sh "for ng in \$(aws eks list-nodegroups --cluster-name '${EKS_CLUSTER_NAME}' --region '${REGION}' --query \"nodegroups[?contains(@, 'NodegroupEva')]\" --output text); do aws eks update-nodegroup-config --cluster-name '${EKS_CLUSTER_NAME}' --region '${REGION}' --nodegroup-name \"\$ng\" --scaling-config minSize=0,maxSize=1,desiredSize=1; done"
    run_sh "for ng in \$(aws eks list-nodegroups --cluster-name '${EKS_CLUSTER_NAME}' --region '${REGION}' --query \"nodegroups[?contains(@, 'NodegroupRol')]\" --output text); do aws eks update-nodegroup-config --cluster-name '${EKS_CLUSTER_NAME}' --region '${REGION}' --nodegroup-name \"\$ng\" --scaling-config minSize=${EKS_NUM_ROLLOUT_WORKERS},maxSize=${EKS_NUM_ROLLOUT_WORKERS},desiredSize=${EKS_NUM_ROLLOUT_WORKERS}; done"
  fi
}

# launch_eval(): provision/point the g6e-only eval cluster at the checkpoint for one stage
# via cdk deploy GR00TRLEKSStack (launch_eval_eks). Everything is env-prefixed and run
# array-safe (run env "${envs[@]}" ...) so dry-run prints the full invocation.
launch_eval() {
  local stage="$1"
  launch_eval_eks "$stage"
}

# _reader: the tfevents scalar reader (shared). Reads the LAST eval/success_once value
# under a LOG_DIR; exits non-zero if the tag isn't present yet.
_TB_READER='
import sys, glob, os
from tensorboard.backend.event_processing import event_accumulator
log_dir = os.environ.get("EC_LOG_DIR", "")
files = sorted(glob.glob(os.path.join(log_dir, "**", "*tfevents*"), recursive=True))
val = None
for f in files:
    ea = event_accumulator.EventAccumulator(f, size_guidance={"scalars": 0})
    ea.Reload()
    if "eval/success_once" in ea.Tags().get("scalars", []):
        val = ea.Scalars("eval/success_once")[-1].value
if val is None:
    sys.exit(1)
print(val)
'

# _newest_logdir(): raw newest eval LOG_DIR (execute-time polling; direct kubectl, echoes
# the path or empty). Uses the resolved FSx pod.
_newest_logdir() {
  kubectl exec -n "$K8S_NAMESPACE" "$FSX_POD" ${FSX_CT:+-c "$FSX_CT"} -- \
    sh -c "ls -1dt ${EVAL_RESULTS_PARENT}/*/ 2>/dev/null | head -1" 2>/dev/null | tr -d '\r'
}

# latest_eval_logdir(): newest eval LOG_DIR (dry-run-visible one-shot; resolves the FSx pod).
latest_eval_logdir() {
  resolve_fsx_pod
  run_capture "kubectl exec -n '${K8S_NAMESPACE}' '${FSX_POD}' ${FSX_CT:+-c '${FSX_CT}'} -- sh -c 'ls -1dt ${EVAL_RESULTS_PARENT}/*/ 2>/dev/null | head -1'"
}

# reform_raycluster_eks(): the SELF-CONTAINED per-stage restart (Phase-13 fix; codifies the
# manually-validated 2026-08-20 procedure). RLinf's only_eval head runs once then exits,
# and KubeRay restarts it with a NEW Ray GCS cluster-id that orphans the workers (they can't
# re-register → "GCS authentication error"). So to run a FRESH stage we delete ALL RayCluster
# pods, let KubeRay rebuild, and poll `ray status` until num_nodes are Active — periodically
# recycling any not-Ready workers so they re-register against the STABILIZED head GCS. This
# is what makes a multi-stage sweep work in ONE invocation with no human/agent in the loop.
# Deadline-guarded (REFORM_TIMEOUT and the per-stage MAX_RUNTIME).
# HEAD_START_EPOCH: the fresh head POD's status.startTime (epoch s) after the most recent
# reform reached num_nodes. wait_for_fresh_eval only accepts a LOG_DIR whose mtime is >=
# this — proving the POST-reform fresh head created it (a pre-reform/old-head dir has an
# earlier mtime). Using the pod startTime (a cluster-side timestamp that is ALWAYS before
# the head creates its LOG_DIR) avoids both the poll-timing race and any control-host clock
# skew that a host `date +%s` cutoff would suffer.
HEAD_START_EPOCH=0
reform_raycluster_eks() {
  local stage_start="$1"
  local num_nodes=$(( EKS_NUM_ROLLOUT_WORKERS + 1 ))
  echo "    Reform RayCluster for a FRESH stage run: delete Ray pods (wait gone) -> KubeRay rebuild -> poll ray status == ${num_nodes} nodes"
  # Delete only RayCluster pods (label ray.io/is-ray-node=yes) — NOT a pinned FSX_HELPER_POD.
  # --wait=true so NO old head lingers writing a stale-stage LOG_DIR during its grace period.
  run kubectl delete pods -n "$K8S_NAMESPACE" -l ray.io/is-ray-node=yes --wait=true --timeout=180s --grace-period=10
  if [[ "$EXECUTE" -ne 1 ]]; then
    echo "    DRY-RUN >> (wait for head Running; recycle not-Ready workers every ${REFORM_RECYCLE_EVERY}s; until ${num_nodes} Ray nodes Active or ${REFORM_TIMEOUT}s)"
    return 0
  fi
  local start now elapsed last_recycle head count notready
  start="$(date +%s)"; last_recycle="$start"
  while true; do
    check_deadline "$stage_start"
    now="$(date +%s)"; elapsed=$(( now - start ))
    (( elapsed >= REFORM_TIMEOUT )) && die "RayCluster reform did not reach ${num_nodes} nodes within ${REFORM_TIMEOUT}s."
    head="$(kubectl get pod -n "$K8S_NAMESPACE" -l ray-role=head -o jsonpath='{range .items[?(@.status.phase=="Running")]}{.metadata.name}{"\n"}{end}' 2>/dev/null | head -1)"
    if [[ -n "$head" ]]; then
      # grep -c prints the count (0 on no match) but exits 1 when 0 — `|| true` keeps a clean
      # single integer; sanitize defensively so the arithmetic/compare below can't choke.
      count="$(kubectl exec -n "$K8S_NAMESPACE" "$head" -c "$RAY_HEAD_CONTAINER" -- ray status 2>/dev/null | grep -c 'node_' || true)"
      [[ "$count" =~ ^[0-9]+$ ]] || count=0
      if (( count >= num_nodes )); then
        # Cutoff = the fresh head POD's startTime (cluster-side; always before it writes the
        # LOG_DIR). Convert the RFC3339 K8s timestamp to epoch. Fail closed if unobtainable.
        local head_start
        head_start="$(kubectl get pod -n "$K8S_NAMESPACE" "$head" -o jsonpath='{.status.startTime}' 2>/dev/null)"
        HEAD_START_EPOCH="$(date -d "$head_start" +%s 2>/dev/null || echo 0)"
        [[ "$HEAD_START_EPOCH" =~ ^[0-9]+$ && "$HEAD_START_EPOCH" -gt 0 ]] \
          || die "could not read the fresh head pod startTime ('${head_start}') to set the stage-freshness cutoff — refusing to risk a stale metric."
        ok "RayCluster reformed: ${count}/${num_nodes} Ray nodes Active (head startTime epoch ${HEAD_START_EPOCH})"; return 0
      fi
      # Periodically recycle workers that started Ray but never registered (stale GCS id).
      if (( now - last_recycle >= REFORM_RECYCLE_EVERY )); then
        last_recycle="$now"
        notready="$(kubectl get pods -n "$K8S_NAMESPACE" -l ray-role=worker --no-headers 2>/dev/null | awk '$2=="0/1"{print $1}' | tr '\n' ' ')"
        if [[ -n "$notready" ]]; then
          echo "    reform: ${count}/${num_nodes} joined; recycling not-Ready workers to re-register: ${notready}"
          kubectl delete pod -n "$K8S_NAMESPACE" $notready --wait=false --grace-period=10 >/dev/null 2>&1 || true
        else
          echo "    reform: ${count}/${num_nodes} joined; waiting (all workers Ready, GCS still registering)"
        fi
      fi
    fi
    sleep 20
  done
}

# wait_for_fresh_eval(): block until a FRESH eval LOG_DIR (newer than pre_logdir) contains
# eval/success_once in its FSx tfevents — the restart-resilient source of truth (head logs
# are lost across the only_eval auto-restart). Echoes the fresh LOG_DIR on stdout. Deadline-guarded.
wait_for_fresh_eval() {
  local stage="$1" stage_start="$2" pre_logdir="$3"
  echo "    Wait for stage ${stage} FRESH eval/success_once (new LOG_DIR, mtime >= head start ${HEAD_START_EPOCH}, deadline/stage ${MAX_RUNTIME}s)" >&2
  if [[ "$EXECUTE" -ne 1 ]]; then
    echo "    DRY-RUN >> (poll FSx for a post-reform LOG_DIR with eval/success_once)" >&2
    echo "${EVAL_RESULTS_PARENT}/<fresh>"; return 0
  fi
  local cur mtime
  while true; do
    check_deadline "$stage_start"
    # Re-resolve the FSx pod every iteration: the reform deleted the worker patch_stage
    # used, so the prior FSX_POD handle is stale — resolve_fsx_pod polls for a live worker.
    resolve_fsx_pod
    cur="$(_newest_logdir || true)"
    if [[ -n "$cur" && "$cur" != "$pre_logdir" ]]; then
      # STAGE-IDENTITY guard (FAIL CLOSED): the LOG_DIR must have been created at/after the
      # fresh head pod started (HEAD_START_EPOCH) — proving the post-reform head wrote it,
      # not a pre-reform/old-head dir. A non-numeric mtime (stat unavailable) does NOT pass;
      # we keep waiting rather than risk a stale metric (MAX_RUNTIME bounds the wait).
      mtime="$(kubectl exec -n "$K8S_NAMESPACE" "$FSX_POD" ${FSX_CT:+-c "$FSX_CT"} -- stat -c %Y "$cur" 2>/dev/null | tr -d '\r' || true)"
      if [[ "$mtime" =~ ^[0-9]+$ ]] && (( mtime >= HEAD_START_EPOCH )); then
        if kubectl exec -n "$K8S_NAMESPACE" "$FSX_POD" ${FSX_CT:+-c "$FSX_CT"} -- \
             env EC_LOG_DIR="$cur" python3 -c "$_TB_READER" >/dev/null 2>&1; then
          ok "stage ${stage} eval complete — fresh LOG_DIR ${cur} (mtime ${mtime} >= head start ${HEAD_START_EPOCH})" >&2
          echo "$cur"; return 0
        fi
      fi
    fi
    sleep 30
  done
}

# read_success_once(): read the final eval/success_once from a LOG_DIR's FSx tfevents.
# Echoes the success fraction (0..1). Array-safe (python source is ONE argv element).
read_success_once() {
  local log_dir="$1"
  resolve_fsx_pod
  run_capture_argv kubectl exec -n "$K8S_NAMESPACE" "$FSX_POD" ${FSX_CT:+-c "$FSX_CT"} -- \
    env EC_LOG_DIR="$log_dir" python3 -c "$_TB_READER"
}

# =============================================================================
#  STEP 2 — Per-stage sweep -> collect k/N
# =============================================================================
say "STEP 2 — Per-stage success_stage sweep (MODE=eval, N=${N})"
# SELF-CONTAINED: the script drives the whole stage machine in ONE invocation with
# NO human/agent in the loop — deploy the stack + form the cluster ONCE, then per stage:
# patch success_stage on FSx -> REFORM the RayCluster (fresh head re-reads the stage) ->
# wait for a FRESH eval/success_once -> read it.
say "Provision the eks eval stack + form the initial RayCluster (once)"
launch_eval "${STAGES%% *}"     # initial cdk deploy + restore_eval_ng_desired

KVALS=()          # collected k (successes) per swept stage, index-aligned with STAGE_ARR
STAGE_ARR=()      # the stages actually swept, in order
for stage in $STAGES; do
  case "$stage" in 1|2|3|4) ;; *) die "stage must be 1..4 (got '$stage')." ;; esac
  STAGE_ARR+=( "$stage" )
  echo ""
  echo "    ---- STAGE ${stage} ----"
  STAGE_START="$(date +%s)"
  patch_stage "$stage"
  # FRESHNESS baseline: the newest eval LOG_DIR BEFORE this stage's run. A genuinely fresh
  # stage run (head re-reads success_stage after the reform) yields a NEW LOG_DIR != this.
  PRE_LOGDIR="$(latest_eval_logdir || true)"
  reform_raycluster_eks "$STAGE_START"    # fresh head reads the just-patched success_stage
  # Wait for a FRESH LOG_DIR (!= baseline) with eval/success_once — fails closed on overrun.
  LOG_DIR="$(wait_for_fresh_eval "$stage" "$STAGE_START" "$PRE_LOGDIR")"
  if [[ "$EXECUTE" -eq 1 ]]; then
    [[ -n "$LOG_DIR" && "$LOG_DIR" != "$PRE_LOGDIR" ]] || die "stage ${stage}: no FRESH eval LOG_DIR appeared — the head did not re-run for this stage; refusing to report a stale metric."
  fi
  P="$(read_success_once "${LOG_DIR:-${EVAL_RESULTS_PARENT}/<latest>}" || true)"
  if [[ "$EXECUTE" -eq 1 ]]; then
    # Require a real numeric metric — fail closed rather than record "?" on a paid run.
    [[ "$P" =~ ^[0-9]*\.?[0-9]+$ ]] || die "stage ${stage}: eval/success_once is not a number (got '${P:-<empty>}') — refusing to report an incomplete eval."
    # k = round(p * N). Use awk for float->int rounding.
    K="$(awk "BEGIN{printf \"%d\", ($P * $N) + 0.5}")"
  else
    K="?"   # dry-run: no real metric; the CI/verdict step prints the plan, not numbers.
  fi
  echo "    stage ${stage}: eval/success_once=${P:-<pending>}  =>  k=${K}/${N}"
  KVALS+=( "$K" )
done

# =============================================================================
#  STEP 3 — Wilson 95% CI per stage + PASS/CONTINUE verdict
# =============================================================================
# Wilson 95% CI recomputation — paste-ready, matches 07.1.1-STEP-A-RESULTS.md at N.
# Verdict rule: for each swept stage, PASS_i holds when the reference p_i lies WITHIN our
# Wilson band OR BELOW its lower bound (i.e. we meet-or-exceed). The only failing case is
# ref_i ABOVE our upper bound (hi) — we are short. Verdict = PASS iff every swept stage
# passes; otherwise CONTINUE, naming the short stages.
say "STEP 3 — Wilson 95% CI + PASS/CONTINUE verdict vs ref row ${REF_ROW}"
compute_ci_verdict() {
  EC_STAGES="${STAGE_ARR[*]}" EC_KVALS="${KVALS[*]}" EC_N="$N" EC_REF="$REF_ROW" \
  python3 - <<'PY'
import os
from math import sqrt

def wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    hw = z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (center - hw) * 100.0, (center + hw) * 100.0

stages = os.environ["EC_STAGES"].split()
kvals  = os.environ["EC_KVALS"].split()
n      = int(os.environ["EC_N"])
ref    = [float(x) for x in os.environ["EC_REF"].split(",")]

hdr = f"{'Stage':>5} | {'k/N':>7} | {'ours %':>7} | {'ref %':>7} | {'Wilson 95% CI':>18} | meet?"
print("    " + hdr)
print("    " + "-" * len(hdr))
short = []
for i, (st, kv) in enumerate(zip(stages, kvals)):
    ref_i = ref[int(st) - 1] if int(st) - 1 < len(ref) else (ref[i] if i < len(ref) else float('nan'))
    if kv == "?":
        print(f"    {st:>5} | {'?/'+str(n):>7} | {'  pending':>7} | {ref_i:>6.2f}% | {'(dry-run: pending)':>18} | ?")
        continue
    k = int(kv)
    lo, hi = wilson(k, n)
    ours = 100.0 * k / n
    meets = ref_i <= hi + 1e-9   # PASS_i: ref within band or below lower bound (we meet-or-exceed)
    if not meets:
        short.append(st)
    band = f"[{lo:.1f}%, {hi:.1f}%]"
    print(f"    {st:>5} | {str(k)+'/'+str(n):>7} | {ours:>6.1f}% | {ref_i:>6.2f}% | {band:>18} | {'yes' if meets else 'NO'}")

any_pending = any(kv == "?" for kv in kvals)
print()
if any_pending:
    print("    VERDICT: (dry-run — no real eval numbers; PASS/CONTINUE computed on --execute)")
elif short:
    print("    VERDICT: CONTINUE — short at stage(s): " + ", ".join(short))
else:
    print("    VERDICT: PASS — meets-or-exceeds the reference row at every swept stage")
PY
}

# Print the table + verdict to stdout, and (in --execute) also persist it under the eval LOG_DIR.
if [[ "$EXECUTE" -eq 1 ]]; then
  RESULTS_FILE="${LOG_DIR:-${EVAL_RESULTS_PARENT}}/eval-checkpoint-results.txt"
  compute_ci_verdict | tee /tmp/eval-checkpoint-verdict.txt
  # Persist the verdict to FSx — BEST EFFORT (the verdict is already on stdout + /tmp, and
  # the cluster is torn down right after). Resolve a real target pod (pinned helper, or a
  # one-shot Running worker — NOT the polling resolve, so this can't hang), and never fail
  # the run if the copy doesn't land. Array-safe: cp target is ONE argv token.
  _pp="${FSX_HELPER_POD:-$(kubectl get pods -n "$K8S_NAMESPACE" -l ray-role=worker -o jsonpath='{range .items[?(@.status.phase=="Running")]}{.metadata.name}{"\n"}{end}' 2>/dev/null | head -1)}"
  _pc=""; [[ -z "$FSX_HELPER_POD" ]] && _pc="$RAY_WORKER_CONTAINER"
  if [[ -n "$_pp" ]]; then
    echo "    Persisting verdict to FSx: ${RESULTS_FILE} (pod ${_pp})"
    if run kubectl cp /tmp/eval-checkpoint-verdict.txt "${K8S_NAMESPACE}/${_pp}:${RESULTS_FILE}" ${_pc:+-c "$_pc"}; then
      ok "verdict persisted to FSx."
    else
      warn "could not persist verdict to FSx — it is printed above and in /tmp/eval-checkpoint-verdict.txt"
    fi
  else
    warn "no Running pod to persist the verdict to FSx — it is printed above and in /tmp/eval-checkpoint-verdict.txt"
  fi
else
  compute_ci_verdict
fi

# =============================================================================
#  Done — the trap tears the eval cluster down on the way out (success path here).
# =============================================================================
say "Runbook complete."
if [[ "$EXECUTE" -eq 1 ]]; then
  ok "EXECUTE finished — see the per-stage table + verdict above (also on FSx)."
else
  echo "    DRY-RUN finished — nothing ran, \$0 spent. Re-run with --execute for the paid eval."
  echo "    (The teardown below is the trap firing on the dry-run's normal exit — printed, not run.)"
fi
