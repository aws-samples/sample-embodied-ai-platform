#!/usr/bin/env bash
#
# prepare-artifacts.sh — one-command, GATED "build image + stage data" wrapper.
#
# This is the tried-and-true, repeatable pre-deploy path. It REPLACES the removed
# fire-and-forget auto-trigger: instead of a build that races the backend deploy,
# this SYNCHRONOUSLY kicks the single CodeBuild pipeline (GR00T-RL-Pipeline),
# WAITS for it, VERIFIES the artifacts actually landed (ECR image digest and/or the
# s3://.../_STAGING_COMPLETE marker), and only THEN hands you (or runs) the exact
# `cdk deploy ... --context image_uri=<resolved>` line. It fails CLOSED at every
# step, so a backend deploy can never proceed on a half-built image or unstaged data.
#
# ─────────────────────────────────────────────────────────────────────────────
#  PREREQUISITE
# ─────────────────────────────────────────────────────────────────────────────
#   The standalone GR00TRLArtifactsStack MUST be deployed BEFORE running this — it
#   owns the GR00T-RL-Pipeline CodeBuild project + the gr00t-rl-unified ECR repo.
#   This script only TRIGGERS + VERIFIES that project; it does not create it.
#     cdk deploy GR00TRLArtifactsStack --context compute_backend=eks \
#         --context s3_data_bucket=<bucket>
#
# ─────────────────────────────────────────────────────────────────────────────
#  INTERFACE (the CodeBuild pipeline this drives)
# ─────────────────────────────────────────────────────────────────────────────
#   * ONE CodeBuild project (default: GR00T-RL-Pipeline) owned by GR00TRLArtifactsStack,
#     driven by the env var STAGE_MODE ∈ {build-image, stage-data, all}.
#       - build-image : docker build docker/Dockerfile.unified -> push ECR gr00t-rl-unified
#       - stage-data  : stage data; writes marker s3://$BUCKET/_STAGING_COMPLETE on success
#       - all         : both of the above
#   * --stack (default GR00TRLArtifactsStack) is the OUTPUT-resolution stack
#     (PipelineProject / DataBucketName / UnifiedECRUri live there).
#   * --deploy-stack (default GR00TRLEKSStack) is the BACKEND stack that --deploy
#     actually deploys with the resolved --context image_uri=<digest>.
#
# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC-MIRROR RULE
# ─────────────────────────────────────────────────────────────────────────────
#   NO internal IDs are baked in. Account is resolved at runtime (sts), region is
#   REQUIRED (--region or AWS_REGION; no baked default), bucket is resolved from the
#   stack output or passed explicitly. Nothing internal is hardcoded.
#
# Usage:
#   ./prepare-artifacts.sh --region <r> [--mode image|data|all] [--project <name>]
#                          [--stack <name>] [--deploy-stack <name>] [--bucket <s3-bucket>]
#                          [--image-uri <uri>] [--deploy]
#                          [-- <extra args forwarded verbatim to cdk deploy>]
#
#   # Build image + stage data, verify, then print the deploy line (no deploy):
#   ./prepare-artifacts.sh --region <r> --mode all
#
#   # Same, but also run the backend deploy once artifacts are verified (the
#   # backend stack needs its own context — pass it verbatim after --):
#   ./prepare-artifacts.sh --region <r> --mode all --deploy -- \
#       --context vpc_id=<vpc> --context s3_data_bucket=<bucket>
#
#   # Data-only re-stage + deploy against an already-built image (required in
#   # data mode: either --image-uri, or a resolvable :latest digest in ECR):
#   ./prepare-artifacts.sh --region <r> --mode data --deploy \
#       --image-uri <acct>.dkr.ecr.<r>.amazonaws.com/gr00t-rl-unified@sha256:... -- \
#       --context vpc_id=<vpc> --context s3_data_bucket=<bucket>
#
#   NOTE: image_uri and compute_backend are set by this wrapper — do NOT pass them
#   as --context passthrough after --; the script rejects them.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# =============================================================================
#  Configuration (no internal IDs; region required, account resolved at runtime)
# =============================================================================
REGION="${AWS_REGION:-}"                 # REQUIRED — pass --region or export AWS_REGION
MODE="all"                               # image | data | all
PROJECT="${PIPELINE_PROJECT:-GR00T-RL-Pipeline}"
# OUTPUT-resolution stack: owns the pipeline + ECR + bucket outputs.
STACK="${ARTIFACTS_STACK_NAME:-GR00TRLArtifactsStack}"
# DEPLOY target: the BACKEND stack that --deploy actually deploys.
DEPLOY_STACK="${EKS_STACK_NAME:-GR00TRLEKSStack}"
BUCKET="${S3_DATA_BUCKET:-}"             # resolved from stack DataBucketName if empty
ECR_REPO="${ECR_REPO:-gr00t-rl-unified}" # the unified image repo (interface-fixed)
IMAGE_TAG="${IMAGE_TAG:-latest}"         # the moving tag the buildspec pushes (+ build-$N)
IMAGE_URI=""                             # data-mode: explicit bring-your-own deploy image
POLL_INTERVAL="${POLL_INTERVAL:-20}"     # seconds between build-status polls
DEPLOY=0
PASSTHRU=()                              # forwarded verbatim to `cdk deploy`

# =============================================================================
#  Echo helpers (house style, mirrors eval-checkpoint.sh)
# =============================================================================
say()  { echo ""; echo "==> $*"; }
ok()   { echo "    [OK]   $*"; }
warn() { echo "    [WARN] $*"; }
die()  { echo ""; echo "    [FATAL] $*" >&2; exit 1; }

usage() { sed -n '2,63p' "$0"; }

# =============================================================================
#  Arg parse (everything after a bare `--` is forwarded to cdk deploy)
# =============================================================================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)         MODE="$2"; shift 2 ;;
    --region)       REGION="$2"; shift 2 ;;
    --project)      PROJECT="$2"; shift 2 ;;
    --stack)        STACK="$2"; shift 2 ;;
    --deploy-stack) DEPLOY_STACK="$2"; shift 2 ;;
    --bucket)       BUCKET="$2"; shift 2 ;;
    --image-uri)    IMAGE_URI="$2"; shift 2 ;;
    --deploy)       DEPLOY=1; shift ;;
    --)             shift; PASSTHRU=("$@"); break ;;
    -h|--help)      usage; exit 0 ;;
    *) die "unknown arg: $1 (try --help)" ;;
  esac
done

# Reject reserved passthrough context keys: this wrapper OWNS image_uri +
# compute_backend on the deploy line. Letting a caller also set them via
# passthrough would silently override the verified digest / backend selection.
for _arg in ${PASSTHRU[@]+"${PASSTHRU[@]}"}; do
  case "$_arg" in
    image_uri=*|compute_backend=*|--context=image_uri=*|--context=compute_backend=*)
      die "reserved passthrough key '$_arg' — this wrapper sets image_uri + compute_backend itself; remove it." ;;
  esac
done

# =============================================================================
#  Input validation (fail closed)
# =============================================================================
[[ -n "$REGION" ]] || die "no region — pass --region <r> or export AWS_REGION (no baked default)."
[[ -n "$PROJECT" ]] || die "--project must be non-empty."
case "$MODE" in
  image) STAGE_MODE="build-image" ;;
  data)  STAGE_MODE="stage-data" ;;
  all)   STAGE_MODE="all" ;;
  *)     die "--mode must be one of image|data|all (got '$MODE')." ;;
esac
command -v aws  >/dev/null 2>&1 || die "aws CLI not found on PATH."
command -v jq   >/dev/null 2>&1 || die "jq not found on PATH."

# Which verifications this mode requires.
NEED_IMAGE=0; NEED_DATA=0
case "$STAGE_MODE" in
  build-image) NEED_IMAGE=1 ;;
  stage-data)  NEED_DATA=1 ;;
  all)         NEED_IMAGE=1; NEED_DATA=1 ;;
esac

say "prepare-artifacts: mode=$MODE (STAGE_MODE=$STAGE_MODE) region=$REGION project=$PROJECT stack=$STACK"

# =============================================================================
#  Resolve account + bucket (bucket from stack DataBucketName if not supplied)
# =============================================================================
ACCOUNT_ID=$(aws sts get-caller-identity --region "$REGION" --query Account --output text) \
  || die "could not resolve account id (aws sts get-caller-identity failed)."
[[ -n "$ACCOUNT_ID" && "$ACCOUNT_ID" != "None" ]] || die "empty account id from sts."
ok "account: $ACCOUNT_ID"

if [[ "$NEED_DATA" -eq 1 && -z "$BUCKET" ]]; then
  say "resolving data bucket from $STACK output DataBucketName"
  BUCKET=$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='DataBucketName'].OutputValue | [0]" \
      --output text 2>/dev/null || true)
  [[ -n "$BUCKET" && "$BUCKET" != "None" ]] \
    || die "could not resolve DataBucketName from $STACK — pass --bucket <s3-bucket> explicitly."
  ok "data bucket: $BUCKET"
fi

# =============================================================================
#  1. Kick the build (STAGE_MODE override) and capture the build id
# =============================================================================
say "starting CodeBuild: project=$PROJECT STAGE_MODE=$STAGE_MODE"
BUILD_ID=$(aws codebuild start-build \
    --project-name "$PROJECT" \
    --region "$REGION" \
    --environment-variables-override "name=STAGE_MODE,value=${STAGE_MODE},type=PLAINTEXT" \
    --query 'build.id' --output text) \
  || die "aws codebuild start-build failed for project '$PROJECT'."
[[ -n "$BUILD_ID" && "$BUILD_ID" != "None" ]] || die "start-build returned no build id."
ok "build id: $BUILD_ID"

# =============================================================================
#  2. Poll until the build leaves IN_PROGRESS; require SUCCEEDED
# =============================================================================
warn "build-image / all can take 30-60 min — polling every ${POLL_INTERVAL}s (Ctrl-C is safe; the build keeps running)."
START_TS=$(date -u +%s)
STATUS="IN_PROGRESS"
while :; do
  STATUS=$(aws codebuild batch-get-builds --ids "$BUILD_ID" --region "$REGION" \
      --query 'builds[0].buildStatus' --output text 2>/dev/null || echo "QUERY_FAILED")
  ELAPSED=$(( $(date -u +%s) - START_TS ))
  if [[ "$STATUS" != "IN_PROGRESS" ]]; then
    break
  fi
  printf '    ... IN_PROGRESS  (elapsed %dm%02ds)\n' $((ELAPSED/60)) $((ELAPSED%60))
  sleep "$POLL_INTERVAL"
done
ELAPSED=$(( $(date -u +%s) - START_TS ))
printf '    build finished: status=%s  (elapsed %dm%02ds)\n' "$STATUS" $((ELAPSED/60)) $((ELAPSED%60))

# Pull the phase + CloudWatch log pointers regardless of outcome (useful on failure).
BUILD_JSON=$(aws codebuild batch-get-builds --ids "$BUILD_ID" --region "$REGION" \
    --query 'builds[0]' --output json 2>/dev/null || echo '{}')
CUR_PHASE=$(echo "$BUILD_JSON" | jq -r '.currentPhase // "unknown"')
FAIL_PHASE=$(echo "$BUILD_JSON" | jq -r '[.phases[]? | select(.phaseStatus=="FAILED") | .phaseType] | join(",") // ""')
LOG_GROUP=$(echo "$BUILD_JSON" | jq -r '.logs.groupName // ""')
LOG_STREAM=$(echo "$BUILD_JSON" | jq -r '.logs.streamName // ""')
LOG_URL=$(echo "$BUILD_JSON" | jq -r '.logs.deepLink // ""')

if [[ "$STATUS" != "SUCCEEDED" ]]; then
  warn "phase: current='${CUR_PHASE}' failed='${FAIL_PHASE:-none}'"
  [[ -n "$LOG_GROUP"  ]] && warn "CloudWatch log group : $LOG_GROUP"
  [[ -n "$LOG_STREAM" ]] && warn "CloudWatch log stream: $LOG_STREAM"
  [[ -n "$LOG_URL"    ]] && warn "console logs         : $LOG_URL"
  die "CodeBuild did not SUCCEED (status=$STATUS) — inspect the log group/stream above and re-run."
fi
ok "CodeBuild SUCCEEDED (id=$BUILD_ID)"

# =============================================================================
#  3. Verify artifacts landed (fail closed if absent)
# =============================================================================
IMAGE_DIGEST=""
if [[ "$NEED_IMAGE" -eq 1 ]]; then
  # Require the EXACT tag this run produced. The buildspec pushes IMAGE_TAG
  # (default latest) AND build-$CODEBUILD_BUILD_NUMBER; we verify + resolve the
  # IMAGE_TAG digest (no "most-recent" fallback — that could pick up an unrelated
  # image from a different build and deploy a stale digest).
  say "verifying ECR image '$ECR_REPO:$IMAGE_TAG' (the exact tag this build pushed)"
  IMAGE_DIGEST=$(aws ecr describe-images --repository-name "$ECR_REPO" --region "$REGION" \
      --image-ids imageTag="$IMAGE_TAG" \
      --query 'imageDetails[0].imageDigest' --output text 2>/dev/null || true)
  [[ -n "$IMAGE_DIGEST" && "$IMAGE_DIGEST" != "None" ]] \
    || die "no '$ECR_REPO:$IMAGE_TAG' image in ECR — the build did not push the expected tag. Check the build logs."
  ok "pushed image digest ($IMAGE_TAG): $IMAGE_DIGEST"
fi

if [[ "$NEED_DATA" -eq 1 ]]; then
  say "verifying staging marker s3://$BUCKET/_STAGING_COMPLETE"
  aws s3 ls "s3://${BUCKET}/_STAGING_COMPLETE" --region "$REGION" >/dev/null 2>&1 \
    || die "staging marker s3://${BUCKET}/_STAGING_COMPLETE not found — data staging did not complete. Check the build logs."
  ok "staging marker present: s3://${BUCKET}/_STAGING_COMPLETE"
fi

# =============================================================================
#  4. Resolve the image_uri to hand to cdk deploy (prefer digest; else :latest)
# =============================================================================
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
RESOLVED_IMAGE=""
if [[ -n "$IMAGE_DIGEST" && "$IMAGE_DIGEST" != "None" ]]; then
  # image/all mode: use the digest we just built + verified.
  RESOLVED_IMAGE="${ECR_REGISTRY}/${ECR_REPO}@${IMAGE_DIGEST}"
else
  # data-only mode: no image was built this run. Do NOT invent a bare :latest tag
  # (that would deploy whatever currently sits at :latest, unverified). Instead:
  #   1. an explicit --image-uri wins, else
  #   2. resolve + verify the existing ':IMAGE_TAG' digest already in ECR.
  # If neither is available we leave RESOLVED_IMAGE empty and refuse to deploy.
  if [[ -n "$IMAGE_URI" ]]; then
    RESOLVED_IMAGE="$IMAGE_URI"
    ok "data mode: using explicit --image-uri: $RESOLVED_IMAGE"
  else
    say "data mode: resolving existing '$ECR_REPO:$IMAGE_TAG' digest from ECR"
    _existing_digest=$(aws ecr describe-images --repository-name "$ECR_REPO" --region "$REGION" \
        --image-ids imageTag="$IMAGE_TAG" \
        --query 'imageDetails[0].imageDigest' --output text 2>/dev/null || true)
    if [[ -n "$_existing_digest" && "$_existing_digest" != "None" ]]; then
      RESOLVED_IMAGE="${ECR_REGISTRY}/${ECR_REPO}@${_existing_digest}"
      ok "data mode: resolved existing '$IMAGE_TAG' digest: $RESOLVED_IMAGE"
    else
      warn "data mode: no '$ECR_REPO:$IMAGE_TAG' image in ECR and no --image-uri given."
    fi
  fi
fi

# =============================================================================
#  5. Print the copy-pasteable deploy line; optionally run it
# =============================================================================
if [[ -z "$RESOLVED_IMAGE" ]]; then
  # No verified image to deploy with. Fail closed if the caller intended a deploy;
  # otherwise just stop after the staging verification.
  if [[ "$DEPLOY" -eq 1 ]]; then
    die "data mode: no deploy image — pass --image-uri <uri>, or build the image first so '$ECR_REPO:$IMAGE_TAG' exists."
  fi
  say "data staged + verified. No deploy image resolved (data mode) — nothing to deploy."
  say "provide --image-uri (or build the image) to produce a backend deploy line."
  exit 0
fi

DEPLOY_CMD=(cdk deploy "$DEPLOY_STACK"
            --context compute_backend=eks
            --context "image_uri=${RESOLVED_IMAGE}")
if [[ ${#PASSTHRU[@]} -gt 0 ]]; then
  DEPLOY_CMD+=("${PASSTHRU[@]}")
fi
DEPLOY_CMD+=(--require-approval never)

say "artifacts verified. Resolved image_uri:"
echo "    ${RESOLVED_IMAGE}"
say "next: deploy the EKS backend ($DEPLOY_STACK) with the verified image:"
echo "    ${DEPLOY_CMD[*]}"

if [[ "$DEPLOY" -eq 1 ]]; then
  command -v cdk >/dev/null 2>&1 || die "--deploy given but 'cdk' not found on PATH."
  say "running deploy (--deploy)"
  echo "    RUN >> ${DEPLOY_CMD[*]}"
  # Pin the region for the app + CLI so app.py's region resolution can't drift
  # (and never disagrees — see the region-conflict guard in app.py).
  env AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION" \
      CDK_DEFAULT_REGION="$REGION" CDK_DEFAULT_ACCOUNT="$ACCOUNT_ID" \
      "${DEPLOY_CMD[@]}"
  ok "cdk deploy completed for $DEPLOY_STACK."
else
  say "no --deploy: stopping after verify. Run the line above when ready."
fi
