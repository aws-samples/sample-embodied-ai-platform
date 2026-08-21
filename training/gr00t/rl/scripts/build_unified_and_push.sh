#!/usr/bin/env bash
#
# build_unified_and_push.sh — EKS-correct MANUAL self-build of the unified image.
#
# Builds docker/Dockerfile.unified and pushes it to the ECR repo the EKS stack uses
# (gr00t-rl-unified), then prints the exact `--context image_uri=<repo>@sha256:<digest>`
# line to hand to `cdk deploy` (or to prepare-artifacts.sh's bring-your-own path).
#
# This REPLACES an obsolete two-image self-build script, which queried a non-existent
# GR00TRLBatchStack (outputs LearnerECR/RolloutECR) and built two now-defunct images
# (learner + rollout). The heterogeneous EKS backend uses ONE unified image.
#
# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC-MIRROR RULE
# ─────────────────────────────────────────────────────────────────────────────
#   NO internal IDs are baked in. Region is REQUIRED (--region or AWS_REGION; no
#   baked default). The repository URI is either passed (--repository-uri) or
#   resolved from the GR00TRLArtifactsStack output UnifiedECRUri at runtime; with
#   --create-repository it is derived from the caller's account+region if that
#   output does not exist yet (bootstrap before the artifacts stack is deployed).
#
# ─────────────────────────────────────────────────────────────────────────────
#  WORKING DIRECTORY
# ─────────────────────────────────────────────────────────────────────────────
#   Run this from the repo's training/gr00t/rl directory, so ./docker is present:
#   the Dockerfile's COPY paths assume the build context is the docker/ dir.
#       cd training/gr00t/rl
#       ./scripts/build_unified_and_push.sh --region <r>
#
# Usage:
#   ./scripts/build_unified_and_push.sh --region <r>
#       [--repository-uri <acct>.dkr.ecr.<r>.amazonaws.com/gr00t-rl-unified]
#       [--tag <t>]              # default: git short SHA, else UTC timestamp
#       [--create-repository]    # create the ECR repo (scan-on-push) if missing
#       [--stack <name>]         # default GR00TRLArtifactsStack (for UnifiedECRUri lookup)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# =============================================================================
#  Configuration (no internal IDs; region required)
# =============================================================================
REGION="${AWS_REGION:-}"                       # REQUIRED — --region or AWS_REGION
REPO_URI="${REPOSITORY_URI:-}"                 # resolved from stack UnifiedECRUri if empty
STACK="${ARTIFACTS_STACK_NAME:-GR00TRLArtifactsStack}"
TAG=""                                         # default computed below (git SHA / timestamp)
CREATE_REPO=0
ECR_REPO_DEFAULT="gr00t-rl-unified"            # interface-fixed repo name
DOCKERFILE="docker/Dockerfile.unified"         # relative to CWD (training/gr00t/rl)
DOCKER_CONTEXT="docker"

# =============================================================================
#  Echo helpers (house style, mirrors eval-checkpoint.sh)
# =============================================================================
say()  { echo ""; echo "==> $*"; }
ok()   { echo "    [OK]   $*"; }
warn() { echo "    [WARN] $*"; }
die()  { echo ""; echo "    [FATAL] $*" >&2; exit 1; }

usage() { sed -n '2,36p' "$0"; }

# =============================================================================
#  Arg parse
# =============================================================================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)            REGION="$2"; shift 2 ;;
    --repository-uri)    REPO_URI="$2"; shift 2 ;;
    --tag)               TAG="$2"; shift 2 ;;
    --create-repository) CREATE_REPO=1; shift ;;
    --stack)             STACK="$2"; shift 2 ;;
    -h|--help)           usage; exit 0 ;;
    *) die "unknown arg: $1 (try --help)" ;;
  esac
done

# =============================================================================
#  Input validation (fail closed)
# =============================================================================
[[ -n "$REGION" ]] || die "no region — pass --region <r> or export AWS_REGION (no baked default)."
command -v aws    >/dev/null 2>&1 || die "aws CLI not found on PATH."
command -v docker >/dev/null 2>&1 || die "docker not found on PATH."

# Must run from training/gr00t/rl so ./docker (the build context) is present.
[[ -f "$DOCKERFILE" ]] \
  || die "$DOCKERFILE not found — run this from the repo's training/gr00t/rl directory (so ./docker exists)."

# Default tag: git short SHA if in a repo, else a UTC timestamp.
if [[ -z "$TAG" ]]; then
  if TAG=$(git rev-parse --short HEAD 2>/dev/null); then
    # A dirty worktree means the source-hash tag no longer uniquely pins the tree
    # that was built — mark it so the built image is not mistaken for the clean SHA.
    if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
      TAG="${TAG}-dirty"
    fi
  else
    TAG="$(date -u +%Y%m%d%H%M%S)"
  fi
fi
ok "image tag: $TAG"

# =============================================================================
#  Resolve the repository URI (arg wins; else stack UnifiedECRUri output)
# =============================================================================
if [[ -z "$REPO_URI" ]]; then
  say "resolving repository URI from $STACK output UnifiedECRUri"
  REPO_URI=$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
      --query "Stacks[0].Outputs[?OutputKey=='UnifiedECRUri'].OutputValue | [0]" \
      --output text 2>/dev/null || true)
  if [[ -z "$REPO_URI" || "$REPO_URI" == "None" ]]; then
    if [[ "$CREATE_REPO" -eq 1 ]]; then
      # Bootstrap path: the artifacts stack isn't deployed yet (no UnifiedECRUri
      # output). Since --create-repository was requested, DERIVE the canonical repo
      # URI from the caller's account + region and let the create step below make it,
      # instead of dying before we ever get a chance to create the repo.
      DERIVED_ACCOUNT=$(aws sts get-caller-identity --region "$REGION" --query Account --output text 2>/dev/null || true)
      [[ -n "$DERIVED_ACCOUNT" && "$DERIVED_ACCOUNT" != "None" ]] \
        || die "could not resolve account id (aws sts get-caller-identity failed) to derive the repository URI."
      REPO_URI="${DERIVED_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_DEFAULT}"
      warn "UnifiedECRUri not found on $STACK; --create-repository set — deriving and creating $REPO_URI"
    else
      die "could not resolve UnifiedECRUri from $STACK — pass --repository-uri explicitly, or re-run with --create-repository to derive + create '${ECR_REPO_DEFAULT}'."
    fi
  fi
fi
ok "repository URI: $REPO_URI"

# Split registry host from repo name (registry = up to first '/'; repo = the rest).
ECR_REGISTRY="${REPO_URI%%/*}"
ECR_REPO_NAME="${REPO_URI#*/}"
[[ "$ECR_REGISTRY" != "$REPO_URI" && -n "$ECR_REPO_NAME" ]] \
  || die "malformed --repository-uri '$REPO_URI' (expected <registry>/<repo>)."

# =============================================================================
#  Optionally create the ECR repository (scan-on-push) if it does not exist
# =============================================================================
if [[ "$CREATE_REPO" -eq 1 ]]; then
  if aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$REGION" >/dev/null 2>&1; then
    ok "ECR repo '$ECR_REPO_NAME' already exists."
  else
    say "creating ECR repo '$ECR_REPO_NAME' (scan-on-push)"
    aws ecr create-repository --repository-name "$ECR_REPO_NAME" --region "$REGION" \
        --image-scanning-configuration scanOnPush=true >/dev/null \
      || die "failed to create ECR repo '$ECR_REPO_NAME'."
    ok "created ECR repo '$ECR_REPO_NAME'."
  fi
fi

# =============================================================================
#  ECR login
# =============================================================================
say "logging in to ECR registry $ECR_REGISTRY"
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$ECR_REGISTRY" \
  || die "ECR docker login failed."
ok "logged in to $ECR_REGISTRY"

# =============================================================================
#  Build (context = ./docker) and tag :<tag> + :latest
# =============================================================================
say "building $DOCKERFILE -> ${REPO_URI}:${TAG} (context: ${DOCKER_CONTEXT}/)"
docker build -f "$DOCKERFILE" -t "${REPO_URI}:${TAG}" -t "${REPO_URI}:latest" "$DOCKER_CONTEXT" \
  || die "docker build failed."
ok "built ${REPO_URI}:${TAG} (also tagged :latest)"

# =============================================================================
#  Push both tags
# =============================================================================
say "pushing ${REPO_URI}:${TAG}"
docker push "${REPO_URI}:${TAG}" || die "docker push ${REPO_URI}:${TAG} failed."
say "pushing ${REPO_URI}:latest"
docker push "${REPO_URI}:latest" || die "docker push ${REPO_URI}:latest failed."
ok "pushed :${TAG} and :latest"

# =============================================================================
#  Resolve the pushed digest and print the cdk deploy line
# =============================================================================
say "resolving pushed image digest for tag '$TAG'"
IMAGE_DIGEST=$(aws ecr describe-images --repository-name "$ECR_REPO_NAME" --region "$REGION" \
    --image-ids "imageTag=${TAG}" \
    --query 'imageDetails[0].imageDigest' --output text 2>/dev/null || true)
[[ -n "$IMAGE_DIGEST" && "$IMAGE_DIGEST" != "None" ]] \
  || die "could not resolve pushed digest for ${ECR_REPO_NAME}:${TAG}."
ok "pushed image digest: $IMAGE_DIGEST"

RESOLVED_IMAGE="${REPO_URI}@${IMAGE_DIGEST}"

say "done. Pass this verified image to cdk deploy (or to prepare-artifacts.sh's bring-your-own path):"
echo "    --context image_uri=${RESOLVED_IMAGE}"
echo ""
echo "    e.g.: cdk deploy ${STACK} --context compute_backend=eks \\"
echo "              --context image_uri=${RESOLVED_IMAGE} --require-approval never"
