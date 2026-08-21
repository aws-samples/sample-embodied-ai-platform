#!/usr/bin/env bash
#
# stage-s3-eks.sh — fail-closed S3 staging for the EKS/FSx-Lustre backend.
#
# Populates the S3 data bucket that the EKS stack's FSx-Lustre DRA lazily imports
# from. It clones the pinned third-party repos, APPLIES the RLinf _broadcast patch
# (patches/RLinf-649e7579-broadcast-raise.patch) to the RLinf checkout, downloads
# the RL model, stages the repo-bundled workflows into the layout entrypoint-eks.sh
# expects, and uploads everything to s3://$S3_DATA_BUCKET/{third_party,models,workflows}/.
#
# This is the EKS sibling of docker/buildspec-stage-efs.yml (the Batch/EFS CodeBuild
# path). It differs in two deliberate ways:
#   1. It FAILS CLOSED. The EFS buildspec masks checkout failures with `|| true`; this
#      script does NOT — a failed clone/checkout/patch aborts the run non-zero so a
#      half-staged bucket can never silently ship.
#   2. It APPLIES the _broadcast patch (the EFS buildspec historically did not). The
#      RLinf pin stays at 649e7579; the patch is applied at stage time (see the patch
#      header + the deploy-eks-training SKILL "Stage Data to S3 (EKS)" step).
#
# ─────────────────────────────────────────────────────────────────────────────
#  SAFETY MODEL — READ THIS FIRST
# ─────────────────────────────────────────────────────────────────────────────
# By DEFAULT this script runs in DRY-RUN mode: every `aws s3 sync` runs with
# `--dryrun` appended, so a non-execute run mutates NOTHING in S3 (it still clones,
# patches, and downloads into a LOCAL workdir so you can inspect what would upload).
#
# To actually WRITE to S3 you must pass --execute AND type a confirmation token.
# Every S3 write is routed through the sync helper's --dryrun/--execute gate.
#
#   Dry-run (default, safe):   S3_DATA_BUCKET=<b> AWS_REGION=<r> ./stage-s3-eks.sh
#   Real (writes to S3):       S3_DATA_BUCKET=<b> AWS_REGION=<r> ./stage-s3-eks.sh --execute
#   Syntax check only:         bash -n ./stage-s3-eks.sh
#   Usage:                     ./stage-s3-eks.sh --help
#
# PUBLIC-MIRROR RULE:
#   This script is destined for the public mirror. It contains NO internal IDs.
#   Bucket + region are supplied via env; nothing internal is baked in. Region-agnostic.
#
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# =============================================================================
#  Pinned versions (mirror docker/buildspec-stage-efs.yml EXACTLY)
# =============================================================================
# Full 40-char SHAs (not short refs) so the checkout is unambiguous + repeatable.
RLINF_SHA="649e7579775997ade74efff33a7c23e90c61e60a"
GROOT_SHA="4af2b622892f7dcb5aae5a3fb70bcb02dc217b96"
ISAACLAB_SHA="941ebdf4ad1fbf89018777012bdfa4b5944c758f"
ISAACLAB_ARENA_SHA="dba09956588dddae52897820686efd329d85da12"

RLINF_URL="https://github.com/RLinf/RLinf.git"
GROOT_URL="https://github.com/NVIDIA/Isaac-GR00T.git"
ISAACLAB_URL="https://github.com/isaac-sim/IsaacLab.git"
ISAACLAB_ARENA_URL="https://github.com/isaac-sim/IsaacLab-Arena.git"

MODEL_REPO="nvidia/GR00T-N1.5-RL-Rheo-AssembleTrocar"
# Pin the model to the shipped revision (b54e142) so a re-stage is byte-repeatable
# even if the HF repo's main branch moves. Override with MODEL_REVISION=<sha> if needed.
MODEL_REVISION="${MODEL_REVISION:-b54e14286ed3f8392d614741748739e09c7fefe4}"
MODEL_DIRNAME="GR00T-N1.5-RL-Rheo-AssembleTrocar"

# =============================================================================
#  Configuration (override via env vars / args — NO internal defaults)
# =============================================================================
S3_DATA_BUCKET="${S3_DATA_BUCKET:-}"   # required
AWS_REGION="${AWS_REGION:-}"           # required — no baked region (region-agnostic)

# SCRIPT_DIR anchors the infra dir; the patch lives at ../patches/ from here.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_PATH="${SCRIPT_DIR}/../patches/RLinf-649e7579-broadcast-raise.patch"
PATCH_SENTINEL="RLINF-COLLECTIVE-DESYNC"
# Where the workflows the entrypoint expects are bundled in the repo.
WORKFLOWS_SRC="${SCRIPT_DIR}/../workflows"

WORKDIR=""     # --workdir <dir>; default = a mktemp dir created below
EXECUTE=0
# Non-interactive confirm bypass for CI / CodeBuild (still REQUIRES --execute).
# Set via --yes or STAGE_S3_ASSUME_YES=1. A human at a TTY should NOT use this —
# it exists so docker/buildspec-stage-s3.yml (the GR00T-RL-Stage-S3 CodeBuild
# project) can run this exact engine unattended.
ASSUME_YES=0
[[ "${STAGE_S3_ASSUME_YES:-0}" =~ ^(1|true|yes|YES)$ ]] && ASSUME_YES=1

# =============================================================================
#  Arg parsing
# =============================================================================
usage() {
  grep -E '^# ' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'USAGE'

stage-s3-eks.sh — usage
  --workdir <dir>   Clone/download into <dir> (default: a fresh mktemp dir).
  --execute         Actually WRITE to S3 (default = dry-run; every sync is --dryrun).
  --yes             Skip the interactive confirm prompt (CI/CodeBuild only; still
                    requires --execute). Also honored via STAGE_S3_ASSUME_YES=1.
  -h | --help       Show this help and exit (zero AWS calls).

Required env vars (no defaults — region-agnostic, no baked bucket):
  S3_DATA_BUCKET    Target S3 bucket (DRA-linked to the EKS stack's FSx-Lustre).
  AWS_REGION        Target region — MUST be the same region as the S3 bucket + FSx.
USAGE
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workdir)  WORKDIR="${2:-}"; shift 2 ;;
    --execute)  EXECUTE=1; shift ;;
    --yes)      ASSUME_YES=1; shift ;;
    -h|--help)  usage ;;
    *) echo "Unknown argument: $1 (use --help)"; exit 2 ;;
  esac
done

# =============================================================================
#  Echo helpers + die
# =============================================================================
say()  { echo ""; echo "==> $*"; }
ok()   { echo "    [OK]   $*"; }
warn() { echo "    [WARN] $*"; }
die()  { echo ""; echo "    [FATAL] $*" >&2; exit 1; }

# =============================================================================
#  Input validation (fail closed BEFORE any clone/download/upload)
# =============================================================================
[[ -n "$S3_DATA_BUCKET" ]] || die "S3_DATA_BUCKET is empty. Export S3_DATA_BUCKET=<bucket> (no baked default in this mirror script)."
[[ -n "$AWS_REGION" ]]     || die "AWS_REGION is empty. Export AWS_REGION=<region> — it MUST match the S3 bucket + FSx region (DRA is same-region)."
[[ -f "$PATCH_PATH" ]]     || die "broadcast patch not found at ${PATCH_PATH} — cannot stage RLinf without it (fail closed)."
[[ -d "$WORKFLOWS_SRC/simulation" && -d "$WORKFLOWS_SRC/policy" ]] \
  || die "repo-bundled workflows not found under ${WORKFLOWS_SRC} (expected simulation/ and policy/)."
command -v git >/dev/null 2>&1 || die "git not found on PATH."
command -v aws >/dev/null 2>&1 || die "aws CLI not found on PATH."
command -v hf  >/dev/null 2>&1 || die "hf (huggingface-hub CLI) not found on PATH — 'pip install huggingface-hub'."

# =============================================================================
#  Workdir
# =============================================================================
if [[ -z "$WORKDIR" ]]; then
  WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/stage-s3-eks.XXXXXX")"
fi
mkdir -p "$WORKDIR"
THIRD_PARTY="${WORKDIR}/third_party"
MODEL_DIR="${WORKDIR}/models/${MODEL_DIRNAME}"
WORKFLOWS_STAGE="${WORKDIR}/workflows"
mkdir -p "$THIRD_PARTY"

# =============================================================================
#  Banner
# =============================================================================
echo "============================================================================"
echo " stage-s3-eks.sh — fail-closed S3 staging for the EKS/FSx-Lustre backend"
echo " Bucket:  s3://${S3_DATA_BUCKET}/    Region: ${AWS_REGION}"
echo " Workdir: ${WORKDIR}"
echo " Patch:   ${PATCH_PATH}"
echo "          (sentinel ${PATCH_SENTINEL} — RLinf pinned @ ${RLINF_SHA:0:7}, patched at stage time)"
if [[ "$EXECUTE" -eq 1 ]]; then
  echo " MODE: *** EXECUTE — THIS WRITES OBJECTS TO s3://${S3_DATA_BUCKET}/ ***"
else
  echo " MODE: DRY-RUN (every 'aws s3 sync' runs with --dryrun; \$0 S3 writes)"
  echo "       Pass --execute to actually upload."
fi
echo "============================================================================"

if [[ "$EXECUTE" -eq 1 ]]; then
  if [[ "$ASSUME_YES" -eq 1 ]]; then
    warn "Non-interactive confirm (--yes / STAGE_S3_ASSUME_YES) — writing to s3://${S3_DATA_BUCKET}/"
  else
    read -r -p "Type 'stage-s3-eks' to confirm writing to s3://${S3_DATA_BUCKET}/ : " _confirm
    [[ "$_confirm" == "stage-s3-eks" ]] || die "Confirmation mismatch — aborting (nothing written)."
  fi
fi

# =============================================================================
#  clone_at(): fail-closed clone + checkout at an exact SHA. Unlike the EFS
#  buildspec, NO `|| true` — a failed clone/checkout aborts the run non-zero.
#  Uses a treeless partial clone (--filter=blob:none) so any SHA is checkout-able
#  ("shallow-ish": full commit graph, blobs fetched lazily) without a full history.
# =============================================================================
clone_at() {
  local url="$1" sha="$2" dest="$3"
  echo "    clone ${url} -> ${dest} @ ${sha}"
  rm -rf "$dest"
  git clone --filter=blob:none "$url" "$dest" \
    || die "git clone failed for ${url} (fail closed)."
  git -C "$dest" checkout "$sha" \
    || die "git checkout ${sha} failed in ${dest} (fail closed — no || true masking)."
  ok "checked out ${dest} @ ${sha}"
}

# =============================================================================
#  STEP 1 — Clone the pinned third-party repos (fail closed)
# =============================================================================
say "STEP 1 — Clone third-party repos at pinned commits"
clone_at "$RLINF_URL"          "$RLINF_SHA"          "${THIRD_PARTY}/RLinf"
clone_at "$GROOT_URL"          "$GROOT_SHA"          "${THIRD_PARTY}/Isaac-GR00T"
clone_at "$ISAACLAB_URL"       "$ISAACLAB_SHA"       "${THIRD_PARTY}/IsaacLab"
clone_at "$ISAACLAB_ARENA_URL" "$ISAACLAB_ARENA_SHA" "${THIRD_PARTY}/IsaacLab-Arena"

# =============================================================================
#  STEP 2 — Apply the _broadcast patch to the RLinf checkout (fail closed + verify)
# =============================================================================
say "STEP 2 — Apply the _broadcast patch to RLinf (${PATCH_SENTINEL})"
RLINF_DIR="${THIRD_PARTY}/RLinf"
COLLECTIVE_FILE="${RLINF_DIR}/rlinf/scheduler/collective/multi_channel_pg.py"
# --check FIRST so a patch that won't apply cleanly aborts before we mutate anything.
git -C "$RLINF_DIR" apply --check "$PATCH_PATH" \
  || die "patch --check failed against ${RLINF_DIR} — the pin may have drifted from 649e7579. Refusing to stage an unpatched RLinf (fail closed)."
git -C "$RLINF_DIR" apply "$PATCH_PATH" \
  || die "git apply failed against ${RLINF_DIR} (fail closed)."
# VERIFY the sentinel actually landed in the target file.
grep -q "$PATCH_SENTINEL" "$COLLECTIVE_FILE" \
  || die "patch verification failed — sentinel ${PATCH_SENTINEL} absent from ${COLLECTIVE_FILE}."
ok "broadcast patch landed — ${PATCH_SENTINEL} present in multi_channel_pg.py (RLinf stays pinned @ ${RLINF_SHA:0:7})."

# =============================================================================
#  STEP 3 — Download the RL model (guard: skip if already present)
# =============================================================================
say "STEP 3 — Download model ${MODEL_REPO}"
if [[ -f "${MODEL_DIR}/config.json" ]]; then
  ok "model already present at ${MODEL_DIR} (config.json exists) — skipping download."
else
  mkdir -p "$MODEL_DIR"
  echo "    hf download ${MODEL_REPO} --revision ${MODEL_REVISION} --local-dir ${MODEL_DIR}"
  hf download "$MODEL_REPO" --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR" \
    || die "hf download failed for ${MODEL_REPO}@${MODEL_REVISION} (fail closed — is the model gated? set HF_TOKEN)."
  ok "model downloaded to ${MODEL_DIR} @ ${MODEL_REVISION}"
fi

# =============================================================================
#  STEP 4 — Stage the repo-bundled workflows into the entrypoint layout
#  entrypoint-eks.sh expects WORKFLOW_PATH=/mnt/fsx/workflows/rheo/scripts, with
#  simulation/rl/rlinf_ext under workflows/rheo/scripts/simulation/rl/rlinf_ext.
#  Mirror the EFS buildspec's cp layout (cp -r simulation and policy into it).
# =============================================================================
say "STEP 4 — Stage workflows into workflows/rheo/scripts/"
WORKFLOWS_SCRIPTS="${WORKFLOWS_STAGE}/rheo/scripts"
rm -rf "$WORKFLOWS_STAGE"
mkdir -p "$WORKFLOWS_SCRIPTS"
cp -r "${WORKFLOWS_SRC}/simulation" "${WORKFLOWS_SCRIPTS}/"
cp -r "${WORKFLOWS_SRC}/policy"     "${WORKFLOWS_SCRIPTS}/"
ok "workflows staged at ${WORKFLOWS_SCRIPTS} (simulation/, policy/)"

# =============================================================================
#  STEP 5 — Upload to S3 (every sync gated by --dryrun unless --execute)
# =============================================================================
# s3_sync(): the single S3-write choke point. In dry-run it appends --dryrun so the
# aws CLI prints what WOULD copy and writes nothing; in --execute it copies for real.
# Always excludes .git/* (partial-clone metadata must never land in the bucket) and
# uses --delete so each prefix CONVERGES to the exact staged tree (stale files from a
# prior/interrupted run are removed — important after the patch or a pin changes).
s3_sync() {
  local src="$1" dst="$2"
  local args=( aws s3 sync "$src" "$dst" --region "$AWS_REGION" --exclude '.git/*' --delete )
  if [[ "$EXECUTE" -eq 1 ]]; then
    echo "    RUN >> ${args[*]}"
    "${args[@]}"
  else
    echo "    DRY-RUN >> ${args[*]} --dryrun"
    "${args[@]}" --dryrun
  fi
}

say "STEP 5 — Upload to s3://${S3_DATA_BUCKET}/"
s3_sync "${THIRD_PARTY}/RLinf"          "s3://${S3_DATA_BUCKET}/third_party/RLinf/"
s3_sync "${THIRD_PARTY}/Isaac-GR00T"    "s3://${S3_DATA_BUCKET}/third_party/Isaac-GR00T/"
s3_sync "${THIRD_PARTY}/IsaacLab"       "s3://${S3_DATA_BUCKET}/third_party/IsaacLab/"
s3_sync "${THIRD_PARTY}/IsaacLab-Arena" "s3://${S3_DATA_BUCKET}/third_party/IsaacLab-Arena/"
s3_sync "${MODEL_DIR}"                  "s3://${S3_DATA_BUCKET}/models/${MODEL_DIRNAME}/"
s3_sync "${WORKFLOWS_STAGE}"            "s3://${S3_DATA_BUCKET}/workflows/"

# =============================================================================
#  STEP 6 — Verification (execute mode)
# =============================================================================
if [[ "$EXECUTE" -eq 1 ]]; then
  say "STEP 6 — Verify uploaded key prefixes"
  for _pfx in "third_party/RLinf/" "third_party/Isaac-GR00T/" "third_party/IsaacLab/" \
              "third_party/IsaacLab-Arena/" "models/${MODEL_DIRNAME}/" "workflows/rheo/scripts/"; do
    echo "    aws s3 ls s3://${S3_DATA_BUCKET}/${_pfx}"
    aws s3 ls "s3://${S3_DATA_BUCKET}/${_pfx}" --region "$AWS_REGION" || warn "no objects listed under ${_pfx}"
  done
  # Publish the READY marker LAST. Because `set -e` aborts on any earlier sync
  # failure, the marker is present ONLY when all six prefixes converged — operators
  # (and a future entrypoint gate) can check `aws s3 ls s3://<bucket>/_STAGING_COMPLETE`
  # to distinguish "staging finished" from "staging still running / failed".
  echo "READY $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    | aws s3 cp - "s3://${S3_DATA_BUCKET}/_STAGING_COMPLETE" --region "$AWS_REGION" \
    || die "failed to publish _STAGING_COMPLETE marker."
  ok "Upload complete — published s3://${S3_DATA_BUCKET}/_STAGING_COMPLETE."
  echo "    NOTE: FSx-Lustre lazily imports these objects on first access via the DRA —"
  echo "          they need not be pre-warmed; the first pod to read a path triggers the import."
else
  say "DRY-RUN finished — nothing written to S3 (\$0). Re-run with --execute to upload."
  echo "    Local staging is intact under ${WORKDIR} for inspection."
fi
