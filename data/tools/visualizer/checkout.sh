#!/bin/sh
# Pin-by-SHA checkout and local Docker image build for
# huggingface/lerobot-dataset-visualizer (D-05/D-06).
#
# This tool is optional and explicitly non-gating: it gates no automated
# check in data/validation/, blocks no verification, and adds no Python
# runtime dependency to this repository. It exists only so a developer can
# visually browse a locally converted LeRobot v2.1 dataset.
#
# Non-negotiable pin discipline (D-06): the upstream repository is checked
# out at exactly one recorded commit SHA, never a branch name or `latest`.
# A quality judgement made with an unversioned tool is not reproducible, so
# this script refuses to build from a SHA that does not resolve in the
# clone rather than silently substituting a moving reference.
#
# No TypeScript, lockfile, or other vendored upstream source is ever copied
# into this repository — the clone lives entirely under CHECKOUT_DIR
# (default: a gitignored scratch path outside this repo's own tree), and
# this script only ever reads from it to run `docker build`.
#
# Usage:
#   data/tools/visualizer/checkout.sh [checkout-dir]
#
# Environment overrides:
#   VISUALIZER_COMMIT_SHA   Override the pinned commit (must be a full
#                           40-hex-character SHA). Default is the value
#                           recorded below, resolved via
#                           `git ls-remote https://github.com/huggingface/lerobot-dataset-visualizer refs/heads/main`
#                           on 2026-09-06 (see README.md in this directory).
#   CHECKOUT_DIR            Override the clone working directory. Default:
#                           "${TMPDIR:-/tmp}/lerobot-dataset-visualizer-checkout"
#
# On success, prints the exact `docker run` command to start the container,
# wiring DATASET_URL at the data/serve_dataset.py shim (Task 2 of this plan)
# and publishing container port 7860.

set -eu

REPO_URL="https://github.com/huggingface/lerobot-dataset-visualizer"
IMAGE_NAME="lerobot-dataset-visualizer"

# Resolved via `git ls-remote "${REPO_URL}" refs/heads/main` on 2026-09-06.
# This is a snapshot, not a permanent fact -- re-resolve before treating an
# old copy of this script as still pointing at a sensible reference, and
# override via the VISUALIZER_COMMIT_SHA environment variable rather than
# editing this line, so a later phase can move the pin without touching the
# script body.
DEFAULT_VISUALIZER_COMMIT_SHA="dc59887796fd41f37040c0df6b10e6f6a30a1854"
VISUALIZER_COMMIT_SHA="${VISUALIZER_COMMIT_SHA:-${DEFAULT_VISUALIZER_COMMIT_SHA}}"

CHECKOUT_DIR="${1:-${CHECKOUT_DIR:-${TMPDIR:-/tmp}/lerobot-dataset-visualizer-checkout}}"

case "${CHECKOUT_DIR}" in
    "$(cd "$(dirname "$0")/../../.." && pwd)"*)
        echo "FATAL: CHECKOUT_DIR (${CHECKOUT_DIR}) resolves inside this repository." >&2
        echo "D-06 forbids vendoring upstream source into this repo -- choose a path outside it." >&2
        exit 1
        ;;
esac

echo "== lerobot-dataset-visualizer checkout =="
echo "Repository:   ${REPO_URL}"
echo "Pinned SHA:   ${VISUALIZER_COMMIT_SHA}"
echo "Checkout dir: ${CHECKOUT_DIR}"

if [ -d "${CHECKOUT_DIR}/.git" ]; then
    echo "-- Existing clone found, fetching..."
    git -C "${CHECKOUT_DIR}" fetch --quiet origin
else
    echo "-- Cloning fresh..."
    rm -rf "${CHECKOUT_DIR}"
    git clone --quiet "${REPO_URL}" "${CHECKOUT_DIR}"
fi

# Verify the requested SHA actually resolves to a commit in this clone
# before doing anything else. A branch name or a moving tag must never be
# silently substituted for the recorded commit (D-06).
if ! git -C "${CHECKOUT_DIR}" cat-file -e "${VISUALIZER_COMMIT_SHA}^{commit}" 2>/dev/null; then
    echo "FATAL: commit ${VISUALIZER_COMMIT_SHA} does not resolve in ${REPO_URL}." >&2
    echo "Refusing to build from an unresolvable pin. Re-resolve the SHA (e.g. via" >&2
    echo "'git ls-remote ${REPO_URL} refs/heads/main') and pass it via VISUALIZER_COMMIT_SHA." >&2
    exit 1
fi

echo "-- Checking out ${VISUALIZER_COMMIT_SHA}..."
git -C "${CHECKOUT_DIR}" checkout --quiet "${VISUALIZER_COMMIT_SHA}"

SHORT_SHA=$(printf '%s' "${VISUALIZER_COMMIT_SHA}" | cut -c1-12)
IMAGE_TAG="${IMAGE_NAME}:${SHORT_SHA}"

echo "-- Building Docker image ${IMAGE_TAG} from the upstream repository's own Dockerfile..."
docker build --tag "${IMAGE_TAG}" "${CHECKOUT_DIR}"

echo ""
echo "== Build complete: ${IMAGE_TAG} =="
echo ""
echo "Supported one-command path (builds the patched image and wires everything up):"
echo ""
echo "  data/tools/visualizer/visualize.sh --root <dataset-dir>"
echo ""
echo "Note: a raw 'docker run' of this base image alone leaves browser-side parquet/mp4"
echo "fetches pointed at huggingface.co -- DATASET_URL only reaches the server-side path"
echo "(see data/tools/visualizer/README.md for why). visualize.sh derives a patched image"
echo "instead of relying on that variable alone."
