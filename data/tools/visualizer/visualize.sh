#!/bin/sh
# One-command local launcher for lerobot-dataset-visualizer against a
# converted LeRobot v2.1 dataset directory (D-05/D-06).
#
# This tool is optional and explicitly non-gating: it gates no automated
# check in data/validation/, blocks no verification step, and adds no
# Python runtime dependency to this repository (data/serve_dataset.py
# remains stdlib-only). It exists only so a developer can look at a
# converted dataset in a browser; nothing it renders is ever recorded,
# reported, or cited as a validation verdict.
#
# It composes three pieces into one command:
#   1. checkout.sh (only if the pinned base image is missing locally)
#   2. an idempotently-derived "patched" image, whose client bundle has the
#      upstream huggingface.co origin rewritten to the local serve port
#      (Next.js inlines that origin at build time; -e DATASET_URL alone
#      only reaches the server-side fetch path -- see README.md)
#   3. a backgrounded `data.serve_dataset` instance plus the visualizer
#      container, wired together, printing the one URL to open.
#
# Usage:
#   data/tools/visualizer/visualize.sh --root PATH [--serve-port N] [--viz-port N]
#
#   --root PATH       (required) Directory containing a converted LeRobot
#                     v2.1 dataset (must contain meta/info.json). No
#                     default: data/output is not gitignored, so defaulting
#                     there would risk pulling a ~60MB untracked dataset
#                     into the working tree.
#   --serve-port N    Port for the local data.serve_dataset backend.
#                     Default: 8000.
#   --viz-port N      Port for the visualizer container (published on the
#                     host; the container's internal port is always 7860).
#                     Default: 7860.
#   --help            Show this help and exit.
#
# Environment overrides:
#   PYTHON                  Python interpreter to run data.serve_dataset
#                            with. Default: "<repo-root>/.venv/bin/python",
#                            falling back to "python3" if that path is
#                            absent (the module is stdlib-only, so either
#                            interpreter works).
#   VISUALIZER_COMMIT_SHA    Override the pinned upstream commit. Passed
#                            through to checkout.sh untouched; must match
#                            checkout.sh's own override so the tags agree.
#   CHECKOUT_DIR             Passed through to checkout.sh if a base image
#                            build is triggered.
#
# On success, prints the exact URL to open. Ctrl-C stops the visualizer
# container and the backgrounded server.

set -eu

usage() {
    cat <<'EOF'
Usage: data/tools/visualizer/visualize.sh --root PATH [--serve-port N] [--viz-port N]

  --root PATH       (required) Directory containing a converted LeRobot
                     v2.1 dataset (must contain meta/info.json). No default.
  --serve-port N    Port for the local data.serve_dataset backend. Default: 8000.
  --viz-port N      Port for the visualizer container. Default: 7860.
  --help            Show this help and exit.

Environment overrides: PYTHON, VISUALIZER_COMMIT_SHA, CHECKOUT_DIR.
See the header comment in this script for details.
EOF
}

ROOT=""
SERVE_PORT=8000
VIZ_PORT=7860

while [ $# -gt 0 ]; do
    case "$1" in
        --root)
            ROOT="${2:-}"
            shift 2
            ;;
        --serve-port)
            SERVE_PORT="${2:-}"
            shift 2
            ;;
        --viz-port)
            VIZ_PORT="${2:-}"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [ -z "${ROOT}" ]; then
    echo "error: --root PATH is required (no default -- data/output is not gitignored, so defaulting there would risk an untracked dataset in the working tree)" >&2
    usage >&2
    exit 1
fi

case "${SERVE_PORT}" in
    ''|*[!0-9]*)
        echo "error: --serve-port must be a positive integer, got: ${SERVE_PORT}" >&2
        exit 1
        ;;
esac

case "${VIZ_PORT}" in
    ''|*[!0-9]*)
        echo "error: --viz-port must be a positive integer, got: ${VIZ_PORT}" >&2
        exit 1
        ;;
esac

# The dataset origin baked into the visualizer's client bundle at patch time.
# (a) This value is inlined into the Next.js client bundle when the patched
#     image is derived below -- it is the browser, not this script, that
#     must be able to resolve and connect to it.
# (b) It MUST be an IPv4 address literal, never a resolvable name:
#     data/serve_dataset.py binds DEFAULT_BIND_HOST = "127.0.0.1" (IPv4-only,
#     deliberate -- see its "Binding." docstring paragraph), while a name can
#     resolve to the IPv6 loopback address first, which is the macOS default.
# (c) A browser's fetch() does not retry across address families the way
#     curl transparently does, so the failure mode is a page shell that
#     loads while every dataset fetch silently fails.
# (d) The guard below only enforces that the first character after the
#     scheme is a digit -- enough to reject a name, deliberately NOT a full
#     IPv4 validator.
SERVE_ORIGIN="http://127.0.0.1:${SERVE_PORT}"
case "${SERVE_ORIGIN}" in
    http://[0-9]*.[0-9]*.[0-9]*.[0-9]*:[0-9]*)
        ;;
    *)
        echo "FATAL: refusing to bake a non-IPv4-literal dataset origin (${SERVE_ORIGIN}) -- data/serve_dataset.py binds the IPv4 loopback only, and a resolvable name can resolve to the IPv6 loopback first (the macOS default), which a browser's fetch() will not fall back from" >&2
        exit 1
        ;;
esac

# --- Root sanity check, before anything is started. ---

if [ ! -d "${ROOT}" ]; then
    echo "error: --root ${ROOT} is not a directory" >&2
    exit 1
fi

ROOT_ABS=$(cd "${ROOT}" && pwd)

if [ ! -f "${ROOT_ABS}/meta/info.json" ]; then
    echo "error: --root ${ROOT_ABS} is missing meta/info.json -- expected a LeRobot dataset directory laid out as <root>/meta/info.json, <root>/data/..., <root>/videos/..." >&2
    exit 1
fi

CODEBASE_VERSION=$(sed -n 's/.*"codebase_version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "${ROOT_ABS}/meta/info.json" | head -n1)
if [ -n "${CODEBASE_VERSION}" ] && [ "${CODEBASE_VERSION}" != "v2.1" ]; then
    echo "warning: ${ROOT_ABS}/meta/info.json declares codebase_version=${CODEBASE_VERSION}, not v2.1 -- the pinned visualizer also supports v2.0/v3.0, continuing anyway" >&2
fi

# --- Single source of truth for the pin: read IMAGE_NAME and the pinned ---
# --- SHA out of checkout.sh rather than re-typing either. ---

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CHECKOUT_SH="${SCRIPT_DIR}/checkout.sh"

IMAGE_NAME=$(sed -n 's/^IMAGE_NAME="\(.*\)"$/\1/p' "${CHECKOUT_SH}")
if [ -z "${IMAGE_NAME}" ]; then
    echo "FATAL: could not extract IMAGE_NAME out of ${CHECKOUT_SH} -- its variable style changed; update this script's extraction pattern" >&2
    exit 1
fi

DEFAULT_SHA=$(sed -n 's/^DEFAULT_VISUALIZER_COMMIT_SHA="\([0-9a-f]\{40\}\)"$/\1/p' "${CHECKOUT_SH}")
if [ -z "${DEFAULT_SHA}" ]; then
    echo "FATAL: could not extract DEFAULT_VISUALIZER_COMMIT_SHA out of ${CHECKOUT_SH} -- its variable style changed; update this script's extraction pattern" >&2
    exit 1
fi

VISUALIZER_COMMIT_SHA="${VISUALIZER_COMMIT_SHA:-${DEFAULT_SHA}}"
SHORT_SHA=$(printf '%s' "${VISUALIZER_COMMIT_SHA}" | cut -c1-12)

BASE_TAG="${IMAGE_NAME}:${SHORT_SHA}"
# The serve port is baked into the tag on purpose: the patch writes a
# literal origin (including its port) into the client bundle, so one
# patched image is only ever valid for one serve port. The viz port is
# never baked (only published at `docker run`), so it stays out of the tag.
PATCHED_TAG="${IMAGE_NAME}:${SHORT_SHA}-localpatch-p${SERVE_PORT}"

# --- Image resolution (D2 -- idempotent). ---

if ! docker image inspect "${BASE_TAG}" >/dev/null 2>&1; then
    echo "Base image ${BASE_TAG} not found locally -- running checkout.sh..."
    "${CHECKOUT_SH}"
    if ! docker image inspect "${BASE_TAG}" >/dev/null 2>&1; then
        echo "FATAL: ${BASE_TAG} still not found after running checkout.sh" >&2
        exit 1
    fi
fi

# Correctness property: absence, an empty label, an inspect failure, and a
# disagreeing label must all resolve to re-derive, never to reuse. Failing
# toward a rebuild costs one derive; failing toward reuse would silently
# serve a broken bundle -- the exact defect this fix closes. Every patched
# image built before this change is unlabeled, so this comparison
# invalidates all of them automatically -- no cleanup command is required.
CACHED_ORIGIN=$(docker image inspect "${PATCHED_TAG}" --format '{{index .Config.Labels "data-engine.serve-origin"}}' 2>/dev/null || true)

if [ "${CACHED_ORIGIN}" = "${SERVE_ORIGIN}" ]; then
    echo "Reusing cached patched image ${PATCHED_TAG} (already derived for serve port ${SERVE_PORT}, origin ${SERVE_ORIGIN})."
else
    if docker image inspect "${PATCHED_TAG}" >/dev/null 2>&1; then
        echo "Cached image ${PATCHED_TAG} predates the origin fix (label: ${CACHED_ORIGIN:-(none)}, required: ${SERVE_ORIGIN}) -- superseding it. docker commit re-points the tag in place, so no docker rmi is needed; the superseded image becomes dangling and is reclaimed by a normal docker image prune."
    fi
    echo "Deriving patched image ${PATCHED_TAG} from ${BASE_TAG} (client bundle origin -> ${SERVE_ORIGIN})..."

    DERIVE_CONTAINER="data-engine-visualizer-derive-p${SERVE_PORT}"
    docker rm -f "${DERIVE_CONTAINER}" >/dev/null 2>&1 || true

    # Built with single quotes (no outer-shell expansion at all) so the
    # inner shell's own $MATCHES/$f stay literal; the only substitution is
    # a plain token swap done afterward with the already-validated numeric
    # SERVE_PORT, never a raw shell interpolation into the container script.
    PATCH_SCRIPT_TEMPLATE='set -eu
MATCHES=$(grep -rl "https://huggingface.co/datasets" /app/.next/static || true)
if [ -z "$MATCHES" ]; then
    echo "FATAL: no client bundle files under /app/.next/static matched the upstream origin literal -- upstream build shape changed" >&2
    exit 1
fi
printf "%s\n" "$MATCHES" | while IFS= read -r f; do
    sed -i "s|https://huggingface.co/datasets|__SERVE_ORIGIN__|g" "$f"
done
if grep -rq "https://huggingface.co/datasets" /app/.next/static; then
    echo "FATAL: residual upstream origin occurrences remain under /app/.next/static after patch" >&2
    exit 1
fi
if ! grep -rq "__SERVE_ORIGIN__" /app/.next/static; then
    echo "FATAL: intended local origin is absent from the client bundle under /app/.next/static after the patch" >&2
    exit 1
fi
'
    PATCH_SCRIPT=$(printf '%s' "${PATCH_SCRIPT_TEMPLATE}" | sed "s|__SERVE_ORIGIN__|${SERVE_ORIGIN}|g")

    if ! docker run --name "${DERIVE_CONTAINER}" --entrypoint sh "${BASE_TAG}" -c "${PATCH_SCRIPT}"; then
        echo "FATAL: patch step failed inside the derive container" >&2
        docker rm -f "${DERIVE_CONTAINER}" >/dev/null 2>&1 || true
        exit 1
    fi

    # `docker commit` on a container created with --entrypoint sh would
    # otherwise persist sh as the image entrypoint. Restore the upstream
    # config by reading it back off the BASE image, not by hardcoding it,
    # so a future pin bump stays honest (D-06).
    BASE_ENTRYPOINT=$(docker image inspect "${BASE_TAG}" --format '{{json .Config.Entrypoint}}')
    BASE_CMD=$(docker image inspect "${BASE_TAG}" --format '{{json .Config.Cmd}}')

    docker commit --change "ENTRYPOINT ${BASE_ENTRYPOINT}" --change "CMD ${BASE_CMD}" --change "LABEL data-engine.serve-origin=${SERVE_ORIGIN}" "${DERIVE_CONTAINER}" "${PATCHED_TAG}" >/dev/null
    docker rm -f "${DERIVE_CONTAINER}" >/dev/null 2>&1 || true
    echo "Derived ${PATCHED_TAG}."
fi

# --- Run: backgrounded data.serve_dataset, then the patched container. ---

REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
if [ ! -x "${PYTHON}" ]; then
    PYTHON="python3"
fi

CONTAINER_NAME="data-engine-visualizer-p${VIZ_PORT}"
SERVER_PID=""

cleanup() {
    docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    if [ -n "${SERVER_PID}" ]; then
        kill "${SERVER_PID}" >/dev/null 2>&1 || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 0' INT TERM

(cd "${REPO_ROOT}" && exec "${PYTHON}" -m data.serve_dataset --root "${ROOT_ABS}" --port "${SERVE_PORT}") &
SERVER_PID=$!

echo "Starting data.serve_dataset (pid ${SERVER_PID}) on port ${SERVE_PORT}, serving ${ROOT_ABS}..."

READY=0
i=1
while [ "${i}" -le 60 ]; do
    if curl -sf -o /dev/null "http://127.0.0.1:${SERVE_PORT}/local/dataset/resolve/main/meta/info.json"; then
        READY=1
        break
    fi
    i=$((i + 1))
    sleep 1
done

if [ "${READY}" -ne 1 ]; then
    echo "FATAL: data.serve_dataset did not respond on port ${SERVE_PORT} within 60s -- refusing to start the container against a dead backend" >&2
    exit 1
fi

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

echo "Starting visualizer container ${CONTAINER_NAME} on port ${VIZ_PORT}..."
if ! docker run -d --rm --name "${CONTAINER_NAME}" \
    -p "${VIZ_PORT}:7860" \
    -e "DATASET_URL=http://host.docker.internal:${SERVE_PORT}" \
    --add-host=host.docker.internal:host-gateway \
    "${PATCHED_TAG}" >/dev/null; then
    echo "FATAL: failed to start the visualizer container ${PATCHED_TAG}" >&2
    exit 1
fi

echo ""
# The hostname form is deliberate here, unlike SERVE_ORIGIN above: Docker
# publishes the viz port dual-stack, so "localhost" resolves to whichever
# loopback address is reachable either way. Do not "fix" this to an address
# literal.
echo "Open: http://localhost:${VIZ_PORT}/local/dataset/0"
echo "The dataset id is fixed and pre-wired -- there is nothing to type in."
echo "Reminder: this view is an inspection aid, never a validation verdict (D-05/D-06)."
echo ""
echo "Press Ctrl-C to stop."

wait "${SERVER_PID}"
