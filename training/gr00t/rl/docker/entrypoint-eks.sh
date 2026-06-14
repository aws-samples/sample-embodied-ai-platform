#!/bin/bash
# =============================================================================
# GR00T RL Training Entrypoint — EKS/KubeRay Mode
# =============================================================================
# This entrypoint runs ONLY on the head pod AFTER KubeRay has started Ray via
# $KUBERAY_GEN_RAY_START_CMD. Worker pods run `sleep infinity` after Ray starts
# (handled in RayCluster CR args, not this script).
#
# KubeRay manages Ray cluster formation — this script does NOT run `ray start`.
# It sets up paths, waits for workers to join, then launches training.
#
# Invoked by RayCluster head container args:
#   "ulimit -n 65536; $KUBERAY_GEN_RAY_START_CMD && /opt/entrypoint-eks.sh"
# =============================================================================

set -euo pipefail

# ==============================================================
# Section 1: Banner/logging
# ==============================================================
echo "=== GR00T RL Training — EKS/KubeRay Mode ==="
echo "Hostname: $(hostname)"
echo "Date: $(date -u)"
echo "NODE_ROLE: ${NODE_ROLE:-unset}"

# ==============================================================
# Section 2: Path setup (EFS mounted via PVC at /mnt/efs)
# ==============================================================
EFS_MOUNT="/mnt/efs"

RLINF_PATH="${EFS_MOUNT}/third_party/RLinf"
GROOT_PATH="${EFS_MOUNT}/third_party/Isaac-GR00T"
EMBODIED_PATH="${EFS_MOUNT}/third_party/embodied-ai-platform"
ISAACLAB_PATH="${EFS_MOUNT}/third_party/IsaacLab"
WORKFLOW_PATH="${EFS_MOUNT}/workflows/rheo/scripts"
MODEL_PATH="${MODEL_PATH:-${EFS_MOUNT}/models/GR00T-N1.5-RL-Rheo-AssembleTrocar}"

export PYTHONPATH="${RLINF_PATH}:${GROOT_PATH}:${EMBODIED_PATH}:${ISAACLAB_PATH}/source:${WORKFLOW_PATH}:${WORKFLOW_PATH}/simulation/rl:${PYTHONPATH:-}"

echo "PYTHONPATH: ${PYTHONPATH}"
echo "MODEL_PATH: ${MODEL_PATH}"

# ==============================================================
# Section 3: Environment variables (redundant with CR env but ensures
# propagation to subprocesses spawned by training script)
# ==============================================================
export RLINF_EXT_MODULE=rlinf_ext
export TORCHDYNAMO_DISABLE=1
export RAY_DISABLE_VERSION_CHECK=1
export EMBODIED_PATH="${EFS_MOUNT}/third_party/embodied-ai-platform"

# ==============================================================
# Section 4: Wait for EFS mount availability
# ==============================================================
echo "Waiting for EFS mount at ${EFS_MOUNT}/third_party..."
if timeout 120 bash -c "until [ -d '${EFS_MOUNT}/third_party' ]; do sleep 2; done"; then
    echo "EFS mounted successfully."
else
    echo "ERROR: EFS mount not available after 120s"
    exit 1
fi

# ==============================================================
# Section 5: Wait for Ray workers (only if NODE_ROLE=learner)
# ==============================================================
if [ "${NODE_ROLE:-}" = "learner" ]; then
    NUM_EXPECTED_WORKERS=${NUM_ROLLOUT_WORKERS:-4}
    TOTAL_EXPECTED=$((NUM_EXPECTED_WORKERS + 1))
    TIMEOUT=600
    ELAPSED=0

    echo "Waiting for ${TOTAL_EXPECTED} Ray nodes (1 head + ${NUM_EXPECTED_WORKERS} workers)..."

    while true; do
        CONNECTED=$(/isaac-sim/python.sh -c "import ray; ray.init(address='auto'); n=len([x for x in ray.nodes() if x['Alive']]); print(n)" 2>/dev/null || echo "0")
        if [ "${CONNECTED}" -ge "${TOTAL_EXPECTED}" ]; then
            echo "All ${TOTAL_EXPECTED} nodes connected to Ray cluster."
            break
        fi
        sleep 10
        ELAPSED=$((ELAPSED + 10))
        if [ ${ELAPSED} -ge ${TIMEOUT} ]; then
            echo "ERROR: Timed out waiting for Ray workers (${CONNECTED}/${TOTAL_EXPECTED} after ${TIMEOUT}s)"
            exit 1
        fi
        echo "  Waiting... ${CONNECTED}/${TOTAL_EXPECTED} nodes connected (${ELAPSED}s/${TIMEOUT}s)"
    done

    # ==============================================================
    # Section 6: Launch training (only if NODE_ROLE=learner)
    # ==============================================================
    # User-configurable training parameters (override via container env vars)
    CONFIG_NAME="${CONFIG_NAME:-isaaclab_ppo_gr00t_assemble_trocar}"
    MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-128}"
    GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
    ENVS_PER_WORKER="${ENVS_PER_WORKER:-32}"
    MAX_EPOCHS="${MAX_EPOCHS:-1000}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-2}"

    TRAIN_SCRIPT="${RLINF_PATH}/examples/embodiment/train_embodied_agent.py"
    EXT_CONFIG_PATH="/tmp/rlinf_config_eks"
    LOG_DIR="${EFS_MOUNT}/rl-training/results/${CONFIG_NAME}_eks/$(date +'%Y%m%d-%H%M%S')"
    TOTAL_ENVS=$((NUM_EXPECTED_WORKERS * ENVS_PER_WORKER))

    # Copy config to /tmp and modify for EKS topology (avoids mutating shared EFS)
    cp -r "${EFS_MOUNT}/workflows/rheo/scripts/simulation/rl/rlinf_ext/config" "${EXT_CONFIG_PATH}"
    sed -i 's/actor: 0-3/actor: 0-7/' "${EXT_CONFIG_PATH}/${CONFIG_NAME}.yaml"
    sed -i "s/env,rollout: 4-7/env,rollout: 8-11/" "${EXT_CONFIG_PATH}/${CONFIG_NAME}.yaml"
    sed -i "s/num_nodes: 2/num_nodes: ${TOTAL_EXPECTED}/" "${EXT_CONFIG_PATH}/${CONFIG_NAME}.yaml"

    # Start GPU monitoring in background (logs to EFS for post-run analysis)
    GPU_LOG_DIR="${LOG_DIR}/gpu_metrics"
    mkdir -p "${GPU_LOG_DIR}"
    nvidia-smi dmon -s uct -d 30 -f "${GPU_LOG_DIR}/gpu_dmon.csv" &
    DMON_PID=$!
    echo "GPU monitoring started (PID ${DMON_PID}, logging every 30s to ${GPU_LOG_DIR}/gpu_dmon.csv)"

    echo "Launching training..."
    echo "  Config: ${CONFIG_NAME}"
    echo "  Nodes: ${TOTAL_EXPECTED}"
    echo "  Total envs: ${TOTAL_ENVS}"
    echo "  Micro batch size: ${MICRO_BATCH_SIZE}"
    echo "  Gradient checkpointing: ${GRADIENT_CHECKPOINTING}"
    echo "  Max epochs: ${MAX_EPOCHS}"
    echo "  Log dir: ${LOG_DIR}"

    /isaac-sim/python.sh "${TRAIN_SCRIPT}" \
        --config-path "${EXT_CONFIG_PATH}" \
        --config-name "${CONFIG_NAME}" \
        "actor.model.model_path=${MODEL_PATH}" \
        "rollout.model.model_path=${MODEL_PATH}" \
        "env.train.total_num_envs=${TOTAL_ENVS}" \
        "runner.logger.log_path=${LOG_DIR}" \
        "runner.max_epochs=${MAX_EPOCHS}" \
        "runner.save_interval=${SAVE_INTERVAL}" \
        "hydra.searchpath=[file://${RLINF_PATH}/examples/embodiment/config]" \
        "actor.micro_batch_size=${MICRO_BATCH_SIZE}" \
        "actor.fsdp_config.gradient_checkpointing=${GRADIENT_CHECKPOINTING}"

    # Cleanup GPU monitoring
    kill $DMON_PID 2>/dev/null || true
else
    # ==============================================================
    # Section 7: Fallback for non-learner (should not be reached since
    # workers run `sleep infinity` in RayCluster CR args, but as safety)
    # ==============================================================
    echo "Worker mode (NODE_ROLE=${NODE_ROLE:-unset}) — blocking."
    echo "Ray is managed by KubeRay. This process blocks until pod termination."
    sleep infinity
fi
