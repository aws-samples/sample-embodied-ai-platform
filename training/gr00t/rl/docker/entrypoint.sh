#!/bin/bash
# =============================================================================
# GR00T RL Training Entrypoint
# =============================================================================
# Supports two runtime modes:
#   1. batch-mnp: AWS Batch Multi-Node Parallel (EFS-backed, homogeneous instances)
#   2. sagemaker: SageMaker Training with FSx for Lustre (heterogeneous InstanceGroups)
#
# In SageMaker mode, FSx for Lustre is auto-mounted by SageMaker at
# /opt/ml/input/data/training/. The Data Repository Association maps S3 artifacts
# to /artifacts/ on FSx, so the full path structure is:
#   /opt/ml/input/data/training/  -> /{mount_name}/artifacts/ on FSx -> s3://bucket/
#
# The EFS_MOUNT variable is reused across both modes as the base path for all
# downstream path derivations (PYTHONPATH, MODEL_PATH, etc.).
# =============================================================================

set -euo pipefail

echo "=== GR00T RL Training Entrypoint ==="
echo "Hostname: $(hostname)"
echo "Date: $(date -u)"

# ==============================================================
# Detect runtime mode
# ==============================================================
if [ -n "${SM_CURRENT_HOST:-}" ]; then
    RUNTIME="sagemaker"
elif [ -d "/opt/ml/input/data/training/third_party" ]; then
    # SubmitServiceJob (Batch+SageMaker) doesn't set SM_CURRENT_HOST
    # but data is downloaded to /opt/ml/input/data/training/
    RUNTIME="sagemaker"
elif [ -n "${AWS_BATCH_JOB_NODE_INDEX:-}" ]; then
    RUNTIME="batch-mnp"
else
    RUNTIME="${RUNTIME:-local}"
fi

echo "Runtime mode: ${RUNTIME}"

# ==============================================================
# Set base data directory based on runtime
# ==============================================================
if [ "${RUNTIME}" = "sagemaker" ]; then
    # FSx for Lustre is auto-mounted by SageMaker at /opt/ml/input/data/training/
    # The DRA maps S3 artifacts to /artifacts/ on FSx, so the full path structure is:
    #   /opt/ml/input/data/training/  -> /{mount_name}/artifacts/ on FSx -> s3://bucket/
    # SM_CHANNEL_TRAINING env var is set by SageMaker for the "training" channel.
    export EFS_MOUNT="${SM_CHANNEL_TRAINING:-/opt/ml/input/data/training}"
elif [ "${RUNTIME}" = "batch-mnp" ]; then
    export EFS_MOUNT="${EFS_MOUNT:-/mnt/efs}"
else
    # Local development mode
    export EFS_MOUNT="${EFS_MOUNT:-/mnt/efs}"
fi

echo "Data directory (EFS_MOUNT): ${EFS_MOUNT}"

# ==============================================================
# Wait for data mount availability
# ==============================================================
if [ "${RUNTIME}" = "batch-mnp" ]; then
    echo "Waiting for EFS mount at ${EFS_MOUNT}..."
    timeout 120 bash -c "until [ -d '${EFS_MOUNT}/third_party' ]; do sleep 2; done"
    echo "EFS mounted successfully."
else
    echo "FSx data directory: ${EFS_MOUNT}"
    ls -la "${EFS_MOUNT}/" 2>/dev/null || echo "WARN: FSx mount listing failed (may still be loading)"
fi

# ==============================================================
# Set up paths (shared across all modes)
# ==============================================================
# Third-party dependencies (cloned to EFS/FSx from S3)
export RLINF_PATH="${EFS_MOUNT}/third_party/RLinf"
export GROOT_PATH="${EFS_MOUNT}/third_party/Isaac-GR00T"
export EMBODIED_PATH="${EFS_MOUNT}/third_party/embodied-ai-platform"
export ISAACLAB_PATH="${EFS_MOUNT}/third_party/IsaacLab"

# Workflow scripts
export WORKFLOW_PATH="${EFS_MOUNT}/workflows/rheo/scripts"

# Model checkpoint
export MODEL_PATH="${MODEL_PATH:-${EFS_MOUNT}/models/GR00T-N1.5-RL-Rheo-AssembleTrocar}"

# ==============================================================
# Set PYTHONPATH
# ==============================================================
export PYTHONPATH="${RLINF_PATH}:${GROOT_PATH}:${EMBODIED_PATH}:${ISAACLAB_PATH}/source:${WORKFLOW_PATH}:${WORKFLOW_PATH}/simulation/rl:${PYTHONPATH:-}"

echo "PYTHONPATH: ${PYTHONPATH}"
echo "MODEL_PATH: ${MODEL_PATH}"

# ==============================================================
# Determine node role
# ==============================================================
if [ "${RUNTIME}" = "sagemaker" ]; then
    # SageMaker heterogeneous cluster: role determined by instance group
    INSTANCE_GROUP="${SM_HP_SAGEMAKER_INSTANCE_GROUPS:-unknown}"
    echo "SageMaker instance group: ${INSTANCE_GROUP}"

    if echo "${INSTANCE_GROUP}" | grep -q "learner"; then
        NODE_ROLE="learner"
    elif echo "${INSTANCE_GROUP}" | grep -q "rollout"; then
        NODE_ROLE="rollout"
    else
        NODE_ROLE="${NODE_ROLE:-learner}"
        echo "WARN: Could not determine role from instance group, defaulting to: ${NODE_ROLE}"
    fi
elif [ "${RUNTIME}" = "batch-mnp" ]; then
    # Batch MNP: role determined by node index (0 = main = learner)
    if [ "${AWS_BATCH_JOB_NODE_INDEX:-0}" = "0" ]; then
        NODE_ROLE="learner"
    else
        NODE_ROLE="rollout"
    fi
else
    NODE_ROLE="${NODE_ROLE:-learner}"
fi

echo "Node role: ${NODE_ROLE}"

# ==============================================================
# Configuration
# ==============================================================
CONFIG_NAME="${CONFIG_NAME:-isaaclab_ppo_gr00t_assemble_trocar}"
NUM_ROLLOUT_ENVS="${NUM_ROLLOUT_ENVS:-64}"

# RLinf extension module — propagated to Ray actor processes via
# Worker._load_user_extensions() when Cluster() captures env vars
export RLINF_EXT_MODULE=rlinf_ext

# Disable torch.compile (inductor) — hangs indefinitely on multi-GPU with Isaac Sim
export TORCHDYNAMO_DISABLE=1

echo "Config: ${CONFIG_NAME}"
echo "Rollout envs: ${NUM_ROLLOUT_ENVS}"

# ==============================================================
# Form Ray cluster (required before RLinf training)
# ==============================================================
NUM_NODES="${AWS_BATCH_JOB_NUM_NODES:-2}"
RAY_PORT="${RAY_PORT:-6379}"
export RAY_DISABLE_VERSION_CHECK=1

if [ "${RUNTIME}" = "batch-mnp" ]; then
    if [ "${NODE_ROLE}" = "learner" ]; then
        # Main node: start Ray head
        echo "Starting Ray head on port ${RAY_PORT}..."
        /isaac-sim/kit/python/bin/ray start --head --port="${RAY_PORT}" --num-gpus=4 --block &
        RAY_PID=$!
        sleep 5

        # Wait for all nodes to connect
        echo "Ray head started. Waiting for ${NUM_NODES} nodes to join..."
        TIMEOUT=600
        ELAPSED=0
        while true; do
            CONNECTED=$(/isaac-sim/python.sh -c "import ray; ray.init(address='auto'); print(len(ray.nodes()))" 2>/dev/null || echo "0")
            if [ "${CONNECTED}" -ge "${NUM_NODES}" ]; then
                echo "All ${NUM_NODES} nodes connected."
                break
            fi
            sleep 10
            ELAPSED=$((ELAPSED + 10))
            if [ ${ELAPSED} -ge ${TIMEOUT} ]; then
                echo "ERROR: Timed out waiting for nodes (${CONNECTED}/${NUM_NODES} after ${TIMEOUT}s)"
                exit 1
            fi
            echo "  Waiting... ${CONNECTED}/${NUM_NODES} nodes connected (${ELAPSED}s)"
        done
    else
        # Worker node: join head via main node IP
        MAIN_IP="${AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS}"
        echo "Joining Ray cluster at ${MAIN_IP}:${RAY_PORT}..."

        # Wait for head to be reachable
        timeout 300 bash -c "until echo > /dev/tcp/${MAIN_IP}/${RAY_PORT} 2>/dev/null; do sleep 2; done"
        echo "Ray head reachable. Joining cluster..."

        /isaac-sim/kit/python/bin/ray start --address="${MAIN_IP}:${RAY_PORT}" --num-gpus=4 --block &
        RAY_PID=$!
        sleep 5
    fi
fi

# ==============================================================
# Launch training
# ==============================================================
echo "Launching ${NODE_ROLE} process (${NUM_NODES} nodes)..."

TRAIN_SCRIPT="${RLINF_PATH}/examples/embodiment/train_embodied_agent.py"
EXT_CONFIG_PATH="${EFS_MOUNT}/workflows/rheo/scripts/simulation/rl/rlinf_ext/config"
LOG_DIR="${EFS_MOUNT}/rl-training/results/${CONFIG_NAME}/$(date +'%Y%m%d-%H%M%S')"

HYDRA_ARGS=(
    --config-path "${EXT_CONFIG_PATH}"
    --config-name "${CONFIG_NAME}"
)
OVERRIDES=(
    "cluster.num_nodes=${NUM_NODES}"
    "actor.model.model_path=${MODEL_PATH}"
    "rollout.model.model_path=${MODEL_PATH}"
    "env.train.total_num_envs=${NUM_ROLLOUT_ENVS}"
    "runner.logger.log_path=${LOG_DIR}"
    "hydra.searchpath=[file://${RLINF_PATH}/examples/embodiment/config]"
    "actor.micro_batch_size=32"
    "actor.fsdp_config.gradient_checkpointing=True"
)

if [ "${NODE_ROLE}" = "learner" ]; then
    /isaac-sim/python.sh "${TRAIN_SCRIPT}" "${HYDRA_ARGS[@]}" "${OVERRIDES[@]}"
elif [ "${NODE_ROLE}" = "rollout" ]; then
    # Rollout nodes block on Ray worker — training is driven by learner
    echo "Rollout node ready. Blocking on Ray worker process..."
    wait ${RAY_PID}
fi
