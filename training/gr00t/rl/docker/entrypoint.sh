#!/bin/bash
# Entrypoint for GR00T RL training nodes on AWS Batch MNP.
#
# AWS Batch injects these env vars into all MNP containers:
#   AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS - main node's private IP
#   AWS_BATCH_JOB_NODE_INDEX - this node's index (0 = main)
#   AWS_BATCH_JOB_NUM_NODES - total node count
#
# Node 0 (main): starts Ray head + RLinf learner
# Node 1..N (children): joins Ray cluster + RLinf rollout workers

set -euo pipefail

echo "============================================"
echo "GR00T RL Training — AWS Batch MNP"
echo "============================================"
echo "Node Index: ${AWS_BATCH_JOB_NODE_INDEX}"
echo "Total Nodes: ${AWS_BATCH_JOB_NUM_NODES}"
echo "Main Node IP: ${AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS}"
echo "Node Role: ${NODE_ROLE:-auto}"
echo "============================================"

RAY_PORT="${RAY_PORT:-6379}"
RAY_HEAD_ADDRESS="${AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS}:${RAY_PORT}"

# Install flash-attn if not present (requires CUDA toolkit on GPU node)
if [ -d "/isaac-sim" ]; then
    /isaac-sim/python.sh -c "import flash_attn" 2>/dev/null || {
        echo "Installing flash-attn (first run on GPU node)..."
        /isaac-sim/python.sh -m pip install --no-cache-dir flash-attn --no-build-isolation 2>&1 | tail -5 || \
            echo "WARN: flash-attn install failed, falling back to PyTorch SDPA"
    }
fi

# Configure PYTHONPATH for RLinf and task extensions
export RLINF_PATH="${EFS_MOUNT}/third_party/RLinf"
export ISAACLAB_PATH="${EFS_MOUNT}/third_party/IsaacLab"
export PYTHONPATH="${RLINF_PATH}:${EFS_MOUNT}/workflows/rheo/scripts:${EFS_MOUNT}/workflows/rheo/scripts/simulation/rl:${PYTHONPATH:-}"
export RLINF_EXT_MODULE=rlinf_ext

# Isaac Sim env (for rollout nodes)
if [ -d "/isaac-sim" ]; then
    export ISAAC_PATH="/isaac-sim"
    export CARB_APP_PATH="${ISAAC_PATH}/kit"
    export EXP_PATH="${ISAAC_PATH}/apps"
    export PYTHONPATH="${ISAAC_PATH}/exts:${ISAAC_PATH}/extscore:${ISAAC_PATH}/extscache:${PYTHONPATH}"
    PYTHON_CMD="/isaac-sim/python.sh"
else
    PYTHON_CMD="python3"
fi

# Wait for EFS mount
echo "Waiting for EFS mount at ${EFS_MOUNT}..."
timeout 120 bash -c "until [ -d '${EFS_MOUNT}/third_party' ]; do sleep 2; done"
echo "EFS mounted successfully."

if [ "${AWS_BATCH_JOB_NODE_INDEX}" = "0" ]; then
    # === MAIN NODE: Ray head + Learner ===
    echo "Starting Ray head on port ${RAY_PORT}..."
    ray start --head \
        --port="${RAY_PORT}" \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="${RAY_DASHBOARD_PORT:-8265}" \
        --num-cpus="${NUM_LEARNER_CPUS:-16}" \
        --num-gpus="${NUM_LEARNER_GPUS:-8}" \
        --block &

    # Wait for Ray to be ready
    sleep 10
    echo "Ray head started. Waiting for ${AWS_BATCH_JOB_NUM_NODES} nodes to join..."

    # Wait for all rollout workers to connect
    EXPECTED_NODES=${AWS_BATCH_JOB_NUM_NODES}
    TIMEOUT=600
    ELAPSED=0
    while true; do
        CONNECTED=$(ray status 2>/dev/null | grep -c "node_" || echo "1")
        if [ "$CONNECTED" -ge "$EXPECTED_NODES" ]; then
            echo "All ${EXPECTED_NODES} nodes connected."
            break
        fi
        if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
            echo "WARNING: Timeout waiting for all nodes. Proceeding with ${CONNECTED} nodes."
            break
        fi
        sleep 10
        ELAPSED=$((ELAPSED + 10))
        echo "  Waiting... ${CONNECTED}/${EXPECTED_NODES} nodes connected (${ELAPSED}s)"
    done

    # Launch RLinf training
    echo "Launching RLinf learner..."
    CONFIG_PATH="${EFS_MOUNT}/workflows/rheo/scripts/simulation/rl/rlinf_ext/config"

    TIMESTAMP=$(date +'%Y%m%d-%H%M%S')
    LOG_DIR="${EFS_MOUNT}/rl-training/results/${CONFIG_NAME}/${TIMESTAMP}"
    mkdir -p "${LOG_DIR}"

    ${PYTHON_CMD} "${RLINF_PATH}/examples/embodiment/train_embodied_agent.py" \
        --config-path "${CONFIG_PATH}" \
        --config-name "${CONFIG_NAME}" \
        actor.model.model_path="${MODEL_PATH}" \
        rollout.model.model_path="${MODEL_PATH}" \
        env.train.total_num_envs="${NUM_ROLLOUT_ENVS:-64}" \
        runner.logger.log_path="${LOG_DIR}" \
        "$@" 2>&1 | tee "${LOG_DIR}/train.log"

    TRAIN_EXIT=$?

    # Upload artifacts to S3
    if [ -n "${S3_BUCKET:-}" ]; then
        echo "Uploading results to s3://${S3_BUCKET}/${S3_PREFIX}/${TIMESTAMP}/..."
        aws s3 sync "${LOG_DIR}" "s3://${S3_BUCKET}/${S3_PREFIX}/${TIMESTAMP}/" \
            --exclude "*.pt" --exclude "*.safetensors"
        echo "Upload complete."
    fi

    exit ${TRAIN_EXIT}

else
    # === CHILD NODE: Ray worker + Rollout ===
    echo "Joining Ray cluster at ${RAY_HEAD_ADDRESS}..."

    # Wait for head node to be reachable
    TIMEOUT=300
    ELAPSED=0
    while ! ray status --address="${RAY_HEAD_ADDRESS}" &>/dev/null; do
        if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
            echo "ERROR: Cannot reach Ray head at ${RAY_HEAD_ADDRESS} after ${TIMEOUT}s"
            exit 1
        fi
        sleep 5
        ELAPSED=$((ELAPSED + 5))
    done

    ray start \
        --address="${RAY_HEAD_ADDRESS}" \
        --num-cpus="${NUM_ROLLOUT_CPUS:-14}" \
        --num-gpus=1 \
        --block

    # Ray worker runs until the head node terminates the job
    echo "Ray worker exiting (head node terminated)."
fi
