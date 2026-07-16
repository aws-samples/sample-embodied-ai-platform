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
# Section 2: Path setup (FSx for Lustre mounted via PVC at /mnt/fsx)
# ==============================================================
FSX_MOUNT="/mnt/fsx"

RLINF_PATH="${FSX_MOUNT}/third_party/RLinf"
GROOT_PATH="${FSX_MOUNT}/third_party/Isaac-GR00T"
EMBODIED_PATH="${FSX_MOUNT}/third_party/embodied-ai-platform"
ISAACLAB_PATH="${FSX_MOUNT}/third_party/IsaacLab"
WORKFLOW_PATH="${FSX_MOUNT}/workflows/rheo/scripts"
MODEL_PATH="${MODEL_PATH:-${FSX_MOUNT}/models/GR00T-N1.5-RL-Rheo-AssembleTrocar}"

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
export EMBODIED_PATH="${FSX_MOUNT}/third_party/embodied-ai-platform"

# ==============================================================
# Section 4: Wait for FSx mount availability
# ==============================================================
echo "Waiting for FSx mount at ${FSX_MOUNT}/third_party..."
if timeout 120 bash -c "until [ -d '${FSX_MOUNT}/third_party' ]; do sleep 2; done"; then
    echo "FSx mounted successfully."
else
    echo "ERROR: FSx mount not available after 120s"
    exit 1
fi

# ==============================================================
# === MODE fork (top level; W2/W3 revision) ====================
# Fork on ${MODE:-train} BEFORE the NODE_ROLE guard so Sections 1-4
# are shared. Each mode arm has its OWN internal NODE_ROLE=learner
# guard; the non-learner `else` inlines the Section 7 worker
# fallback (sleep infinity) so worker reachability is preserved
# under both modes. An unknown MODE value drops into the top-level
# `else` and exits 1 regardless of role.
# ==============================================================
if [ "${MODE:-train}" = "train" ]; then
    if [ "${NODE_ROLE:-}" = "learner" ]; then
        # ==============================================================
        # Section 5: Wait for Ray workers (train mode, learner pod)
        # ==============================================================
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
        # Section 6: Launch training (train mode, learner pod)
        # ==============================================================
        # User-configurable training parameters (override via container env vars)
        CONFIG_NAME="${CONFIG_NAME:-isaaclab_ppo_gr00t_assemble_trocar}"
        MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-32}"
        GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
        ENVS_PER_WORKER="${ENVS_PER_WORKER:-32}"
        MAX_EPOCHS="${MAX_EPOCHS:-1000}"
        SAVE_INTERVAL="${SAVE_INTERVAL:-2}"

        TRAIN_SCRIPT="${RLINF_PATH}/examples/embodiment/train_embodied_agent.py"
        EXT_CONFIG_PATH="/tmp/rlinf_config_eks"
        if [ -n "${RESUME_DIR:-}" ] && [ -n "${RESUME_LOG_DIR:-}" ]; then
            LOG_DIR="${RESUME_LOG_DIR}"
            echo "Resuming run, reusing existing log dir: ${LOG_DIR}"
        else
            LOG_DIR="${FSX_MOUNT}/rl-training/results/${CONFIG_NAME}_eks/$(date +'%Y%m%d-%H%M%S')"
        fi
        TOTAL_ENVS=$((NUM_EXPECTED_WORKERS * ENVS_PER_WORKER))

        # Wait for config to be available on FSx (DRA lazy-loads from S3)
        CONFIG_SRC="${FSX_MOUNT}/workflows/rheo/scripts/simulation/rl/rlinf_ext/config"
        echo "Waiting for training config on FSx..."
        for i in $(seq 1 30); do
            if [ -f "${CONFIG_SRC}/${CONFIG_NAME}.yaml" ]; then
                echo "Config ready."
                break
            fi
            echo "  Config not ready yet (attempt $i/30)..."
            sleep 10
        done

        # Copy config to /tmp and modify for EKS topology (avoids mutating shared FSx)
        cp -r "${CONFIG_SRC}" "${EXT_CONFIG_PATH}"

        # Handle both config formats:
        # Format A (multi-node): actor: 0-3 / env,rollout: 4-7 / num_nodes: 2
        # Format B (single-node): actor,env,rollout: all / num_nodes: 1
        # Rollout GPU indices: 8 learner GPUs (0-7) + N rollout GPUs (8 to 8+N-1)
        ROLLOUT_END=$((7 + NUM_EXPECTED_WORKERS))
        CFG_FILE="${EXT_CONFIG_PATH}/${CONFIG_NAME}.yaml"
        if grep -q "actor,env,rollout: all" "$CFG_FILE"; then
            # Format B: replace colocated placement with heterogeneous EKS topology
            sed -i "s/num_nodes: 1/num_nodes: ${TOTAL_EXPECTED}/" "$CFG_FILE"
            sed -i "s/actor,env,rollout: all/actor: 0-7/" "$CFG_FILE"
            sed -i "/actor: 0-7/a\\      env,rollout: 8-${ROLLOUT_END}" "$CFG_FILE"
        else
            # Format A: adjust existing split placement for 8-GPU learner
            sed -i 's/actor: 0-3/actor: 0-7/' "$CFG_FILE"
            sed -i "s/env,rollout: 4-7/env,rollout: 8-${ROLLOUT_END}/" "$CFG_FILE"
            sed -i "s/num_nodes: 2/num_nodes: ${TOTAL_EXPECTED}/" "$CFG_FILE"
        fi


        # Start GPU monitoring in background (logs to FSx for post-run analysis)
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

        RESUME_ARGS=()
        if [ -n "${RESUME_DIR:-}" ]; then
            echo "  Resume dir: ${RESUME_DIR}"
            RESUME_ARGS+=("runner.resume_dir=${RESUME_DIR}")
        fi

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
            "actor.fsdp_config.gradient_checkpointing=${GRADIENT_CHECKPOINTING}" \
            "${RESUME_ARGS[@]}"

        # Cleanup GPU monitoring
        kill $DMON_PID 2>/dev/null || true
    else
        # ==============================================================
        # Section 7 (inlined): Fallback for non-learner in train mode
        # ==============================================================
        echo "Worker mode (NODE_ROLE=${NODE_ROLE:-unset}) — blocking."
        echo "Ray is managed by KubeRay. This process blocks until pod termination."
        sleep infinity
    fi
elif [ "${MODE}" = "eval" ]; then
    if [ "${NODE_ROLE:-}" = "learner" ]; then
        # ==============================================================
        # Section 5b: EVAL_CKPT validation (eval mode)
        # Empty EVAL_CKPT is a legitimate case (base-model eval — no RL
        # overlay, just the HF snapshot at rollout.model.model_path).
        # If EVAL_CKPT is set, the file must exist (fail-fast). If unset
        # or empty, warn and continue — Hydra `++runner.ckpt_path` is
        # skipped below so rollout worker uses base weights only.
        # ==============================================================
        if [ -z "${EVAL_CKPT:-}" ]; then
            echo "WARN: EVAL_CKPT unset — running base-model eval (no RL checkpoint overlay)"
        elif [ ! -f "${EVAL_CKPT}" ]; then
            echo "ERROR: EVAL_CKPT file not found: ${EVAL_CKPT}" >&2
            exit 1
        else
            echo "EVAL_CKPT validated: ${EVAL_CKPT}"
        fi

        # ==============================================================
        # Section 5b: Wait for Ray workers (eval mode; 2-pod default)
        # 07-CONTEXT.md decision 2: eval branch overrides
        # NUM_EXPECTED_WORKERS default to -1 so TOTAL_EXPECTED=2 matches
        # the 2-pod eval fleet (1 head + 1 worker).
        # ==============================================================
        NUM_EXPECTED_WORKERS=${NUM_ROLLOUT_WORKERS:-1}
        TOTAL_EXPECTED=$((NUM_EXPECTED_WORKERS + 1))
        TIMEOUT=600
        ELAPSED=0

        echo "Waiting for ${TOTAL_EXPECTED} Ray nodes (1 head + ${NUM_EXPECTED_WORKERS} workers) [eval]..."

        while true; do
            CONNECTED=$(/isaac-sim/python.sh -c "import ray; ray.init(address='auto'); n=len([x for x in ray.nodes() if x['Alive']]); print(n)" 2>/dev/null || echo "0")
            if [ "${CONNECTED}" -ge "${TOTAL_EXPECTED}" ]; then
                echo "All ${TOTAL_EXPECTED} nodes connected to Ray cluster [eval]."
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
        # Section 6b: Launch eval (eval mode, learner pod)
        # ==============================================================
        CONFIG_NAME="${CONFIG_NAME:-isaaclab_ppo_gr00t_assemble_trocar}"
        LOG_DIR="${FSX_MOUNT}/rl-training/results/${CONFIG_NAME}_eks/$(date +'%Y%m%d-%H%M%S')"
        EVAL_SCRIPT="${RLINF_PATH}/examples/embodiment/eval_embodied_agent.py"

        # Video sink parent must exist before Hydra evaluates env.eval.video_cfg.
        mkdir -p "${LOG_DIR}/video/eval"

        # Wait for config to be available on FSx (DRA lazy-loads from S3)
        # W2 revision: eval branch mirrors the training branch's config-wait
        # loop so a cold FSx-DRA state doesn't fail with "file not found".
        CONFIG_SRC="${FSX_MOUNT}/workflows/rheo/scripts/simulation/rl/rlinf_ext/config"
        echo "Waiting for eval config on FSx..."
        for i in $(seq 1 30); do
            if [ -f "${CONFIG_SRC}/${CONFIG_NAME}.yaml" ]; then
                echo "Config ready."
                break
            fi
            echo "  Config not ready yet (attempt $i/30)..."
            sleep 10
        done

        # NOTE: eval mode does NOT copy or sed the config. All differences
        # from the training-shaped YAML are expressed as Hydra CLI overrides
        # below (07-CONTEXT.md decision 2). The shared FSx YAML is not mutated.

        # Start GPU monitoring in background (logs to FSx for post-run analysis)
        GPU_LOG_DIR="${LOG_DIR}/gpu_metrics"
        mkdir -p "${GPU_LOG_DIR}"
        nvidia-smi dmon -s uct -d 30 -f "${GPU_LOG_DIR}/gpu_dmon.csv" &
        DMON_PID=$!
        echo "GPU monitoring started (PID ${DMON_PID}, logging every 30s to ${GPU_LOG_DIR}/gpu_dmon.csv)"

        echo "Launching eval..."
        echo "  Config: ${CONFIG_NAME}"
        echo "  Ckpt:   ${EVAL_CKPT:-<none — base-model eval>}"
        echo "  Log dir: ${LOG_DIR}"
        echo "  Video sink: ${LOG_DIR}/video/eval/"

        # Hydra overrides (order from 07-RESEARCH.md §5 / 07-01-PLAN.md interfaces):
        #   cluster.component_placement=...       (RESET to eval-shaped dict; the YAML default
        #                                          may be Format A (actor/env,rollout keys) or
        #                                          Format B (single actor,env,rollout key) —
        #                                          either way we replace the whole dict via
        #                                          Hydra's `+key=...` syntax after clearing it)
        #   cluster.num_nodes=2                   (2-pod head+worker topology)
        #   env.eval.total_num_envs=8             (proven per-pod env footprint)
        #   algorithm.eval_rollout_epoch=8        (8 × 8 = 64 episodes; Wilson 95% CI ~±0.07)
        #   rollout.model.model_path              (base HF snapshot)
        #   actor.model.model_path                (schema-only reference, per hf_worker.py:67)
        #   runner.only_eval=True                 (drives EmbodiedEvalRunner)
        #   runner.ckpt_path                      (torch.load target at hf_worker.py:74-76)
        #   runner.logger.log_path                (video sink parent)
        #   env.eval.video_cfg.save_video=True    (explicit override guards future YAML edits)
        #   hydra.searchpath                      (resolves default_runtime + trainer.yaml refs)
        # `++key=val` = set-or-add (works whether the key exists or not).
        # Used liberally here because RLinf's config schema is a strict struct
        # and adding new keys via plain `key=val` fails.
        # NOTE: `~cluster.component_placement` clears any existing dict-key from the YAML
        # (Format B's `actor,env,rollout: all`) then we re-add just the eval-needed keys.
        # We MUST include `actor` even though eval doesn't spawn an actor worker, because
        # RLinf's shared `validate_cfg` unconditionally calls `get_world_size('actor')` on
        # FSDP configs — colocating actor on the same GPU as env/rollout satisfies the
        # validator without spawning anything (the eval script only creates env+rollout).
        # Build Hydra overrides. `++runner.ckpt_path` is only emitted when
        # EVAL_CKPT is non-empty — otherwise rollout worker's torch.load()
        # would receive an empty string. Empty EVAL_CKPT means "base-model
        # eval": use rollout.model.model_path (base HF snapshot) as-is.
        CKPT_ARG=()
        if [ -n "${EVAL_CKPT:-}" ]; then
            CKPT_ARG+=("++runner.ckpt_path=${EVAL_CKPT}")
        fi

        /isaac-sim/python.sh "${EVAL_SCRIPT}" \
            --config-path "${CONFIG_SRC}" \
            --config-name "${CONFIG_NAME}" \
            '~cluster.component_placement' \
            '+cluster.component_placement.actor=1' \
            '+cluster.component_placement.rollout=1' \
            '+cluster.component_placement.env=1' \
            "cluster.num_nodes=2" \
            "env.eval.total_num_envs=8" \
            "algorithm.eval_rollout_epoch=8" \
            "rollout.model.model_path=${MODEL_PATH}" \
            "actor.model.model_path=${MODEL_PATH}" \
            "++runner.only_eval=True" \
            "${CKPT_ARG[@]}" \
            "runner.logger.log_path=${LOG_DIR}" \
            "++env.eval.video_cfg.save_video=${SAVE_VIDEO:-False}" \
            "hydra.searchpath=[file://${RLINF_PATH}/examples/embodiment/config]"

        # Cleanup GPU monitoring
        kill $DMON_PID 2>/dev/null || true
    else
        # ==============================================================
        # Section 7 (inlined): Fallback for non-learner in eval mode
        # ==============================================================
        echo "Worker mode (NODE_ROLE=${NODE_ROLE:-unset}) — blocking."
        echo "Ray is managed by KubeRay. This process blocks until pod termination."
        sleep infinity
    fi
else
    # ==============================================================
    # Top-level unknown-MODE guard — dies at pod startup regardless of role.
    # ==============================================================
    echo "ERROR: unknown MODE: ${MODE}" >&2
    exit 1
fi
