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

# --- PPO_MODE async branch ---------------------------------------------------
# The async arm uses a SEPARATE RLinf checkout at
# ${FSX_MOUNT}/third_party/RLinf-async (@ fbc72dd6, native async PPO + upstream
# #1414 _broadcast fix), leaving the frozen sync checkout at
# ${FSX_MOUNT}/third_party/RLinf (@ 649e7579, with the _broadcast patch applied at
# S3-staging time by infra/stage-s3-eks.sh — see patches/RLinf-649e7579-broadcast-raise.patch)
# byte-untouched. This RLINF_PATH reassignment happens HERE — BEFORE the
# PYTHONPATH construction below — so that under PPO_MODE=async, python imports
# (including train_async.py's own package imports) resolve from RLinf-async and
# NOT the sync checkout. Deferring this to Section 6 would leave PYTHONPATH
# pinned at the sync RLinf, silently importing the wrong framework while running
# train_async.py — the one failure that would invalidate the paid comparison.
# When PPO_MODE is unset (or "sync"), RLINF_PATH is byte-identical to the
# historical default and the sync arm is unchanged.
if [ "${PPO_MODE:-sync}" = "async" ]; then
    RLINF_PATH="${FSX_MOUNT}/third_party/RLinf-async"
else
    RLINF_PATH="${FSX_MOUNT}/third_party/RLinf"
fi
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
        # PPO_MODE-gated defaults. When PPO_MODE is unset or "sync", every default
        # below is byte-identical to the sync arm (CONFIG_NAME=isaaclab_ppo_gr00t_assemble_trocar,
        # mbs 32, gradient_checkpointing True, train_embodied_agent.py). When "async", the
        # defaults switch to the async config + validated H100 async FSDP settings
        # (mbs 128, gradient_checkpointing False) so the entrypoint's own Hydra CLI
        # overrides below do NOT clobber the async yaml's baked settings. Explicit
        # container env vars still win in either mode (the ${VAR:-default} pattern).
        # rollout_epoch is NOT touched here — the async yaml keeps it at 1.
        if [ "${PPO_MODE:-sync}" = "async" ]; then
            _DEFAULT_CONFIG_NAME="isaaclab_async_ppo_gr00t_assemble_trocar"
            _DEFAULT_MICRO_BATCH_SIZE="128"
            _DEFAULT_GRADIENT_CHECKPOINTING="False"
            _DEFAULT_TRAIN_SCRIPT="${RLINF_PATH}/examples/embodiment/train_async.py"
        else
            _DEFAULT_CONFIG_NAME="isaaclab_ppo_gr00t_assemble_trocar"
            _DEFAULT_MICRO_BATCH_SIZE="32"
            _DEFAULT_GRADIENT_CHECKPOINTING="True"
            _DEFAULT_TRAIN_SCRIPT="${RLINF_PATH}/examples/embodiment/train_embodied_agent.py"
        fi
        CONFIG_NAME="${CONFIG_NAME:-${_DEFAULT_CONFIG_NAME}}"
        MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-${_DEFAULT_MICRO_BATCH_SIZE}}"
        GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-${_DEFAULT_GRADIENT_CHECKPOINTING}}"
        ENVS_PER_WORKER="${ENVS_PER_WORKER:-32}"
        MAX_EPOCHS="${MAX_EPOCHS:-1000}"
        SAVE_INTERVAL="${SAVE_INTERVAL:-2}"
        # Additive + default-off (mirrors EVAL_TOTAL_ENVS/TASK_DESCRIPTION):
        # optional in-TB eval cadence. -1 => OFF => NO Hydra override is emitted below, so
        # the train launch args stay byte-identical to the default behavior. When set to
        # N, emits runner.val_check_interval=N via the conditional VAL_CHECK_ARGS array.
        # NOTE: runner.save_interval MUST be divisible by val_check_interval.
        VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:--1}"

        TRAIN_SCRIPT="${_DEFAULT_TRAIN_SCRIPT}"
        EXT_CONFIG_PATH="/tmp/rlinf_config_eks"
        if [ -n "${RESUME_DIR:-}" ] && [ -n "${RESUME_LOG_DIR:-}" ]; then
            LOG_DIR="${RESUME_LOG_DIR}"
            echo "Resuming run, reusing existing log dir: ${LOG_DIR}"
        else
            # Provenance-in-path: <config>_<backend>_<mode>/<timestamp>. COMPUTE_BACKEND
            # is set by the CDK RayCluster env; defaults to eks.
            # Distinguishes backend and train vs eval at a glance (was ..._eks/,
            # which collided across modes).
            LOG_DIR="${FSX_MOUNT}/rl-training/results/${CONFIG_NAME}_${COMPUTE_BACKEND:-eks}_train/$(date +'%Y%m%d-%H%M%S')"
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

        # Additive in-TB eval override (conditional-args-array pattern, mirrors RESUME_ARGS):
        # emit runner.val_check_interval ONLY when VAL_CHECK_INTERVAL != -1. When -1/unset the
        # array is empty and the launch args are byte-identical to today (reversibility).
        VAL_CHECK_ARGS=()
        if [ "${VAL_CHECK_INTERVAL}" != "-1" ]; then
            echo "  Val check interval: ${VAL_CHECK_INTERVAL} (in-TB eval every N steps)"
            echo "  NOTE: ensure runner.save_interval=${SAVE_INTERVAL} is divisible by ${VAL_CHECK_INTERVAL}."
            VAL_CHECK_ARGS+=("runner.val_check_interval=${VAL_CHECK_INTERVAL}")
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
            "${RESUME_ARGS[@]}" \
            "${VAL_CHECK_ARGS[@]}"

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
        # Eval branch defaults NUM_EXPECTED_WORKERS to 1 so TOTAL_EXPECTED=2 matches
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
        # Provenance-in-path (see train branch): <config>_<backend>_eval/<timestamp>.
        LOG_DIR="${FSX_MOUNT}/rl-training/results/${CONFIG_NAME}_${COMPUTE_BACKEND:-eks}_eval/$(date +'%Y%m%d-%H%M%S')"
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
        # below. The shared FSx YAML is not mutated.

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

        # Hydra overrides (eval-mode component placement + topology):
        #   cluster.component_placement=...       (RESET to eval-shaped dict; the YAML default
        #                                          may be Format A (actor/env,rollout keys) or
        #                                          Format B (single actor,env,rollout key) —
        #                                          either way we replace the whole dict via
        #                                          Hydra's `+key=...` syntax after clearing it)
        #   cluster.num_nodes=${TOTAL_EXPECTED}   (1 head + N workers, computed from
        #                                          NUM_ROLLOUT_WORKERS env var which is
        #                                          injected by CDK from the top-level
        #                                          `num_rollout_workers` context param)
        #   env.eval.total_num_envs=${EVAL_TOTAL_ENVS:-64}
        #                                          (Benchmark topology uses N=64. Default 64 matches
        #                                          the yaml config byte-identically when EVAL_TOTAL_ENVS
        #                                          is unset. Setting EVAL_TOTAL_ENVS at deploy time — via
        #                                          the CDK `--context eval_total_envs=<N>` param —
        #                                          overrides the yaml default for one deploy without
        #                                          changing the entrypoint's shipped byte-identical
        #                                          fallback. N MUST be divisible by cluster.num_nodes.)
        #   actor.global_batch_size=${EVAL_ACTOR_GBS:-2048}
        #                                          (RLinf's `validate_embodied_cfg` asserts
        #                                          `actor.global_batch_size % (actor.micro_batch_size * actor_world_size) == 0`.
        #                                          At shipped yaml defaults (gbs=2048, mbs=128),
        #                                          world_size must divide 2048/128 = 16 → world_size
        #                                          ∈ {1, 2, 4, 8, 16}. Unusual node counts can violate
        #                                          this; e.g. a 10-node topology needs world_size=10,
        #                                          which requires EVAL_ACTOR_GBS=1280 (1280/128 = 10).
        #                                          Safe to override: actor.global_batch_size is consumed
        #                                          ONLY by the FSDP trainer, which eval_embodied_agent.py
        #                                          never spawns (only MultiStepRolloutWorker + EnvWorker).
        #                                          Override is a validator-appeasement no-op in eval mode.
        #                                          Omit both the context param and env var to keep the
        #                                          shipped default 2048.)
        #   ++env.eval.init_params.task_description=${TASK_DESCRIPTION:-install trocar from box}
        #                                          (RLinf's shipped yaml env/isaaclab_assemble_trocar.yaml
        #                                          uses `task_description: "install trocar from box"` with a
        #                                          comment noting it "yields better results than 'assemble
        #                                          trocar from tray' because the model was trained with
        #                                          this specific task description". This override lets you
        #                                          compare against a baseline that used a different prompt
        #                                          (e.g. "assemble trocar from tray", the SFT dataset's
        #                                          canonical annotation) without editing the shared yaml.
        #                                          Omit both the context param and env var to keep the
        #                                          shipped default "install trocar from box".)
        #
        # EVAL_INJECT_NOISE — diagnostic passthrough (no Hydra override needed here).
        #                                          When EVAL_INJECT_NOISE=true is set at pod-env level
        #                                          (via `--context eval_inject_noise=true` in CDK), the
        #                                          patched RLinf gr00t_action_model.py's `sample_mean_var_val`
        #                                          reads the env var directly at Python inference time and
        #                                          switches the mode=="eval" branch to use the train-mode
        #                                          noise formula (flow_sde + noise_level yaml default).
        #                                          This is a diagnostic for probing whether eval-time
        #                                          sampling temperature affects results — by default the
        #                                          RLinf GR00T eval path is deterministic (x_t_std=0 in
        #                                          gr00t_action_model.py; huggingface_worker.py only passes
        #                                          kwargs={"mode": mode}). Default (unset) preserves the
        #                                          deterministic behavior byte-identically because the FSx-
        #                                          side patch only takes the alternate branch when
        #                                          EVAL_INJECT_NOISE=="true". Fully reversible: unset the
        #                                          env var (and restore the FSx-side patched file).
        #   algorithm.eval_rollout_epoch=1        (episodes = eval_rollout_epoch × total_num_envs; with
        #                                          epoch=1 that is exactly total_num_envs episodes, and
        #                                          ignore_terminations=True; matches yaml)
        #   ++env.eval.ignore_terminations=True   (matches yaml default, explicit for defense)
        #   ++env.eval.use_fixed_reset_state_ids=True (matches yaml default, explicit)
        #   ++env.eval.max_episode_steps=256      (matches yaml default, explicit)
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
        # (upstream's `actor: 0-3, env,rollout: 4-7` DGX-shape) then we re-add
        # ranges scoped to our cluster (0..TOTAL_EXPECTED-1). Placement values MUST
        # be ranges — a scalar (`env=1`) is parsed as "1 process on GPU rank 1"
        # and world_size stays at 1 (which piles every env onto one GPU and OOMs). We MUST
        # include `actor` even though eval doesn't spawn an actor worker, because
        # RLinf's shared `validate_cfg` unconditionally calls `get_world_size('actor')` on
        # FSDP configs — a matching range satisfies the validator without spawning
        # anything (the eval script only creates env+rollout).
        # Build Hydra overrides. `++runner.ckpt_path` is only emitted when
        # EVAL_CKPT is non-empty — otherwise rollout worker's torch.load()
        # would receive an empty string. Empty EVAL_CKPT means "base-model
        # eval": use rollout.model.model_path (base HF snapshot) as-is.
        CKPT_ARG=()
        if [ -n "${EVAL_CKPT:-}" ]; then
            CKPT_ARG+=("++runner.ckpt_path=${EVAL_CKPT}")
        fi

        # RLinf placement values must be resource-rank RANGES (e.g. "0-7"), not
        # scalars. A scalar `env=1` is parsed as "1 process on GPU rank 1" so
        # world_size stays at 1 regardless of cluster.num_nodes, and all
        # total_num_envs pile onto a single worker (observed: OOM at
        # total_num_envs=64). "0-N" gives one worker per GPU across N+1
        # Ray nodes; actor=0-N is validator-appeasement only (eval never spawns
        # an actor worker; see rlinf/config.py:validate_cfg for the assertion).
        # Matches the shipped RLinf eval pattern in
        # examples/embodiment/config/robotwin_place_empty_cup_ppo_openvlaoft_eval.yaml.
        RANK_RANGE="0-$((TOTAL_EXPECTED - 1))"

        # Noise-level sweep Hydra override (only emitted when NOISE_LEVEL is set):
        # `++actor.rl_head_config.noise_level=<X>` overrides the yaml default (0.3) for the
        # flow_sde train-mode noise formula. Only matters when the FSx-side patch is active
        # AND EVAL_INJECT_NOISE=true, because otherwise mode=eval takes the deterministic
        # branch (x_t_std=0). When NOISE_LEVEL unset, entire override is omitted → yaml default
        # (0.3) applies → byte-identical to the default deterministic-eval behavior.
        NOISE_LEVEL_ARG=()
        if [ -n "${NOISE_LEVEL:-}" ]; then
            NOISE_LEVEL_ARG+=("++actor.rl_head_config.noise_level=${NOISE_LEVEL}")
        fi

        /isaac-sim/python.sh "${EVAL_SCRIPT}" \
            --config-path "${CONFIG_SRC}" \
            --config-name "${CONFIG_NAME}" \
            '~cluster.component_placement' \
            "+cluster.component_placement.actor=${RANK_RANGE}" \
            "+cluster.component_placement.rollout=${RANK_RANGE}" \
            "+cluster.component_placement.env=${RANK_RANGE}" \
            "cluster.num_nodes=${TOTAL_EXPECTED}" \
            "env.eval.total_num_envs=${EVAL_TOTAL_ENVS:-64}" \
            "actor.global_batch_size=${EVAL_ACTOR_GBS:-2048}" \
            "++env.eval.init_params.task_description=${TASK_DESCRIPTION:-install trocar from box}" \
            "${NOISE_LEVEL_ARG[@]}" \
            "algorithm.eval_rollout_epoch=1" \
            "++env.eval.ignore_terminations=True" \
            "++env.eval.use_fixed_reset_state_ids=True" \
            "++env.eval.max_episode_steps=256" \
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
