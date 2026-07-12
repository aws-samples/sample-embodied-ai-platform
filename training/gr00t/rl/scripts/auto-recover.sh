#!/bin/bash
# Auto-recovery monitor for GR00T RL training on EKS + KubeRay.
#
# Watches the head pod for two failure modes and, when either fires, patches
# the RayCluster RESUME_DIR to the latest global_step_N checkpoint and forces
# KubeRay to recreate the pods:
#
#   Trigger 1 — train process gone
#     `train_embodied_agent.py` is not in the head pod's process list for
#     three consecutive samples (~180s), even though the pod is Running.
#
#   Trigger 2 — RLinf collective desync
#     >= 5 log lines in the last 10 minutes matching the known collective
#     error pattern (Gloo peer-close, timeout, `Unsupported object type`,
#     `UnpicklingError`). This catches the case where the child collective
#     thread dies but the parent driver stays alive as a zombie.
#
# Environment variables (all optional):
#   NAMESPACE            - k8s namespace of the RayCluster (default: training)
#   RAYCLUSTER           - RayCluster resource name (default: gr00t-rl-training)
#   RESULTS_ROOT         - FSx run-dir root (default: /mnt/fsx/rl-training/results/isaaclab_ppo_gr00t_assemble_trocar_eks)
#   AWS_REGION           - AWS region for kubectl (default: us-east-2)
#   LOG                  - log file path (default: /tmp/auto-recover.log)
#   MAX_RECOVERIES       - hard cap on recovery attempts (default: 20)
#
# The script tolerates head-pod restarts during checkpoint lookup by retrying
# via any worker pod (FSx is mounted on all pods).

set -u

NAMESPACE="${NAMESPACE:-training}"
RAYCLUSTER="${RAYCLUSTER:-gr00t-rl-training}"
RESULTS_ROOT="${RESULTS_ROOT:-/mnt/fsx/rl-training/results/isaaclab_ppo_gr00t_assemble_trocar_eks}"
export AWS_REGION="${AWS_REGION:-us-east-2}"
LOG="${LOG:-/tmp/auto-recover.log}"
MAX_RECOVERIES="${MAX_RECOVERIES:-20}"

# Latest run dir discovered dynamically each iteration
CHECKPOINTS_DIR_TEMPLATE='${RESULTS_ROOT}/$(ls -1 ${RESULTS_ROOT} 2>/dev/null | grep -E "^2[0-9]{7}-" | sort | tail -1)/gr00t_assemble_trocar/checkpoints'

RECOVERY_COUNT=0
CONFIRM_EXIT=0

echo "[$(date -u)] auto-recover monitor start (ns=$NAMESPACE cluster=$RAYCLUSTER)" >> "$LOG"

while [ $RECOVERY_COUNT -lt $MAX_RECOVERIES ]; do
  HEAD=$(kubectl get pods -n "$NAMESPACE" -l ray-role=head --no-headers -o custom-columns=":metadata.name" 2>/dev/null)
  if [ -z "$HEAD" ]; then
    echo "[$(date -u)] no head pod found yet, waiting" >> "$LOG"
    sleep 30
    continue
  fi
  STATE=$(kubectl get pod "$HEAD" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null)

  # Trigger 1: train process exited
  TRAIN_PID=$(kubectl exec "$HEAD" -n "$NAMESPACE" -- bash -c "pgrep -f train_embodied_agent.py | head -1" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]+$' || true)

  # Trigger 2: RLinf collective desync patterns in the last 10 min
  COLLECTIVE_ERR=$(kubectl logs "$HEAD" -n "$NAMESPACE" --since=10m 2>/dev/null | \
    grep -cE "Connection closed by peer|Application timeout caused pair closure|Unsupported object type|UnpicklingError|Timed out waiting [0-9]+ms for recv operation" || true)
  COLLECTIVE_ERR=${COLLECTIVE_ERR:-0}

  TRIGGER=""
  if [ -z "$TRAIN_PID" ] && [ "$STATE" = "Running" ]; then
    CONFIRM_EXIT=$((CONFIRM_EXIT + 1))
    if [ $CONFIRM_EXIT -ge 3 ]; then
      TRIGGER="train_process_exited"
      CONFIRM_EXIT=0
    fi
  else
    CONFIRM_EXIT=0
  fi

  if [ -z "$TRIGGER" ] && [ "$COLLECTIVE_ERR" -ge 5 ]; then
    echo "[$(date -u)] Detected collective desync ($COLLECTIVE_ERR error lines in 10min); killing train_pid=$TRAIN_PID" >> "$LOG"
    if [ -n "$TRAIN_PID" ]; then
      kubectl exec "$HEAD" -n "$NAMESPACE" -- bash -c "kill -9 $TRAIN_PID" >> "$LOG" 2>&1 || true
    fi
    TRIGGER="collective_desync"
  fi

  if [ -n "$TRIGGER" ]; then
    echo "[$(date -u)] FAILURE DETECTED: $TRIGGER (head=$HEAD, state=$STATE, train_pid=$TRAIN_PID)" >> "$LOG"

    # Find latest checkpoint — try up to 5 times as the head pod may be restarting
    LATEST_CKPT=""
    ACCESS_POD=""
    for retry in 1 2 3 4 5; do
      H=$(kubectl get pods -n "$NAMESPACE" -l ray-role=head --no-headers -o custom-columns=":metadata.name" 2>/dev/null)
      if [ -n "$H" ]; then
        HSTATE=$(kubectl get pod "$H" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null)
        if [ "$HSTATE" = "Running" ]; then
          LATEST_CKPT=$(kubectl exec "$H" -n "$NAMESPACE" -- bash -c "CKDIR=$(eval echo $CHECKPOINTS_DIR_TEMPLATE); ls -1 \$CKDIR 2>/dev/null | grep -E '^global_step_[0-9]+\$' | sort -t_ -k3 -n | tail -1" 2>/dev/null | tr -d '\r')
          if [ -n "$LATEST_CKPT" ]; then ACCESS_POD="$H"; break; fi
        fi
      fi
      # Fallback via worker pod — FSx is mounted on all pods
      W=$(kubectl get pods -n "$NAMESPACE" -l ray-role=worker --no-headers -o custom-columns=":metadata.name" 2>/dev/null | head -1)
      if [ -n "$W" ]; then
        WSTATE=$(kubectl get pod "$W" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null)
        if [ "$WSTATE" = "Running" ]; then
          LATEST_CKPT=$(kubectl exec "$W" -n "$NAMESPACE" -- bash -c "CKDIR=$(eval echo $CHECKPOINTS_DIR_TEMPLATE); ls -1 \$CKDIR 2>/dev/null | grep -E '^global_step_[0-9]+\$' | sort -t_ -k3 -n | tail -1" 2>/dev/null | tr -d '\r')
          if [ -n "$LATEST_CKPT" ]; then ACCESS_POD="$W"; break; fi
        fi
      fi
      echo "[$(date -u)] checkpoint lookup retry $retry/5..." >> "$LOG"
      sleep 30
    done
    if [ -z "$LATEST_CKPT" ]; then
      echo "[$(date -u)] ERROR: no checkpoint found after 5 retries; will retry full loop in 60s" >> "$LOG"
      sleep 60
      continue
    fi

    CHECKPOINTS_DIR_RESOLVED=$(kubectl exec "$ACCESS_POD" -n "$NAMESPACE" -- bash -c "eval echo $CHECKPOINTS_DIR_TEMPLATE" 2>/dev/null | tr -d '\r')
    NEW_RESUME="${CHECKPOINTS_DIR_RESOLVED}/${LATEST_CKPT}"
    echo "[$(date -u)] Latest checkpoint: $LATEST_CKPT, new RESUME_DIR=$NEW_RESUME" >> "$LOG"

    # Patch RayCluster head container env RESUME_DIR in place
    kubectl get raycluster "$RAYCLUSTER" -n "$NAMESPACE" -o json | \
      NEW_RESUME="$NEW_RESUME" python3 -c "
import json, os, sys
spec = json.load(sys.stdin)
env = spec['spec']['headGroupSpec']['template']['spec']['containers'][0]['env']
for e in env:
    if e['name'] == 'RESUME_DIR':
        e['value'] = os.environ['NEW_RESUME']
        break
print(json.dumps(spec))
" > /tmp/rc-recover.json
    kubectl apply -f /tmp/rc-recover.json >> "$LOG" 2>&1

    # Delete all pods; KubeRay will recreate with the new spec
    kubectl delete pods -n "$NAMESPACE" -l "ray.io/cluster=$RAYCLUSTER" --grace-period=10 >> "$LOG" 2>&1
    echo "[$(date -u)] Pods deleted; KubeRay will recreate" >> "$LOG"

    # Wait up to 30 min for a fresh head + fresh train PID
    for i in $(seq 1 60); do
      sleep 30
      NEW_HEAD=$(kubectl get pods -n "$NAMESPACE" -l ray-role=head --no-headers -o custom-columns=":metadata.name" 2>/dev/null)
      [ -z "$NEW_HEAD" ] && continue
      NEW_STATE=$(kubectl get pod "$NEW_HEAD" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null)
      NEW_TRAIN_PID=$(kubectl exec "$NEW_HEAD" -n "$NAMESPACE" -- bash -c "pgrep -f train_embodied_agent.py | head -1" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]+$' || true)
      if [ -n "$NEW_TRAIN_PID" ] && [ "$NEW_STATE" = "Running" ]; then
        echo "[$(date -u)] Recovery complete: new head=$NEW_HEAD train_pid=$NEW_TRAIN_PID" >> "$LOG"
        break
      fi
    done
    RECOVERY_COUNT=$((RECOVERY_COUNT + 1))
  fi

  sleep 60
done

echo "[$(date -u)] auto-recover hit MAX_RECOVERIES ($MAX_RECOVERIES); stopping" >> "$LOG"
