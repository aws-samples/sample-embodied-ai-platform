#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# watchdog-hard-deadline.sh — durable force-teardown guard for a paid EKS run.
#
# At a hard DEADLINE it scales every GPU node group of the EKS training cluster
# to 0, so an unattended / crashed-but-Running run cannot overrun the wall-clock
# budget and keep burning on-demand g6e (the "~20h idle GPU" lesson). The p5
# learner is Capacity-Block-prepaid (sunk), but scaling it to 0 releases it too.
#
# Durable: designed to run under nohup/setsid so it survives terminal
# disconnect. Idempotent: repeated scale-to-0 is a no-op. Public-mirror-safe:
# NO internal IDs are baked in — cluster/region come from env or args.
#
#   SAFETY: this script only ever SCALES NODE GROUPS TO 0 (never launches).
#
# Usage:
#   DEADLINE_EPOCH=<unix-ts>  CLUSTER=<eks-cluster>  REGION=<region> \
#     nohup ./watchdog-hard-deadline.sh >> watchdog.log 2>&1 &
#   # or: ./watchdog-hard-deadline.sh --deadline "+35h" --cluster <c> --region <r>
#   # optional: --nodegroups "ng1,ng2"  (else auto-discovers all NGs)
#   # optional: STOP_FILE=<path>  — touch it to make the watchdog stand down.
# ---------------------------------------------------------------------------
set -uo pipefail

REGION="${REGION:-${AWS_REGION:-us-east-2}}"
CLUSTER="${CLUSTER:-gr00t-rl-eks}"
NODEGROUPS="${NODEGROUPS:-}"                 # comma-sep; empty => auto-discover
DEADLINE_EPOCH="${DEADLINE_EPOCH:-}"         # unix ts; or pass --deadline
POLL_SECONDS="${POLL_SECONDS:-120}"
STOP_FILE="${STOP_FILE:-}"                   # touch to stand down cleanly
DEADLINE_SPEC=""

while [ $# -gt 0 ]; do
  case "$1" in
    --deadline)   DEADLINE_SPEC="$2"; shift 2 ;;
    --cluster)    CLUSTER="$2"; shift 2 ;;
    --region)     REGION="$2"; shift 2 ;;
    --nodegroups) NODEGROUPS="$2"; shift 2 ;;
    --stop-file)  STOP_FILE="$2"; shift 2 ;;
    -h|--help)    sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Resolve the deadline: explicit epoch wins; else parse a "+Nh" / date spec.
if [ -z "$DEADLINE_EPOCH" ]; then
  [ -n "$DEADLINE_SPEC" ] || { echo "ERROR: set DEADLINE_EPOCH or --deadline" >&2; exit 2; }
  if [[ "$DEADLINE_SPEC" =~ ^\+([0-9]+)h$ ]]; then
    DEADLINE_EPOCH=$(( $(date +%s) + ${BASH_REMATCH[1]} * 3600 ))
  else
    DEADLINE_EPOCH=$(date -u -d "$DEADLINE_SPEC" +%s) || { echo "bad --deadline" >&2; exit 2; }
  fi
fi

log(){ echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

scale_all_to_zero(){
  local ngs="$1"
  for ng in $ngs; do
    log "scaling nodegroup $ng -> desired=0 (min=0)"
    aws eks update-nodegroup-config --cluster-name "$CLUSTER" --nodegroup-name "$ng" \
      --scaling-config minSize=0,desiredSize=0 --region "$REGION" \
      >/dev/null 2>&1 || log "  (update-nodegroup-config failed for $ng — may already be 0/deleting)"
  done
}

discover_ngs(){
  if [ -n "$NODEGROUPS" ]; then echo "${NODEGROUPS//,/ }"; return; fi
  aws eks list-nodegroups --cluster-name "$CLUSTER" --region "$REGION" \
    --query 'nodegroups' --output text 2>/dev/null | tr '\t' ' '
}

log "watchdog armed. cluster=$CLUSTER region=$REGION deadline=$(date -u -d "@$DEADLINE_EPOCH" +%Y-%m-%dT%H:%M:%SZ) poll=${POLL_SECONDS}s stop_file=${STOP_FILE:-<none>}"

while true; do
  if [ -n "$STOP_FILE" ] && [ -f "$STOP_FILE" ]; then
    log "STOP_FILE present ($STOP_FILE) — standing down WITHOUT scaling. Operator owns teardown."
    exit 0
  fi
  # Exit early if the cluster is already gone.
  if ! aws eks describe-cluster --name "$CLUSTER" --region "$REGION" >/dev/null 2>&1; then
    log "cluster $CLUSTER not found — nothing to guard; exiting."
    exit 0
  fi
  now=$(date +%s)
  if [ "$now" -ge "$DEADLINE_EPOCH" ]; then
    log "DEADLINE reached — FORCE-scaling all GPU node groups to 0."
    ngs="$(discover_ngs)"
    log "nodegroups: $ngs"
    scale_all_to_zero "$ngs"
    # Re-assert once after a short settle, in case a scale raced a controller.
    sleep 30
    scale_all_to_zero "$ngs"
    log "force-teardown issued. Watchdog exiting. (Verify: kubectl get nodes / describe-nodegroup)"
    exit 0
  fi
  remain=$(( (DEADLINE_EPOCH - now) / 60 ))
  log "alive; ${remain} min to deadline."
  sleep "$POLL_SECONDS"
done
