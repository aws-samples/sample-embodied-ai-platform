#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# capacity-probe.sh — live g6e On-Demand capacity probe for a given AZ subnet.
#
# Determines — synchronously and reliably — whether On-Demand capacity for an
# instance type (default g6e.8xlarge) exists in a specific AZ subnet, by placing
# a one-time `aws ec2 create-fleet --type instant` request and reading its
# response. This is the ONLY reliable live-capacity signal:
#   - describe-instance-type-offerings reports AZ *support*, not capacity.
#   - RunInstances / CreateFleet DryRun report *permissions* only.
# create-fleet --type instant actually attempts the launch and returns
# Errors[].ErrorCode (incl. InsufficientInstanceCapacity) plus Instances[] for
# what launched. See RESEARCH Q4.
#
#   SAFETY: this script LAUNCHES REAL INSTANCES for a few seconds and then
#   SELF-TERMINATES them. A `trap ... EXIT` is armed BEFORE the launch, so an
#   interrupt (Ctrl-C / kill) still terminates any launched instances and
#   deletes the temporary launch template. Instances are also tagged so the
#   cleanup can reap them by tag even if the launch response was never captured.
#   Keep CAPACITY small (default 2) — it reliably detects a hard-zero, which is
#   the known incident, without launching a large fleet.
#
# Public-mirror-safe: NO internal subnet/AMI/VPC/CR IDs are baked in. The subnet
# is a required argument; the AMI is either passed or resolved from a PUBLIC SSM
# parameter. The verdict is emitted as a machine-readable line on stdout:
#       CAPACITY=available     (no InsufficientInstanceCapacity, instances launched)
#       CAPACITY=unavailable   (InsufficientInstanceCapacity present)
# The availability verdict is carried by the CAPACITY= line, NOT the exit code:
# a clean determination exits 0 even when capacity is a hard-zero.
#
# Usage:
#   ./capacity-probe.sh --subnet <SUBNET_ID> [--region us-east-2] \
#       [--instance-type g6e.8xlarge] [--capacity 2] [--image-id <AMI_ID>]
#   # or via env:
#   SUBNET_ID=<sid> REGION=us-east-2 CAPACITY=2 ./capacity-probe.sh
#
#   Example (Wave 2, under the paid gate — probe the FSx AZ private subnet):
#     ./capacity-probe.sh --subnet <the-us-east-2a-private-subnet-id>
#
# Required IAM on the deploy principal (existing admin role; no new grants):
#   ec2:CreateFleet, ec2:CreateLaunchTemplate, ec2:DeleteLaunchTemplate,
#   ec2:TerminateInstances, ec2:DescribeInstances, ssm:GetParameter.
# ---------------------------------------------------------------------------
set -euo pipefail

# --- Configuration (non-secret defaults; NO internal IDs) -------------------
REGION="${REGION:-${AWS_REGION:-us-east-2}}"
SUBNET_ID="${SUBNET_ID:-}"                       # REQUIRED — the AZ subnet to probe
INSTANCE_TYPE="${INSTANCE_TYPE:-g6e.8xlarge}"
CAPACITY="${CAPACITY:-2}"                         # small canary (RESEARCH open-Q1)
IMAGE_ID="${IMAGE_ID:-}"                          # optional; resolved from SSM if unset

# Public EKS-optimized NVIDIA AMI SSM parameter (no internal ID; used only to give
# the temporary launch template a valid image so the launch attempt is realistic).
SSM_AMI_PARAM="/aws/service/eks/optimized-ami/1.31/amazon-linux-2023/x86_64/nvidia/recommended/image_id"

while [ $# -gt 0 ]; do
  case "$1" in
    --subnet|--subnet-id)  SUBNET_ID="$2"; shift 2 ;;
    --region)              REGION="$2"; shift 2 ;;
    --instance-type)       INSTANCE_TYPE="$2"; shift 2 ;;
    --capacity)            CAPACITY="$2"; shift 2 ;;
    --image-id)            IMAGE_ID="$2"; shift 2 ;;
    -h|--help)             sed -n '2,42p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1 (try --help)" >&2; exit 2 ;;
  esac
done

log(){ echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >&2; }

[ -n "$SUBNET_ID" ] || { echo "ERROR: --subnet <SUBNET_ID> is required (the AZ to probe)" >&2; exit 2; }

# A unique tag so cleanup can find launched instances even if we were interrupted
# mid-launch and never captured the create-fleet response.
PROBE_TAG="capacity-probe-$(date -u +%Y%m%d%H%M%S)-$$"

# --- State the trap reads (set as the run progresses) -----------------------
LT_ID=""
INSTANCE_IDS=""

cleanup(){
  local rc=$?
  set +e
  # 1) Terminate any instance IDs we captured from the fleet response.
  local ids="$INSTANCE_IDS"
  # 2) Fallback: reap anything tagged for this probe that is still pending/running
  #    (covers an interrupt during the create-fleet call before we parsed IDs).
  local tagged
  tagged=$(aws ec2 describe-instances --region "$REGION" \
      --filters "Name=tag:CapacityProbe,Values=$PROBE_TAG" \
                "Name=instance-state-name,Values=pending,running" \
      --query 'Reservations[].Instances[].InstanceId' --output text 2>/dev/null | tr '\t' ' ')
  ids="$ids $tagged"
  # Dedupe.
  ids=$(echo "$ids" | tr ' ' '\n' | grep -v '^$' | sort -u | tr '\n' ' ')
  if [ -n "${ids// /}" ]; then
    log "cleanup: terminating probe instances: $ids"
    aws ec2 terminate-instances --region "$REGION" --instance-ids $ids >/dev/null 2>&1 \
      || log "cleanup: terminate-instances failed (may already be terminating)"
  else
    log "cleanup: no probe instances to terminate"
  fi
  # 3) Delete the temporary launch template.
  if [ -n "$LT_ID" ]; then
    log "cleanup: deleting temporary launch template $LT_ID"
    aws ec2 delete-launch-template --region "$REGION" --launch-template-id "$LT_ID" >/dev/null 2>&1 \
      || log "cleanup: delete-launch-template failed (may already be gone)"
  fi
  exit "$rc"
}
# Arm the trap BEFORE any launch so an interrupt still reaps resources.
trap cleanup EXIT INT TERM

# --- 1. Resolve the AMI (arg wins; else public SSM parameter) ---------------
if [ -z "$IMAGE_ID" ]; then
  log "resolving image id from SSM parameter $SSM_AMI_PARAM"
  IMAGE_ID=$(aws ssm get-parameter --region "$REGION" --name "$SSM_AMI_PARAM" \
      --query 'Parameter.Value' --output text)
fi
[ -n "$IMAGE_ID" ] && [ "$IMAGE_ID" != "None" ] || { echo "ERROR: could not resolve an image id" >&2; exit 3; }
log "image id: $IMAGE_ID  instance-type: $INSTANCE_TYPE  subnet: $SUBNET_ID  capacity: $CAPACITY  region: $REGION"

# --- 2. Create a TEMPORARY launch template ----------------------------------
LT_NAME="capacity-probe-lt-$$-$(date -u +%s)"
LT_DATA=$(printf '{"ImageId":"%s","InstanceType":"%s"}' "$IMAGE_ID" "$INSTANCE_TYPE")
log "creating temporary launch template $LT_NAME"
LT_ID=$(aws ec2 create-launch-template --region "$REGION" \
    --launch-template-name "$LT_NAME" \
    --launch-template-data "$LT_DATA" \
    --query 'LaunchTemplate.LaunchTemplateId' --output text)
[ -n "$LT_ID" ] && [ "$LT_ID" != "None" ] || { echo "ERROR: failed to create launch template" >&2; exit 3; }
log "launch template: $LT_ID"

# --- 3. create-fleet --type instant (the live capacity attempt) -------------
# LaunchTemplateConfigs references the temp LT; the Overrides entry pins the
# subnet (AZ) + instance type. TagSpecifications tags launched instances so the
# cleanup trap can reap them by tag even on an interrupt.
LT_CONFIGS=$(printf '[{"LaunchTemplateSpecification":{"LaunchTemplateId":"%s","Version":"$Latest"},"Overrides":[{"InstanceType":"%s","SubnetId":"%s"}]}]' \
    "$LT_ID" "$INSTANCE_TYPE" "$SUBNET_ID")
TARGET_SPEC=$(printf '{"TotalTargetCapacity":%s,"DefaultTargetCapacityType":"on-demand"}' "$CAPACITY")
TAG_SPEC=$(printf '[{"ResourceType":"instance","Tags":[{"Key":"CapacityProbe","Value":"%s"}]}]' "$PROBE_TAG")

log "placing create-fleet --type instant (this launches real instances for seconds)"
# Do NOT use DryRun — it checks permissions only and never surfaces capacity.
FLEET_JSON=$(aws ec2 create-fleet --region "$REGION" \
    --type instant \
    --launch-template-configs "$LT_CONFIGS" \
    --target-capacity-specification "$TARGET_SPEC" \
    --tag-specifications "$TAG_SPEC" \
    --output json)

# --- 4. Extract launched instance IDs (for cleanup) -------------------------
INSTANCE_IDS=$(echo "$FLEET_JSON" | jq -r '[.Instances[]?.InstanceIds[]?] | join(" ")')
log "launched instances: ${INSTANCE_IDS:-<none>}"

# --- 5. Scan Errors[] for InsufficientInstanceCapacity ----------------------
ERRORS=$(echo "$FLEET_JSON" | jq -c '.Errors // []')
ICE_COUNT=$(echo "$FLEET_JSON" | jq '[.Errors[]?.ErrorCode | select(. == "InsufficientInstanceCapacity")] | length')
LAUNCHED_COUNT=$(echo "$FLEET_JSON" | jq '[.Instances[]?.InstanceIds[]?] | length')

# Echo the raw Errors[] for the operator.
log "create-fleet Errors[]: $ERRORS"

# --- 6. Deterministic machine-readable verdict on stdout --------------------
if [ "$ICE_COUNT" -gt 0 ]; then
  echo "CAPACITY=unavailable"
  log "verdict: InsufficientInstanceCapacity present ($ICE_COUNT) for $INSTANCE_TYPE in $SUBNET_ID"
elif [ "$LAUNCHED_COUNT" -gt 0 ]; then
  echo "CAPACITY=available"
  log "verdict: $LAUNCHED_COUNT instance(s) launched for $INSTANCE_TYPE in $SUBNET_ID"
else
  # No ICE and nothing launched — some other launch error; not a capacity signal.
  echo "CAPACITY=unknown"
  log "verdict: no instances launched and no InsufficientInstanceCapacity — inspect Errors[] above"
fi

# Clean determination made; the trap now reaps instances + the temp LT. Exit 0
# so a hard-zero is not treated as a script failure (verdict is on the CAPACITY= line).
exit 0
