#!/bin/bash
# Submit GR00T RL training job to AWS Batch MNP.
#
# Usage:
#   bash submit_training.sh [--num-nodes N] [--model-path PATH] [--num-envs N]
#
# Prerequisites:
#   - GR00TRLBatchStack deployed (cdk deploy)
#   - Learner and rollout container images pushed to ECR
#   - Model checkpoint available on EFS at /mnt/efs/models/

set -euo pipefail

# Defaults
NUM_NODES=5  # 1 learner + 4 rollouts
MODEL_PATH="/mnt/efs/models/GR00T-N1.5-RL-Rheo-AssembleTrocar"
NUM_ENVS=64
JOB_QUEUE="GR00T-RL-JobQueue"
JOB_NAME="gr00t-rl-trocar-$(date +%Y%m%d-%H%M%S)"
REGION="${AWS_DEFAULT_REGION:-us-west-2}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-nodes) NUM_NODES="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --num-envs) NUM_ENVS="$2"; shift 2 ;;
        --job-queue) JOB_QUEUE="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --num-nodes N      Total nodes: 1 learner + (N-1) rollouts (default: 5)"
            echo "  --model-path PATH  Model checkpoint path on EFS (default: /mnt/efs/models/GR00T-N1.5-RL-Rheo-AssembleTrocar)"
            echo "  --num-envs N       Parallel environments per rollout node (default: 64)"
            echo "  --job-queue NAME   Batch job queue name (default: GR00T-RL-JobQueue)"
            echo "  --region REGION    AWS region (default: us-west-2)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "Submitting GR00T RL Training Job"
echo "============================================"
echo "Job Name:     ${JOB_NAME}"
echo "Job Queue:    ${JOB_QUEUE}"
echo "Total Nodes:  ${NUM_NODES} (1 learner + $((NUM_NODES - 1)) rollouts)"
echo "Model Path:   ${MODEL_PATH}"
echo "Num Envs:     ${NUM_ENVS}"
echo "Region:       ${REGION}"
echo "============================================"

# Get job definition ARN from stack outputs
JOB_DEF_ARN=$(aws cloudformation describe-stacks \
    --stack-name GR00TRLBatchStack \
    --region "${REGION}" \
    --query 'Stacks[0].Outputs[?OutputKey==`JobDefinitionArn`].OutputValue' \
    --output text)

if [ -z "${JOB_DEF_ARN}" ] || [ "${JOB_DEF_ARN}" = "None" ]; then
    echo "ERROR: Could not find JobDefinitionArn in stack outputs."
    echo "Ensure GR00TRLBatchStack is deployed with container image URIs."
    exit 1
fi

# Submit the MNP job
# Note: numNodes is NOT included in node-overrides because the job definition uses
# closed target-node ranges (0:0, 1:4). AWS Batch only allows numNodes override when
# at least one range has an open end (e.g., "1:"). The node count is already baked
# into the job definition via CDK (num_rollout_nodes + 1).
JOB_ID=$(aws batch submit-job \
    --job-name "${JOB_NAME}" \
    --job-queue "${JOB_QUEUE}" \
    --job-definition "${JOB_DEF_ARN}" \
    --node-overrides "{
        \"nodePropertyOverrides\": [
            {
                \"targetNodes\": \"0:0\",
                \"containerOverrides\": {
                    \"environment\": [
                        {\"name\": \"MODEL_PATH\", \"value\": \"${MODEL_PATH}\"},
                        {\"name\": \"NUM_ROLLOUT_ENVS\", \"value\": \"${NUM_ENVS}\"}
                    ]
                }
            },
            {
                \"targetNodes\": \"1:$((NUM_NODES - 1))\",
                \"containerOverrides\": {
                    \"environment\": [
                        {\"name\": \"MODEL_PATH\", \"value\": \"${MODEL_PATH}\"},
                        {\"name\": \"NUM_ROLLOUT_ENVS\", \"value\": \"${NUM_ENVS}\"}
                    ]
                }
            }
        ]
    }" \
    --region "${REGION}" \
    --query 'jobId' \
    --output text)

echo ""
echo "Job submitted: ${JOB_ID}"
echo ""
echo "Monitor:"
echo "  aws batch describe-jobs --jobs ${JOB_ID} --region ${REGION} --query 'jobs[0].status'"
echo ""
echo "Logs (once RUNNING):"
echo "  aws logs tail /aws/batch/job --follow --region ${REGION}"
echo ""
echo "Cancel:"
echo "  aws batch cancel-job --job-id ${JOB_ID} --reason 'Manual cancel' --region ${REGION}"
