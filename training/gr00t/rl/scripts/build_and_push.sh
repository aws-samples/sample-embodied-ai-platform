#!/bin/bash
# Build and push learner and rollout container images to ECR.
#
# Usage:
#   bash build_and_push.sh [--region REGION]
#
# Prerequisites:
#   - GR00TRLBatchStack deployed (creates ECR repositories)
#   - Docker installed and running
#   - AWS CLI configured with ECR push permissions

set -euo pipefail

REGION="${AWS_DEFAULT_REGION:-us-west-2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="${SCRIPT_DIR}/../docker"

while [[ $# -gt 0 ]]; do
    case $1 in
        --region) REGION="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Get ECR URIs from stack
STACK_OUTPUTS=$(aws cloudformation describe-stacks \
    --stack-name GR00TRLBatchStack \
    --region "${REGION}" \
    --query 'Stacks[0].Outputs' \
    --output json)

LEARNER_ECR=$(echo "${STACK_OUTPUTS}" | jq -r '.[] | select(.OutputKey=="LearnerECR") | .OutputValue')
ROLLOUT_ECR=$(echo "${STACK_OUTPUTS}" | jq -r '.[] | select(.OutputKey=="RolloutECR") | .OutputValue')
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "============================================"
echo "Building GR00T RL Container Images"
echo "============================================"
echo "Learner ECR:  ${LEARNER_ECR}"
echo "Rollout ECR:  ${ROLLOUT_ECR}"
echo "============================================"

# ECR login
aws ecr get-login-password --region "${REGION}" | \
    docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# Build learner image
echo ""
echo "Building learner image..."
docker build \
    -f "${DOCKER_DIR}/Dockerfile.learner" \
    -t "${LEARNER_ECR}:latest" \
    "${DOCKER_DIR}"

echo "Pushing learner image..."
docker push "${LEARNER_ECR}:latest"

# Build rollout image
echo ""
echo "Building rollout image..."
docker build \
    -f "${DOCKER_DIR}/Dockerfile.rollout" \
    -t "${ROLLOUT_ECR}:latest" \
    "${DOCKER_DIR}"

echo "Pushing rollout image..."
docker push "${ROLLOUT_ECR}:latest"

echo ""
echo "============================================"
echo "Done. Update CDK context and redeploy:"
echo ""
echo "  cd training/gr00t/rl/infra"
echo "  cdk deploy \\"
echo "    --context learner_image_uri=${LEARNER_ECR}:latest \\"
echo "    --context rollout_image_uri=${ROLLOUT_ECR}:latest"
echo "============================================"
