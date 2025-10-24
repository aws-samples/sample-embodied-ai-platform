#!/bin/bash

# Build script for GR00T Fine-tuning Docker image

set -Eeuo pipefail

echo "=========================================="
echo "Building GR00T Fine-tuning Docker Image"
echo "=========================================="

# Default values
IMAGE_NAME="gr00t-finetune"
TAG="latest"
DOCKERFILE="Dockerfile"
PUSH_IMAGE=false
TEST_IMAGE=false
USE_STABLE=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --latest)
            USE_STABLE=false
            shift
            ;;
        --push)
            PUSH_IMAGE=true
            shift
            ;;
        --test)
            TEST_IMAGE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -t, --tag TAG         Tag for the fine-tuning image (default: latest)"
            echo "  -n, --name NAME       Name for the fine-tuning image (default: gr00t-finetune)"
            echo "  --latest              Use latest GR00T from main branch (default: stable commit)"
            echo "  --push                Push image to registry after building"
            echo "  --test                Run basic tests after building"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

echo "Building image: ${FULL_IMAGE_NAME}"
echo "Using Dockerfile: ${DOCKERFILE}"

# Display GR00T version selection
if [[ "${USE_STABLE}" == "true" ]]; then
    echo "GR00T version: STABLE (tested commit from Sep 4, 2025) [default]"
else
    echo "GR00T version: LATEST (main branch, may have breaking changes)"
fi

# Build the fine-tuning image directly from the combined Dockerfile
echo "Building fine-tuning image..."
docker build \
    --build-arg USE_STABLE=${USE_STABLE} \
    -f ${DOCKERFILE} \
    -t ${FULL_IMAGE_NAME} \
    .

echo "Image built successfully: ${FULL_IMAGE_NAME}"

# Run basic tests if requested
if [[ "${TEST_IMAGE}" == "true" ]]; then
    echo "=========================================="
    echo "Running Basic Tests"
    echo "=========================================="
    
    # Test 1: Check if the image runs without errors (dry run)
    echo "Test 1: Checking if image starts correctly..."
    docker run --rm \
        -e HF_TOKEN="dummy" \
        -e HF_DATASET_ID="dummy/dummy" \
        -e HF_MODEL_REPO_ID="dummy/dummy" \
        --entrypoint /bin/bash \
        ${FULL_IMAGE_NAME} \
        -c "echo 'Image starts correctly' && python -c 'import sys; print(f\"Python version: {sys.version}\")' && which huggingface-cli"
    
    # Test 2: Check if finetune script can be imported
    echo "Test 2: Checking if finetune script imports correctly..."
    docker run --rm \
        --entrypoint /bin/bash \
        ${FULL_IMAGE_NAME} \
        -c "cd /workspace && python -c 'from scripts.finetune_gr00t import FinetuneWorkflow; print(\"Finetune script imports successfully\")'"
    
    # Test 3: Check if training script exists and can show help
    echo "Test 3: Checking if training script is accessible..."
    docker run --rm \
        --entrypoint /bin/bash \
        ${FULL_IMAGE_NAME} \
        -c "cd /workspace && python scripts/gr00t_finetune.py --help" | head -10
    
    echo "All tests passed!"
fi

# Push to registry if requested
if [[ "${PUSH_IMAGE}" == "true" ]]; then
    echo "=========================================="
    echo "Pushing to Registry"
    echo "=========================================="
    
    if [[ -z "${DOCKER_REGISTRY}" ]]; then
        echo "Warning: DOCKER_REGISTRY environment variable not set."
        echo "Assuming you want to push to Docker Hub or have already tagged appropriately."
    else
        # Re-tag with registry prefix
        REGISTRY_IMAGE="${DOCKER_REGISTRY}/${FULL_IMAGE_NAME}"
        docker tag ${FULL_IMAGE_NAME} ${REGISTRY_IMAGE}
        FULL_IMAGE_NAME=${REGISTRY_IMAGE}
    fi
    
    echo "Pushing image: ${FULL_IMAGE_NAME}"
    docker push ${FULL_IMAGE_NAME}
    echo "Image pushed successfully!"
fi

echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo "Image: ${FULL_IMAGE_NAME}"
if [[ "${USE_STABLE}" == "true" ]]; then
    echo "GR00T Version: STABLE (tested commit)"
else
    echo "GR00T Version: LATEST (main branch)"
fi
echo ""
echo "To run locally, create a local directory to simulate EFS mount:"
echo "mkdir -p ~/mock-efs/gr00t/checkpoints"
echo "Then run with a small number of steps for testing:"
echo "docker run --gpus all --network host \\"
echo "  -e MAX_STEPS=100 -e SAVE_STEPS=100 \\"
echo "  -v ~/mock-efs:/mnt/efs \\"
echo "  ${FULL_IMAGE_NAME}"
echo ""
echo "To rebuild with latest GR00T version from main branch:"
echo "  ./build_container.sh --latest" 