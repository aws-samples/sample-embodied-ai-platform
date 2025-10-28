#!/bin/bash

# Fine-tuning Workflow Entrypoint Script for Input/Output Processing
# This script is the entry point for the Docker container and runs the complete workflow:
# - Validates required environment variables and 3rd party authentication
# - Validates EFS mount and prepares output directories
# - Downloads dataset based on source selection
# - Executes the Python workflow that: validates dataset and trains model
# - Uploads model to S3 or Hugging Face or skip
# - Cleans up temporary local artifacts
#
# Dataset source selection (by env var priority):
# 1) DATASET_LOCAL_DIR (use dataset bundled or pre-mounted)
# 2) DATASET_S3_URI (sync from S3 URI s3://bucket/path)
# 3) HF_DATASET_ID (download from Hugging Face Datasets)
# 4) Sample dataset (git clone with Git LFS): /workspace/sample-embodied-ai-platform/training/sample_dataset

set -e  # Exit on any error

echo "=========================================="
echo "Fine-tuning Workflow Starting"
echo "=========================================="

# Print environment variables for debugging
echo "Environment Configuration:"
echo "DATASET_LOCAL_DIR: ${DATASET_LOCAL_DIR}"
echo "DATASET_S3_URI: ${DATASET_S3_URI}"
echo "HF_DATASET_ID: ${HF_DATASET_ID}"
echo "HF_MODEL_REPO_ID: ${HF_MODEL_REPO_ID}"
echo "UPLOAD_TARGET: ${UPLOAD_TARGET}"
echo "S3_UPLOAD_URI: ${S3_UPLOAD_URI}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "MAX_STEPS: ${MAX_STEPS}"
echo "SAVE_STEPS: ${SAVE_STEPS}"
echo "NUM_GPUS: ${NUM_GPUS}"
echo "DATA_CONFIG: ${DATA_CONFIG}"
echo "VIDEO_BACKEND: ${VIDEO_BACKEND}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "LEARNING_RATE: ${LEARNING_RATE}"
echo "BASE_MODEL_PATH: ${BASE_MODEL_PATH}"
echo "EMBODIMENT_TAG: ${EMBODIMENT_TAG}"
echo "REPORT_TO: ${REPORT_TO}"
echo ""
echo "Training Configuration:"
echo "TUNE_LLM: ${TUNE_LLM}"
echo "TUNE_VISUAL: ${TUNE_VISUAL}"
echo "TUNE_PROJECTOR: ${TUNE_PROJECTOR}"
echo "TUNE_DIFFUSION_MODEL: ${TUNE_DIFFUSION_MODEL}"
echo "LORA_RANK: ${LORA_RANK}"
echo "LORA_ALPHA: ${LORA_ALPHA}"
echo "LORA_DROPOUT: ${LORA_DROPOUT}"
echo "LORA_FULL_MODEL: ${LORA_FULL_MODEL}"
echo "WEIGHT_DECAY: ${WEIGHT_DECAY}"
echo "WARMUP_RATIO: ${WARMUP_RATIO}"
echo "DATALOADER_NUM_WORKERS: ${DATALOADER_NUM_WORKERS}"
echo "DATALOADER_PREFETCH_FACTOR: ${DATALOADER_PREFETCH_FACTOR}"
echo ""
echo "Dataset Configuration:"
echo "BALANCE_DATASET_WEIGHTS: ${BALANCE_DATASET_WEIGHTS}"
echo "BALANCE_TRAJECTORY_WEIGHTS: ${BALANCE_TRAJECTORY_WEIGHTS}"
echo ""
echo "Workflow Configuration:"
echo "RESUME: ${RESUME}"
echo "CLEANUP_DATASET: ${CLEANUP_DATASET}"
echo "CLEANUP_CHECKPOINTS: ${CLEANUP_CHECKPOINTS}"
echo "=========================================="

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi
    echo "=========================================="
else
    echo "WARNING: nvidia-smi not found. GPU may not be available."
fi

# If REPORT_TO is 'wandb', ensure WANDB_API_KEY is set
if [ "$REPORT_TO" = "wandb" ]; then
    if [ -z "$WANDB_API_KEY" ]; then
        echo "ERROR: WANDB_API_KEY environment variable is required when REPORT_TO=wandb"
        exit 1
    fi
fi

# Validate upload target requirements (HF creds are only required when uploading to HF)
UPLOAD_TARGET_LOWER=$(echo "${UPLOAD_TARGET:-none}" | tr '[:upper:]' '[:lower:]')
case "$UPLOAD_TARGET_LOWER" in
  hf|huggingface)
    if [ -z "$HF_TOKEN" ]; then
        echo "ERROR: HF_TOKEN is required when UPLOAD_TARGET=hf"
        exit 1
    fi
    if [ -z "$HF_MODEL_REPO_ID" ]; then
        echo "ERROR: HF_MODEL_REPO_ID is required when UPLOAD_TARGET=hf"
        exit 1
    fi
    ;;
  s3)
    if [ -z "$S3_UPLOAD_URI" ]; then
        echo "ERROR: S3_UPLOAD_URI is required when UPLOAD_TARGET=s3"
        exit 1
    fi
    ;;
  none|skip|"")
    echo "Upload target set to 'none'. Model upload will be skipped."
    ;;
  *)
    echo "WARNING: Unknown UPLOAD_TARGET='$UPLOAD_TARGET'. Supported: hf|s3|none. Defaulting to 'hf'."
    UPLOAD_TARGET_LOWER=hf
    if [ -z "$HF_TOKEN" ] || [ -z "$HF_MODEL_REPO_ID" ]; then
        echo "ERROR: HF_TOKEN and HF_MODEL_REPO_ID are required for default UPLOAD_TARGET=hf"
        exit 1
    fi
    ;;
esac

# Authenticate to Hugging Face if HF resources are specified and token is provided
if [ -n "$HF_TOKEN" ] && { [ -n "$HF_DATASET_ID" ] || [ -n "$HF_MODEL_REPO_ID" ]; }; then
    echo "Authenticating to Hugging Face..."
    HF_CLI=hf
    if ! command -v "$HF_CLI" >/dev/null 2>&1; then
        if command -v huggingface-cli >/dev/null 2>&1; then
            HF_CLI=huggingface-cli
        else
            echo "ERROR: Could not find 'hf' or 'huggingface-cli'. Please ensure Hugging Face CLI is installed."
            exit 1
        fi
    fi
    
    if [ "$HF_CLI" = "hf" ]; then
        hf auth login --token "$HF_TOKEN" --non-interactive || true
    else
        huggingface-cli login --token "$HF_TOKEN" || true
    fi
fi

# Validate default EFS mount and prepare output/log directories (no env var usage)
DEFAULT_EFS_BASE="/mnt/efs"
if mountpoint -q "$DEFAULT_EFS_BASE" || grep -qs " $DEFAULT_EFS_BASE " /proc/mounts; then
    echo "EFS mount is accessible: $DEFAULT_EFS_BASE"
    export OUTPUT_DIR=${OUTPUT_DIR:-"$DEFAULT_EFS_BASE/gr00t/checkpoints"}
else
    echo "WARNING: EFS default mount not detected at $DEFAULT_EFS_BASE. Writing outputs to container-local storage."
    export OUTPUT_DIR=${OUTPUT_DIR:-"/workspace/checkpoints"}
fi
mkdir -p "$OUTPUT_DIR" || true

# Resolve dataset source according to priority and ensure accessibility
SAMPLE_REPO_DIR="/workspace/sample-embodied-ai-platform"
DEFAULT_SAMPLE_DATASET_DIR="$SAMPLE_REPO_DIR/training/sample_dataset"
RESOLVED_DATASET_DIR=""

echo "[Step] Resolve dataset source (priority: local -> s3 -> hf -> sample)"

# 1) DATASET_LOCAL_DIR
if [ -n "$DATASET_LOCAL_DIR" ] && [ -d "$DATASET_LOCAL_DIR" ] && [ -n "$(ls -A "$DATASET_LOCAL_DIR" 2>/dev/null)" ]; then
    RESOLVED_DATASET_DIR="$DATASET_LOCAL_DIR"
    echo "Using DATASET_LOCAL_DIR: $RESOLVED_DATASET_DIR"
# 2) DATASET_S3_URI
elif [ -n "$DATASET_S3_URI" ]; then
    RESOLVED_DATASET_DIR="${DATASET_LOCAL_DIR:-/workspace/train}"
    echo "DATASET_S3_URI provided. Will sync into: $RESOLVED_DATASET_DIR"
    mkdir -p "$RESOLVED_DATASET_DIR"
    if ! command -v aws >/dev/null 2>&1; then
        echo "ERROR: aws CLI is required to sync S3 dataset but was not found in PATH"
        exit 1
    fi
    echo "Syncing dataset from $DATASET_S3_URI to $RESOLVED_DATASET_DIR ..."
    aws s3 sync "$DATASET_S3_URI" "$RESOLVED_DATASET_DIR" --no-progress
    if [ -z "$(ls -A "$RESOLVED_DATASET_DIR" 2>/dev/null)" ]; then
        echo "ERROR: No files synced from S3. Please verify DATASET_S3_URI"
        exit 1
    fi
# 3) HF_DATASET_ID
elif [ -n "$HF_DATASET_ID" ]; then
    RESOLVED_DATASET_DIR="${DATASET_LOCAL_DIR:-/workspace/train}"
    echo "HF_DATASET_ID provided. Will download into: $RESOLVED_DATASET_DIR"
    mkdir -p "$RESOLVED_DATASET_DIR"
    HF_CLI=hf
    if ! command -v "$HF_CLI" >/dev/null 2>&1; then
        if command -v huggingface-cli >/dev/null 2>&1; then
            HF_CLI=huggingface-cli
        else
            echo "ERROR: Could not find 'hf' or 'huggingface-cli'. Please ensure Hugging Face CLI is installed."
            exit 1
        fi
    fi
    echo "Downloading dataset $HF_DATASET_ID to $RESOLVED_DATASET_DIR using $HF_CLI ..."
    if [ "$HF_CLI" = "hf" ]; then
        if ! hf download --repo-type dataset "$HF_DATASET_ID" --local-dir "$RESOLVED_DATASET_DIR"; then
            echo "ERROR: Failed to download Hugging Face dataset '$HF_DATASET_ID'. It may be missing, private, or your token lacks access."
            exit 1
        fi
    else
        if ! huggingface-cli download --repo-type dataset "$HF_DATASET_ID" --local-dir "$RESOLVED_DATASET_DIR"; then
            echo "ERROR: Failed to download Hugging Face dataset '$HF_DATASET_ID'. It may be missing, private, or your token lacks access."
            exit 1
        fi
    fi
    if [ -z "$(ls -A "$RESOLVED_DATASET_DIR" 2>/dev/null)" ]; then
        echo "ERROR: Dataset directory is empty after Hugging Face download: $RESOLVED_DATASET_DIR"
        exit 1
    fi
# 4) Sample dataset via git clone with Git LFS
else
    echo "No dataset source provided. Attempting to use sample dataset via Git LFS..."
    if ! command -v git >/dev/null 2>&1; then
        echo "ERROR: git is required to clone the sample dataset"
        exit 1
    fi
    if ! command -v git-lfs >/dev/null 2>&1; then
        echo "git-lfs not found. Attempting to install..."
        if command -v apt-get >/dev/null 2>&1; then
            if ! apt-get update || ! apt-get install -y git-lfs; then
                echo "ERROR: Failed to install git-lfs via apt-get."
                exit 1
            fi
        else
            echo "ERROR: git-lfs not found and automatic installation is unavailable on this system."
            exit 1
        fi
    fi
    if ! git lfs install; then
        echo "ERROR: 'git lfs install' failed. Please ensure git-lfs is configured correctly."
        exit 1
    fi
    if [ ! -d "$SAMPLE_REPO_DIR/.git" ]; then
        if ! git clone https://github.com/aws-samples/sample-embodied-ai-platform.git "$SAMPLE_REPO_DIR"; then
            echo "ERROR: Failed to clone sample repository to $SAMPLE_REPO_DIR"
            exit 1
        fi
    fi
    if [ -d "$SAMPLE_REPO_DIR" ]; then
        if ! (cd "$SAMPLE_REPO_DIR" && git lfs pull); then
            echo "ERROR: 'git lfs pull' failed in $SAMPLE_REPO_DIR. Ensure LFS files are accessible."
            exit 1
        fi
    fi
    if [ -d "$DEFAULT_SAMPLE_DATASET_DIR" ] && [ -n "$(ls -A "$DEFAULT_SAMPLE_DATASET_DIR" 2>/dev/null)" ]; then
        RESOLVED_DATASET_DIR="$DEFAULT_SAMPLE_DATASET_DIR"
        echo "Using sample dataset at: $RESOLVED_DATASET_DIR"
    else
        echo "ERROR: Failed to prepare sample dataset. Please provide DATASET_LOCAL_DIR / DATASET_S3_URI / HF_DATASET_ID"
        exit 1
    fi
fi

# Export back so Python workflow picks up the resolved dataset dir
export DATASET_LOCAL_DIR="$RESOLVED_DATASET_DIR"

# Set PYTHONPATH to ensure proper imports
export PYTHONPATH="/workspace:${PYTHONPATH}"

# Change to GR00T directory
cd /workspace

# Run the Python finetune script
echo "Starting Python finetune script..."
python scripts/finetune_gr00t.py

echo "[Step] Post-training: model upload and cleanup"

# Determine latest checkpoint directory to upload, if any
CHECKPOINT_TO_UPLOAD="$OUTPUT_DIR"
if [ -d "$OUTPUT_DIR" ]; then
    LATEST_STEP=-1
    LATEST_DIR=""
    for d in "$OUTPUT_DIR"/checkpoint-*; do
        [ -d "$d" ] || continue
        step=${d##*-}
        if [[ "$step" =~ ^[0-9]+$ ]]; then
            if [ "$step" -gt "$LATEST_STEP" ]; then
                LATEST_STEP=$step
                LATEST_DIR="$d"
            fi
        fi
    done
    if [ -n "$LATEST_DIR" ]; then
        CHECKPOINT_TO_UPLOAD="$LATEST_DIR"
        echo "Latest checkpoint detected: $CHECKPOINT_TO_UPLOAD"
    else
        echo "No numbered checkpoints found. Will upload OUTPUT_DIR if upload is enabled."
    fi
fi

case "$UPLOAD_TARGET_LOWER" in
  s3)
    echo "Uploading model artifacts to S3: $S3_UPLOAD_URI"
    if ! command -v aws >/dev/null 2>&1; then
        echo "ERROR: aws CLI is required for S3 upload"
        exit 1
    fi
    aws s3 sync "$CHECKPOINT_TO_UPLOAD" "${S3_UPLOAD_URI%/}/" --no-progress
    ;;
  hf|huggingface)
    echo "Uploading model artifacts to Hugging Face: $HF_MODEL_REPO_ID"
    HF_CLI=hf
    if ! command -v "$HF_CLI" >/dev/null 2>&1; then
        if command -v huggingface-cli >/dev/null 2>&1; then
            HF_CLI=huggingface-cli
        else
            echo "ERROR: Could not find 'hf' or 'huggingface-cli' for upload"
            exit 1
        fi
    fi
    if [ "$HF_CLI" = "hf" ]; then
        hf upload --repo-type model "$HF_MODEL_REPO_ID" "$CHECKPOINT_TO_UPLOAD"
    else
        huggingface-cli upload --repo-type model "$HF_MODEL_REPO_ID" "$CHECKPOINT_TO_UPLOAD"
    fi
    ;;
  none|skip|"")
    echo "Upload target is 'none'. Skipping upload."
    ;;
  *)
    echo "WARNING: Unknown UPLOAD_TARGET='$UPLOAD_TARGET_LOWER'. Skipping upload."
    ;;
esac

# Cleanup
if [ "${CLEANUP_DATASET}" = "true" ]; then
    echo "CLEANUP_DATASET=true set. Attempting to remove dataset directory: $DATASET_LOCAL_DIR"
    if [[ "$DATASET_LOCAL_DIR" != /workspace/sample-embodied-ai-platform/* ]]; then
        rm -rf "$DATASET_LOCAL_DIR" || true
        echo "Removed dataset directory $DATASET_LOCAL_DIR"
    else
        echo "Skipping removal of sample dataset under /workspace/sample-embodied-ai-platform"
    fi
fi

if [ "${CLEANUP_CHECKPOINTS}" = "true" ] && [ -d "$OUTPUT_DIR" ]; then
    echo "CLEANUP_CHECKPOINTS=true set. Keeping only latest checkpoint under $OUTPUT_DIR"
    for d in "$OUTPUT_DIR"/checkpoint-*; do
        [ -d "$d" ] || continue
        if [ "$d" != "$CHECKPOINT_TO_UPLOAD" ]; then
            rm -rf "$d" || true
            echo "Removed checkpoint directory $d"
        fi
    done
fi

echo "=========================================="
echo "Fine-tuning Workflow Completed Successfully!"
echo "==========================================" 