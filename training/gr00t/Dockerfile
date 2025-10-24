# Alternative Dockerfile from the GR00T repo: https://github.com/NVIDIA/Isaac-GR00T/blob/main/Dockerfile
# Optimized Dockerfile for Isaac-GR00T using UV package manager
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Setting the frontend to be non-interactive
# This is to avoid any user input required during the installation of packages
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies - consolidated for better layer caching
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core utilities
    wget curl ca-certificates \
    # Git and version control
    git git-lfs \
    # Build essentials
    build-essential cmake \
    # Media processing
    ffmpeg \
    # OpenCV dependencies
    libopencv-dev libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    # Python development
    python3.10 python3.10-dev python3.10-distutils python3-pip \
    # Utilities for debugging
    vim less htop \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install UV package manager (much faster than pip/conda)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Configure UV to use system Python
ENV UV_SYSTEM_PYTHON=1
ENV PYTHONPATH=/workspace

# ============================================================================
# Clone Isaac-GR00T repository
# This layer changes when using --latest
# ============================================================================
# Build argument to control GR00T version (stable vs latest)
ARG USE_STABLE=true
ARG STABLE_COMMIT=db107f03d165060998df166292578f1d7fb3c79a

RUN git clone https://github.com/NVIDIA/Isaac-GR00T.git /workspace && \
    cd /workspace && \
    if [ "${USE_STABLE}" = "true" ]; then \
    echo "Using stable commit: ${STABLE_COMMIT} (default)"; \
    git checkout ${STABLE_COMMIT}; \
    else \
    echo "Using latest version from main branch"; \
    fi && \
    echo "GR00T version info:" && \
    git log -1 --format="%H %ai %s"

# Set working directory
WORKDIR /workspace

# Upgrade pip and setuptools using UV
RUN uv pip install --upgrade pip setuptools wheel

# Install GR00T base dependencies using UV (faster resolution and installation)
RUN uv pip install --no-cache -e .[base]

# Install flash-attention separately (requires build isolation disabled)
RUN pip install --no-build-isolation flash-attn==2.7.1.post4

# Install additional utilities
RUN uv pip install --no-cache notebook gpustat wandb

# Install HuggingFace CLI and additional dependencies if necessary
# RUN pip install huggingface_hub[cli] datasets

# Copy the workflow scripts
COPY finetune_gr00t.py /workspace/scripts/
COPY run_finetune_workflow.sh /workspace/scripts/
RUN chmod +x /workspace/scripts/run_finetune_workflow.sh

# Set environment variables with defaults
ENV DATASET_LOCAL_DIR="/workspace/train"
ENV OUTPUT_DIR="/workspace/checkpoints"

# If there is issue with the latest version, use tested checkpoint as of 09 July 2025 by commenting out the following two lines
# RUN hf download nvidia/GR00T-N1.5-3B --revision 869830fc749c35f34771aa5209f923ac57e4564e --local-dir ./GR00T-N1.5-3B
# ENV BASE_MODEL_PATH="./GR00T-N1.5-3B"

# Create directories using environment variables
RUN mkdir -p ${DATASET_LOCAL_DIR} ${OUTPUT_DIR}

# Setting the Entrypoint and Command to /bin/bash whenever executed
ENTRYPOINT ["/bin/bash"]
# Default command to run the workflow, but can be overridden
CMD ["/workspace/scripts/run_finetune_workflow.sh"]
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
