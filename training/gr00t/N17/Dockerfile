# Isaac-GR00T N1.7 Fine-tuning Container
# Based on upstream: https://github.com/NVIDIA/Isaac-GR00T/blob/main/Dockerfile
FROM nvcr.io/nvidia/pytorch:25.04-py3

ARG USE_STABLE=true
# https://github.com/NVIDIA/Isaac-GR00T/releases/tag/n1.7-release
ARG STABLE_COMMIT=23ace64f17aa5015259b8609d371eb61a357c776

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# System deps (N1.7 upstream + our additions: aws-cli, git-lfs, vim/htop)
RUN apt-get update && apt-get install -y \
    build-essential yasm cmake libtool git git-lfs pkg-config \
    libass-dev libfreetype6-dev libvorbis-dev \
    autoconf automake texinfo ffmpeg \
    wget curl ca-certificates unzip vim less htop \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

# AWS CLI v2 (needed for S3 integration)
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && ./aws/install && rm -rf awscliv2.zip aws

# UV package manager
RUN pip install uv

# Upstream dep fixes (from N1.7 Dockerfile)
RUN sed -i s/dill==0.3.9/dill==0.3.8/g /etc/pip/constraint.txt && \
    sed -i s/scipy==1.15.2/scipy==1.15.3/g /etc/pip/constraint.txt && \
    sed -i '/flash.attn/d' /etc/pip/constraint.txt

# N1.7 pip deps (from upstream Dockerfile)
RUN pip install imageio h5py boto3 transformers[torch] deepspeed timm peft \
    diffusers wandb tianshou dm_tree openai \
    albumentations==1.4.18 decord==0.6.0 torchcodec==0.4.0

# PyTorch3D from source (N1.7 requirement)
# MAX_JOBS=2 limits parallel C++ compilation to avoid OOM in memory-constrained build envs
RUN cd /opt && git clone https://github.com/facebookresearch/pytorch3d.git && \
    cd pytorch3d && MAX_JOBS=2 pip install .

# Clone Isaac-GR00T
RUN git clone https://github.com/NVIDIA/Isaac-GR00T.git /workspace/gr00t-repo && \
    cd /workspace/gr00t-repo && \
    if [ "${USE_STABLE}" = "true" ]; then \
        echo "Using stable commit: ${STABLE_COMMIT}"; \
        git checkout ${STABLE_COMMIT}; \
    else \
        echo "Using latest version from main branch"; \
    fi && \
    echo "GR00T version info:" && \
    git log -1 --format="%H %ai %s"

# Install gr00t package
RUN cd /workspace/gr00t-repo && uv sync && uv pip install -e .

# Activate the uv venv so all subsequent python/pip commands use it
ENV VIRTUAL_ENV="/workspace/gr00t-repo/.venv"
ENV PATH="/workspace/gr00t-repo/.venv/bin:${PATH}"

# flash-attn (required by GR00T N1.7; in pyproject.toml but needs explicit install)
# Use uv pip to bypass /etc/pip/constraint.txt which pins an older flash-attn version
RUN uv pip install flash-attn==2.7.4.post1 --no-build-isolation

# EGL/Vulkan setup (from N1.7 Dockerfile)
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
RUN mkdir -p /usr/share/vulkan/icd.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.2.140"}}' \
    > /usr/share/vulkan/icd.d/nvidia_icd.json

# Install additional utilities
RUN pip install notebook gpustat

# Copy workflow scripts + modality config
COPY finetune_gr00t.py /workspace/scripts/
COPY run_finetune_workflow.sh /workspace/scripts/
COPY so101_modality_config.py /workspace/scripts/
RUN chmod +x /workspace/scripts/run_finetune_workflow.sh

WORKDIR /workspace/gr00t-repo

ENV PYTHONPATH="/workspace/gr00t-repo:/workspace"
ENV DATASET_LOCAL_DIR="/workspace/train"
ENV OUTPUT_DIR="/workspace/checkpoints"
ENV BASE_MODEL_PATH="nvidia/GR00T-N1.7-3B"

RUN mkdir -p ${DATASET_LOCAL_DIR} ${OUTPUT_DIR}

ENTRYPOINT ["/bin/bash"]
CMD ["/workspace/scripts/run_finetune_workflow.sh"]
