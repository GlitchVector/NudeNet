# Stage 1: Base image with CUDA and dependencies
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

LABEL maintainer="NudeNet Maintainers"
LABEL description="Base image for GPU-accelerated NudeNet"

# Set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive

# Set up environment variables for GPU and paths
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    CUDA_LAUNCH_BLOCKING=1 \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    coreutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/local/cuda/lib64/libcudart.so /usr/lib/libcudart.so \
    && ln -s /usr/local/cuda/lib64/libcublas.so /usr/lib/libcublas.so \
    && ln -s /usr/local/cuda/lib64/libnvrtc.so /usr/lib/libnvrtc.so \
    && python3 -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip3 install --no-cache-dir "numpy<2.0.0" opencv-python-headless && \
    pip3 install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir ultralytics onnxruntime-gpu && \
    pip3 install --no-cache-dir psutil

# Stage 2: Core NudeNet with models
FROM base AS nudenet-core

# Create directory structure
RUN mkdir -p /app/nudenet /app/models

# Copy nudenet core files and models
COPY ./nudenet/__init__.py /app/nudenet/
COPY ./nudenet/nudenet.py /app/nudenet/
COPY ./nudenet/320n.onnx /app/nudenet/
COPY ./docker-scripts/models/320n.pt /app/models/
COPY ./docker-scripts/models/640m.pt /app/models/

# Stage 3: Final image with scripts (fastest to rebuild)
FROM nudenet-core AS final

# Create directory for scripts
RUN mkdir -p /app/docker-scripts

# Copy test image
COPY ./docker-scripts/cory_chase.jpeg /app/docker-scripts/cory_chase.jpeg

# Copy scripts that change frequently
COPY ./docker-scripts/test_gpu.py /app/test_gpu.py
COPY ./docker-scripts/batch_processor.py /app/batch_processor.py
COPY ./docker-scripts/entrypoint.sh /app/entrypoint.sh

# Fix line endings and make scripts executable
RUN sed -i 's/\r$//' /app/entrypoint.sh /app/test_gpu.py /app/batch_processor.py && \
    chmod +x /app/entrypoint.sh /app/test_gpu.py /app/batch_processor.py

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]