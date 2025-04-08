FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

LABEL maintainer="NudeNet Maintainers"
LABEL description="GPU-accelerated NudeNet container with CUDA 11.8.0 on Ubuntu 22.04"

# Set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive

# Add current directory to Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip \
    gcc g++ make cmake git wget unzip curl \
    software-properties-common dos2unix && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install essential packages
RUN python3 -m pip install --upgrade pip setuptools wheel

# Create app directory
WORKDIR /app

# Copy the application
COPY . .

# Make scripts executable and fix line endings
RUN chmod +x /app/docker-scripts/*.py /app/docker-scripts/*.sh && \
    dos2unix /app/docker-scripts/*.sh /app/docker-scripts/*.py

# Install dependencies in separate layers
RUN pip install numpy opencv-python-headless

# Install ONNX Runtime with GPU support
RUN pip install onnxruntime onnxruntime-gpu

# Install PyTorch with CUDA 11.8 compatibility
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install monitoring tools and utilities
RUN pip install gpustat fastdeploy pyyaml

# Install package in development mode
RUN cd /app && pip install -e .

# Create a simple test script to check imports
RUN echo '#!/usr/bin/python3\ntry:\n  import nudenet\n  print("NudeNet imported successfully")\n  print("Path:", nudenet.__file__)\nexcept Exception as e:\n  print("Error:", e)\n  exit(1)' > /app/test_import.py && \
    chmod +x /app/test_import.py && \
    python3 /app/test_import.py

# Set up environment variables for GPU
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Set logging level to show GPU-related info
ENV ONNXRUNTIME_LOG_LEVEL=INFO

ENTRYPOINT ["/bin/bash", "/app/docker-scripts/entrypoint.sh"]