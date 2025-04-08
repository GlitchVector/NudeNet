FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04

LABEL maintainer="NudeNet Maintainers"
LABEL description="GPU-accelerated NudeNet container with CUDA 12.8.1 on Ubuntu 20.04"

# Set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive

# Add current directory to Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.9 python3.9-dev python3-pip \
    gcc g++ make cmake git wget unzip curl \
    software-properties-common dos2unix && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    update-alternatives --set python3 /usr/bin/python3.9 && \
    python3 -m pip install --upgrade pip setuptools wheel

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

# Install PyTorch
RUN pip install torch torchvision

# Install monitoring tools and utilities
RUN pip install gpustat fastdeploy

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