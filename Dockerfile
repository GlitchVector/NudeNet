FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

LABEL maintainer="NudeNet Maintainers"
LABEL description="Simplified GPU-accelerated NudeNet container with CUDA 11.8.0 on Ubuntu 22.04"

# Set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive

# Add current directory to Python path
ENV PYTHONPATH=/app

# Install system dependencies - only the essentials
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip \
    gcc g++ wget curl \
    dos2unix \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create symbolic links for CUDA libraries to ensure they're found
RUN ln -s /usr/local/cuda/lib64/libcudart.so /usr/lib/libcudart.so && \
    ln -s /usr/local/cuda/lib64/libcublas.so /usr/lib/libcublas.so

# Upgrade pip and install essential packages
RUN python3 -m pip install --upgrade pip setuptools wheel

# Create app directory
WORKDIR /app

# Copy the application
COPY . .

# Make scripts executable and fix line endings
RUN chmod +x /app/docker-scripts/*.py /app/docker-scripts/*.sh && \
    dos2unix /app/docker-scripts/*.sh /app/docker-scripts/*.py

# Install core dependencies (minimal)
RUN pip install "numpy<2.0.0" opencv-python-headless

# Install PyTorch with CUDA 11.8 compatibility - ensuring a version that works with our models
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install Ultralytics for proper YOLOv8 model loading
RUN pip install ultralytics

# Install only essential tools for API server
RUN pip install fastdeploy

# Install package in development mode
RUN cd /app && pip install -e .

# Create a simple test script to check imports
RUN echo '#!/usr/bin/python3\ntry:\n  import nudenet\n  print("NudeNet imported successfully")\n  print("Path:", nudenet.__file__)\nexcept Exception as e:\n  print("Error:", e)\n  exit(1)' > /app/test_import.py && \
    chmod +x /app/test_import.py && \
    python3 /app/test_import.py

# Create model directories
RUN mkdir -p /app/models/onnx /app/models/pytorch

# Copy the provided PyTorch models without conversion
COPY docker-scripts/pytorch-models/*.pt /app/models/pytorch/
RUN ls -la /app/models/pytorch/ && echo "PyTorch models copied successfully"

# Ensure models are properly set up
RUN python3 /app/docker-scripts/model_manager.py --action ensure

# Install ONNX Runtime (after PyTorch to ensure compatibility)
RUN pip install onnxruntime-gpu

# Set up environment variables for GPU
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV CUDA_VISIBLE_DEVICES=0

# Set library paths for CUDA
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Prioritize GPU execution
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV CUDA_LAUNCH_BLOCKING=1

ENTRYPOINT ["/bin/bash", "/app/docker-scripts/entrypoint.sh"]