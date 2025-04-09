FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

LABEL maintainer="NudeNet Maintainers"
LABEL description="Simplified GPU-accelerated NudeNet container with CUDA 11.8.0 on Ubuntu 22.04"

# Set working directory
WORKDIR /app

# Install Python and required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for CUDA libraries to ensure they're found
RUN ln -s /usr/local/cuda/lib64/libcudart.so /usr/lib/libcudart.so && \
    ln -s /usr/local/cuda/lib64/libcublas.so /usr/lib/libcublas.so && \
    ln -s /usr/local/cuda/lib64/libnvrtc.so /usr/lib/libnvrtc.so

# Upgrade pip and install essential packages
RUN python3 -m pip install --upgrade pip setuptools wheel

# Create directory structure
RUN mkdir -p /app/nudenet /app/models /app/docker-scripts

# Copy nudenet files
COPY ./nudenet/__init__.py /app/nudenet/
COPY ./nudenet/nudenet.py /app/nudenet/
COPY ./nudenet/320n.onnx /app/nudenet/

# Copy PyTorch models
COPY ./docker-scripts/models/320n.pt /app/models/
COPY ./docker-scripts/models/640m.pt /app/models/

# Copy scripts and test image
COPY ./docker-scripts/entrypoint.sh /app/entrypoint.sh
COPY ./docker-scripts/test_gpu.py /app/test_gpu.py
COPY ./docker-scripts/cory_chase.jpeg /app/docker-scripts/cory_chase.jpeg

# Fix line endings and make scripts executable
RUN sed -i 's/\r$//' /app/entrypoint.sh /app/test_gpu.py && \
    chmod +x /app/entrypoint.sh /app/test_gpu.py

# Install dependencies with specific versions
RUN pip3 install --no-cache-dir "numpy<2.0.0" opencv-python-headless
RUN pip3 install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip3 install --no-cache-dir ultralytics onnxruntime-gpu

# Set up environment variables for GPU
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV CUDA_VISIBLE_DEVICES=0

# Set library paths for CUDA
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Prioritize GPU execution
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV CUDA_LAUNCH_BLOCKING=1

# Add Python path to include the current directory
ENV PYTHONPATH=/app

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]