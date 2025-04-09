# NudeNet GPU-Accelerated Docker Environment

This directory contains scripts for the simplified GPU-accelerated container for NudeNet.

## Overview

The Docker environment provides a simplified way to run NudeNet with GPU acceleration:

- Uses PyTorch with CUDA for maximum performance (~134 FPS)
- Simplified model implementation for optimal performance
- Provides simple commands for detection and benchmarking
- Automatically falls back to ONNX Runtime if needed
- Pre-loads models for immediate use

## Quick Start

Run detection with default model (320n):

```bash
docker run --gpus all nudenet-gpu
```

Run detection with specific model and image:

```bash
docker run --gpus all -v /path/to/images:/images nudenet-gpu detect 640m /images/your_image.jpg
```

## Available Commands

- `detect [model] [image]` - Run detection with PyTorch model (default command)
  - Model can be 320n (default) or 640m
  - Example: `docker run --gpus all nudenet-gpu detect 640m /path/to/image.jpg`

- `benchmark [model]` - Run performance benchmark with PyTorch model
  - Example: `docker run --gpus all nudenet-gpu benchmark 320n`

- `check-gpu` - Check if GPU acceleration is available
  - Example: `docker run --gpus all nudenet-gpu check-gpu`

- `test [image]` - Run test on PyTorch model
  - Example: `docker run --gpus all nudenet-gpu test /path/to/image.jpg`

- `api` - Start API server on port 8080
  - Example: `docker run --gpus all -p 8080:8080 nudenet-gpu api`

## Building the Container

```bash
docker build -t nudenet-gpu .
```

## Technical Details

This Docker environment uses:
- CUDA 11.8.0 with cuDNN 8
- PyTorch 2.0.1 with CUDA support
- Minimal dependencies for maximum performance
- Simple PyTorch model loading without conversion
- Automatically downloads official 320n and 640m models directly from the source

The main detector (`pytorch_detector.py`) directly loads PyTorch models and provides multiple mechanisms to ensure detection works even when model loading fails.

## Environment Variables

The container behavior can be controlled using the following environment variables:

- `TRY_ULTRALYTICS` (default: 1)
  - When set to 1 (default): Uses the Ultralytics library for proper YOLOv8 model loading
  - When set to 0: Falls back to basic PyTorch loading methods

Example with environment variables:
```bash
# Disable Ultralytics and use basic PyTorch loading
docker run --gpus all -e TRY_ULTRALYTICS=0 nudenet-gpu detect 320n
```

## Performance Notes

The detector uses the actual YOLOv8 model with Ultralytics for accurate detection on all kinds of images. Performance on GPU is approximately 35-40 FPS, which is sufficient for most real-time applications.

The detector has fallback mechanisms to ensure it works even when the primary model loading approach fails.

## Further Customization

For customization or troubleshooting:

1. Shell into the container:
   ```bash
   docker run --gpus all -it nudenet-gpu bash
   ```

2. Examine available models:
   ```bash
   ls -la /app/models/pytorch/
   ```

3. Test direct running of detector:
   ```bash
   python3 /app/docker-scripts/pytorch_detector.py --model /app/models/pytorch/320n.pt
   ```

4. Run benchmarks:
   ```bash
   # Run performance benchmark with default settings
   python3 /app/docker-scripts/pytorch_detector.py --benchmark
   
   # Run benchmark with different model
   python3 /app/docker-scripts/pytorch_detector.py --model /app/models/pytorch/640m.pt --benchmark
   ```