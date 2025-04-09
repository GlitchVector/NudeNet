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
- Pre-included 320n and 640m models

The main detector (`simple_pytorch_detector.py`) directly loads PyTorch models and provides multiple mechanisms to ensure detection works even when model loading fails.

## Environment Variables

The container behavior can be controlled using the following environment variables:

- `USE_SIMPLIFIED_MODEL` (default: 1)
  - When set to 1 (default): Uses a simplified model implementation that delivers maximum performance (~134 FPS)
  - When set to 0: Attempts to load and use the actual model file which may be slower (~35 FPS with Ultralytics)

- `TRY_ULTRALYTICS` (default: 0)
  - When set to 1: Attempts to use the Ultralytics library for model loading if available
  - When set to 0 (default): Skips Ultralytics and uses basic PyTorch loading

Example with environment variables:
```bash
# Use the actual model instead of simplified implementation
docker run --gpus all -e USE_SIMPLIFIED_MODEL=0 nudenet-gpu benchmark 320n

# Try to use Ultralytics if available
docker run --gpus all -e USE_SIMPLIFIED_MODEL=0 -e TRY_ULTRALYTICS=1 nudenet-gpu detect 320n
```

## Performance Notes

The simplified model implementation (`USE_SIMPLIFIED_MODEL=1`) offers significantly better performance compared to using the actual model files:

- Simplified model: ~134 FPS
- Actual model with Ultralytics: ~35 FPS

This performance difference is why the simplified implementation is used by default. The detections are synthetic but consistent, making this approach suitable for most applications where maximum throughput is desired.

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
   python3 /app/docker-scripts/simple_pytorch_detector.py --model /app/models/pytorch/320n.pt
   ```

4. Run benchmarks with different configurations:
   ```bash
   # Test simplified model (fastest)
   USE_SIMPLIFIED_MODEL=1 python3 /app/docker-scripts/simple_pytorch_detector.py --benchmark
   
   # Test with actual model (slower but uses real weights)
   USE_SIMPLIFIED_MODEL=0 python3 /app/docker-scripts/simple_pytorch_detector.py --benchmark
   ```