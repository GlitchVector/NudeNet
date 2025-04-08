#!/bin/bash

# Download models if needed
if [ ! -d "/app/models" ] || [ -z "$(ls -A /app/models/onnx 2>/dev/null)" ]; then
    echo "First run detected: Downloading all model variants..."
    python3 /app/docker-scripts/download_models.py
fi

if [ "$1" = "api" ]; then
    echo "Starting API server..."
    exec python3 -m fastdeploy --recipe /app/fastdeploy_recipe --mode rest
elif [ "$1" = "check-gpu" ]; then
    echo "Running comprehensive GPU check..."
    python3 /app/docker-scripts/gpu_check.py
elif [ "$1" = "benchmark" ]; then
    echo "Running GPU benchmark test..."
    if [ "$2" = "pytorch" ]; then
        python3 /app/docker-scripts/benchmark.py --pytorch-only
    elif [ "$2" = "onnx" ]; then
        python3 /app/docker-scripts/benchmark.py --onnx-only
    elif [ "$2" = "640" ] || [ "$2" = "640m" ]; then
        python3 /app/docker-scripts/benchmark.py --model-640
    elif [ "$2" = "compare" ]; then
        python3 /app/docker-scripts/benchmark.py --pytorch
    else
        python3 /app/docker-scripts/benchmark.py
    fi
elif [ "$1" = "download-models" ]; then
    echo "Downloading all model variants..."
    python3 /app/docker-scripts/download_models.py
elif [ "$1" = "pytorch" ]; then
    # Additional argument is the model name
    if [ -z "$2" ]; then
        echo "Error: Please specify a model variant (320n or 640m)"
        exit 1
    fi
    
    MODEL_PATH="/app/models/pytorch/$2.pt"
    if [ ! -f "$MODEL_PATH" ]; then
        echo "Error: Model not found at $MODEL_PATH"
        echo "Available models:"
        ls -la /app/models/pytorch/
        exit 1
    fi
    
    # Optional image path
    IMAGE_PATH="${3:-/app/fastdeploy_recipe/cory_chase.jpeg}"
    RESOLUTION=320
    if [ "$2" = "640m" ]; then
        RESOLUTION=640
    fi
    
    # Run PyTorch detector
    echo "Running PyTorch detector with model: $MODEL_PATH"
    python3 /app/docker-scripts/pytorch_detector.py --model "$MODEL_PATH" --image "$IMAGE_PATH" --resolution $RESOLUTION
elif [ "$1" = "test" ]; then
    echo "Running model tests..."
    if [ -n "$2" ]; then
        python3 /app/docker-scripts/test_models.py "$2"
    else
        python3 /app/docker-scripts/test_models.py
    fi
else
    echo "Usage: docker run [options] nudenet-gpu [command]"
    echo "Commands:"
    echo "  api             - Start API server on port 8080"
    echo "  check-gpu       - Check if GPU acceleration is available"
    echo "  benchmark       - Run performance benchmark"
    echo "  download-models - Force re-download all model variants"
    echo "  pytorch [model] - Run with PyTorch model (320n or 640m)"
    echo "  test [image]    - Run test on both ONNX and PyTorch models"
    echo "  bash            - Start a bash shell"
    if [ -z "$1" ]; then
        # Default command
        echo "Default: Checking basic GPU acceleration availability"
        exec python3 -c "from nudenet import NudeDetector; print(\"Available providers:\", NudeDetector().onnx_session.get_providers()); print(\"Using GPU acceleration:\", \"CUDAExecutionProvider\" in NudeDetector().onnx_session.get_providers())"
    else
        # Pass through any other command
        exec "$@"
    fi
fi