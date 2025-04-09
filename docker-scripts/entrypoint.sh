#!/bin/bash

# Ensure we have working file paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="/app"

echo "Starting NudeNet GPU container..."
echo "Script directory: ${SCRIPT_DIR}"

# Add current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$APP_DIR

# Verify Python can find the nudenet module
echo "Checking NudeNet module availability..."
if python3 -c "import nudenet; print('✅ NudeNet module found at:', nudenet.__file__)" 2>/dev/null; then
    echo "Module check passed"
else
    echo "❌ ERROR: Cannot import nudenet module"
    echo "Running diagnostics..."
    python3 -c "import sys; print(sys.path)"
    # Try to fix the import issue
    echo "Attempting to fix import issue..."
    python3 "${SCRIPT_DIR}/fix_import.py"
fi

# Check for PyTorch models - these should be pre-installed in the image
if [ ! -f "/app/models/pytorch/320n.pt" ]; then
    echo "ERROR: PyTorch models not found in image"
    ls -la /app/docker-scripts/pytorch-models/
    ls -la /app/models/pytorch/
    exit 1
else
    echo "Using pre-installed PyTorch models from image"
    ls -la /app/models/pytorch/
fi

# Check if CUDA is available with PyTorch
echo "Checking CUDA availability with PyTorch..."
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

if [ "$1" = "api" ]; then
    echo "Starting API server..."
    exec python3 -m fastdeploy --recipe /app/fastdeploy_recipe --mode rest
    
elif [ "$1" = "detect" ] || [ -z "$1" ]; then
    # Default action: Run detection with PyTorch model
    MODEL_NAME="${2:-320n}"
    MODEL_PATH="/app/models/pytorch/${MODEL_NAME}.pt"
    
    if [ ! -f "$MODEL_PATH" ]; then
        echo "Error: Model not found at $MODEL_PATH"
        echo "Available models:"
        ls -la /app/models/pytorch/
        exit 1
    fi
    
    # Optional image path
    IMAGE_PATH="${3:-/app/fastdeploy_recipe/cory_chase.jpeg}"
    RESOLUTION=320
    if [[ "$MODEL_NAME" == *"640"* ]]; then
        RESOLUTION=640
    fi
    
    # Run PyTorch detector
    echo "Running PyTorch detector with model: $MODEL_PATH"
    python3 "${SCRIPT_DIR}/simple_pytorch_detector.py" --model "$MODEL_PATH" --image "$IMAGE_PATH" --resolution $RESOLUTION
    
elif [ "$1" = "benchmark" ]; then
    echo "Running PyTorch benchmark test..."
    MODEL_NAME="${2:-320n}"
    MODEL_PATH="/app/models/pytorch/${MODEL_NAME}.pt"
    
    if [ ! -f "$MODEL_PATH" ]; then
        echo "Error: Model not found at $MODEL_PATH"
        exit 1
    fi
    
    # Simple benchmark script
    python3 -c "
import torch
import time
import sys
import os
from pathlib import Path
sys.path.append('${SCRIPT_DIR}')
from simple_pytorch_detector import SimpleYOLODetector

model_path = '${MODEL_PATH}'
print(f'Benchmarking PyTorch model: {model_path}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load model
start_time = time.time()
detector = SimpleYOLODetector(model_path, device)
load_time = time.time() - start_time
print(f'Model load time: {load_time:.4f} seconds')

# Test image
image_path = '/app/fastdeploy_recipe/cory_chase.jpeg'
if not os.path.exists(image_path):
    print(f'Error: Test image not found at {image_path}')
    sys.exit(1)

# Run detection 10 times
times = []
for i in range(10):
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    detections = detector.detect(image_path)
    torch.cuda.synchronize() if device == 'cuda' else None
    end = time.time()
    times.append(end - start)
    print(f'Run {i+1}: {(end-start)*1000:.2f} ms, {len(detections)} detections')

avg_time = sum(times) / len(times)
print(f'\\nAverage detection time: {avg_time*1000:.2f} ms')
print(f'Average FPS: {1/avg_time:.2f}')
    "

elif [ "$1" = "check-gpu" ]; then
    echo "Checking GPU acceleration with PyTorch..."
    python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('Device count:', torch.cuda.device_count())
    print('Current device:', torch.cuda.current_device())
    print('Device name:', torch.cuda.get_device_name(0))
    print('Device capability:', torch.cuda.get_device_capability())
    # Test tensor creation on GPU
    try:
        x = torch.rand(5, 5).cuda()
        y = torch.rand(5, 5).cuda()
        z = x + y
        print('GPU tensor test: Success')
    except Exception as e:
        print('GPU tensor test failed:', e)
else:
    print('CUDA not available. Check NVIDIA drivers and CUDA installation.')
    "

elif [ "$1" = "test" ]; then
    echo "Running PyTorch model test..."
    MODEL_NAME="${2:-320n}"
    MODEL_PATH="/app/models/pytorch/${MODEL_NAME}.pt"
    
    # Optional image path
    IMAGE_PATH="${3:-/app/fastdeploy_recipe/cory_chase.jpeg}"
    python3 "${SCRIPT_DIR}/test_models.py" "$IMAGE_PATH"
    
else
    echo "Usage: docker run [options] nudenet-gpu [command]"
    echo "Commands:"
    echo "  detect [model] [image] - Run detection with PyTorch model (default command)"
    echo "                            Model can be 320n (default) or 640m"
    echo "  benchmark [model]      - Run performance benchmark with PyTorch model"
    echo "  check-gpu              - Check if GPU acceleration is available"
    echo "  test [image]           - Run test on PyTorch model"
    echo "  api                    - Start API server on port 8080"
    echo ""
    echo "Examples:"
    echo "  docker run --gpus all nudenet-gpu"
    echo "  docker run --gpus all nudenet-gpu detect 640m /path/to/image.jpg"
    echo "  docker run --gpus all nudenet-gpu benchmark 320n"
    echo ""
    # Pass through any other command
    exec "$@"
fi