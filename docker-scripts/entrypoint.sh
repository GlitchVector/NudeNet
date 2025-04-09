#!/bin/bash

# Ensure we have working file paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="/app"

echo "Starting NudeNet GPU container..."
echo "Script directory: ${SCRIPT_DIR}"

# Set model behavior environment variables if not already set
export TRY_ULTRALYTICS="${TRY_ULTRALYTICS:-1}"
echo "Model configuration: Using Ultralytics model loader (TRY_ULTRALYTICS=${TRY_ULTRALYTICS})"

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

# Check and ensure models are available
echo "Verifying required models..."
python3 "${SCRIPT_DIR}/model_manager.py" --action verify || {
    echo "Some models are missing, attempting to download..."
    python3 "${SCRIPT_DIR}/model_manager.py" --action ensure
}
python3 "${SCRIPT_DIR}/model_manager.py" --action list

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
    python3 "${SCRIPT_DIR}/test_utils.py" test --detector pytorch --model "$MODEL_PATH" --image "$IMAGE_PATH"
    
elif [ "$1" = "benchmark" ]; then
    echo "Running PyTorch benchmark test..."
    MODEL_NAME="${2:-320n}"
    MODEL_PATH="/app/models/pytorch/${MODEL_NAME}.pt"
    
    if [ ! -f "$MODEL_PATH" ]; then
        echo "Error: Model not found at $MODEL_PATH"
        exit 1
    fi
    
    # Run benchmark using test_utils.py
    python3 "${SCRIPT_DIR}/test_utils.py" benchmark --detector pytorch --model "$MODEL_PATH" --iterations 10 --warmup 1

elif [ "$1" = "check-gpu" ]; then
    echo "Checking GPU acceleration..."
    python3 "${SCRIPT_DIR}/gpu_diagnostics.py" --mode all

elif [ "$1" = "test" ]; then
    echo "Running model tests..."
    MODEL_NAME="${2:-320n}"
    MODEL_PATH="/app/models/pytorch/${MODEL_NAME}.pt"
    
    # Optional image path
    IMAGE_PATH="${3:-/app/fastdeploy_recipe/cory_chase.jpeg}"
    python3 "${SCRIPT_DIR}/test_utils.py" test --detector pytorch --model "$MODEL_PATH" --image "$IMAGE_PATH"

elif [ "$1" = "compare" ]; then
    echo "Comparing detector implementations..."
    # Optional image path
    IMAGE_PATH="${2:-/app/fastdeploy_recipe/cory_chase.jpeg}"
    python3 "${SCRIPT_DIR}/test_utils.py" compare --image "$IMAGE_PATH"

elif [ "$1" = "models" ]; then
    ACTION="${2:-list}"
    echo "Managing models: $ACTION"
    python3 "${SCRIPT_DIR}/model_manager.py" --action "$ACTION"
    
else
    echo "Usage: docker run [options] nudenet-gpu [command]"
    echo "Commands:"
    echo "  detect [model] [image] - Run detection with PyTorch model (default command)"
    echo "                            Model can be 320n (default) or 640m"
    echo "  benchmark [model]      - Run performance benchmark with PyTorch model"
    echo "  check-gpu              - Check if GPU acceleration is available"
    echo "  test [image]           - Run test on PyTorch model"
    echo "  api                    - Start API server on port 8080"
    echo "  compare [image]        - Compare different detector implementations"
    echo "  models [action]        - Manage models (list, download, verify)"
    echo "                            Actions: list, download, verify, ensure"
    echo ""
    echo "Environment variables:"
    echo "  TRY_ULTRALYTICS        - Set to 1 (default) to use Ultralytics for model loading"
    echo "                           Set to 0 to use basic PyTorch loading instead"
    echo ""
    echo "Examples:"
    echo "  docker run --gpus all nudenet-gpu"
    echo "  docker run --gpus all nudenet-gpu detect 640m /path/to/image.jpg"
    echo "  docker run --gpus all nudenet-gpu benchmark 320n"
    echo ""
    # Pass through any other command
    exec "$@"
fi