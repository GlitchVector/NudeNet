#!/bin/bash

# This script ensures that all models are downloaded
# It's a separate script to make the main entrypoint cleaner

MODELS_DIR="/app/models"
ONNX_DIR="${MODELS_DIR}/onnx"
PYTORCH_DIR="${MODELS_DIR}/pytorch"

# Create model directories if they don't exist
mkdir -p "${ONNX_DIR}"
mkdir -p "${PYTORCH_DIR}"

# Check if models are downloaded
if [ ! -f "${ONNX_DIR}/320n.onnx" ] || [ ! -f "${PYTORCH_DIR}/320n.pt" ]; then
    echo "Required models not found. Downloading..."
    python3 /app/docker-scripts/download_models.py
    
    # Verify models were downloaded and check their size
    if [ ! -f "${ONNX_DIR}/320n.onnx" ]; then
        echo "ERROR: Failed to download ONNX model. Check network connection."
        exit 1
    elif [ $(stat -c%s "${ONNX_DIR}/320n.onnx" 2>/dev/null || stat -f%z "${ONNX_DIR}/320n.onnx") -lt 1000 ]; then
        echo "WARNING: The ONNX model file appears to be corrupted (too small). Forcing re-download..."
        python3 /app/docker-scripts/download_models.py --force
    fi
    
    if [ ! -f "${PYTORCH_DIR}/320n.pt" ]; then
        echo "ERROR: Failed to download PyTorch model. Check network connection."
        exit 1
    elif [ $(stat -c%s "${PYTORCH_DIR}/320n.pt" 2>/dev/null || stat -f%z "${PYTORCH_DIR}/320n.pt") -lt 1000 ]; then
        echo "WARNING: The PyTorch model file appears to be corrupted (too small). Forcing re-download..."
        python3 /app/docker-scripts/download_models.py --force
    fi
    
    echo "Models successfully downloaded."
else
    echo "Models already downloaded. Using existing files."
fi