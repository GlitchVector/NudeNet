#!/bin/bash

# This script performs a diagnostic check of the container environment
# Useful for debugging container startup issues

echo "===== CONTAINER ENVIRONMENT DIAGNOSTIC ====="
echo

echo "=== System Information ==="
uname -a
echo

echo "=== Bash Information ==="
bash --version | head -n 1
echo

echo "=== File System Checks ==="
echo "Script directory structure:"
ls -la /app/docker-scripts/
echo

echo "Checking script permissions:"
find /app/docker-scripts -type f -name "*.sh" -exec ls -l {} \;
echo

echo "Checking script line endings:"
file /app/docker-scripts/*.sh
echo

echo "=== Python Environment ==="
python3 --version
pip list | grep -E "onnxruntime|torch|numpy|opencv"
echo

echo "=== Model Directory Structure ==="
find /app/models -type f -name "*.onnx" -o -name "*.pt" 2>/dev/null | sort
echo

echo "=== NVIDIA Information ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not found or not accessible"
fi
echo

echo "=== CUDA Information ==="
if python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"; then
    echo "PyTorch CUDA check succeeded"
else
    echo "PyTorch CUDA check failed"
fi
echo

echo "=== Container Environment Variables ==="
env | grep -E "NVIDIA|CUDA|ONNX"
echo

echo "===== DIAGNOSTIC COMPLETE ====="