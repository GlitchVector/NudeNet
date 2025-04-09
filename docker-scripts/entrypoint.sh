#!/bin/bash
set -e

# Function to show usage
show_usage() {
  echo "NudeNet Image Detector"
  echo "Usage:"
  echo "  * Process images: docker run --gpus all -v /path/to/images:/images nudenet-gpu /images/image1.jpg [/images/image2.jpg ...]"
  echo "  * Run GPU test:   docker run --gpus all nudenet-gpu --test-gpu"
  echo "Results will be printed to stdout"
}

# Check for GPU test flag
if [ "$1" = "--test-gpu" ]; then
  echo "Running NudeNet GPU Test..."
  python3 /app/test_gpu.py
  exit $?
fi

# If no arguments provided, show usage
if [ $# -eq 0 ]; then
  show_usage
  exit 0
fi

# Process each image file provided as argument
for img_path in "$@"; do
  echo "Processing: $img_path"
  python3 -c "
from nudenet import NudeDetector
import sys
import json

try:
    detector = NudeDetector()
    results = detector.detect('$img_path')
    print(json.dumps(results, indent=2))
except Exception as e:
    print(f'Error processing {img_path}: {str(e)}', file=sys.stderr)
"
done