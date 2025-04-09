#!/bin/bash
set -e

# Function to show usage
show_usage() {
  echo "NudeNet Image Detector"
  echo "Usage:"
  echo "  * Process individual images: docker run --gpus all nudenet-gpu /path/to/image1.jpg [/path/to/image2.jpg ...]"
  echo "  * Process batch from JSON:   docker run --gpus all nudenet-gpu --batch /path/to/images.json --output /path/to/results.json"
  echo "  * Run GPU test:              docker run --gpus all nudenet-gpu --test-gpu"
  echo ""
  echo "Batch processing options:"
  echo "  --batch <json_file>          JSON file containing list of image paths"
  echo "  --output, -o <file>          Output JSON file for results"
  echo "  --batch-size, -b <num>       Number of images to process in each batch (default: 16)"
  echo "  --model, -m <file>           Path to alternative model file"
  echo "  --json-progress              Output progress information in JSON format"
  echo "  --memory-warning <percent>   Memory usage threshold for warnings (default: 85.0)"
  echo "  --memory-limit <percent>     Memory usage threshold to abort processing (default: 95.0)"
  echo ""
  echo "Note: If Docker cannot access your files, you may need to add volume mounts:"
  echo "  docker run --gpus all -v /some/path:/some/path nudenet-gpu --batch /some/path/images.json --output /some/path/results.json"
  echo ""
  echo "JSON Progress Format (when using --json-progress):"
  echo "  The script will output JSON objects with progress information."
  echo "  Each object will have a 'type' field indicating the event type:"
  echo "  - 'start': Beginning of processing with total image count"
  echo "  - 'initialized': Model initialized and ready"
  echo "  - 'progress': Regular progress updates with memory usage"
  echo "  - 'warning': Memory usage warning"
  echo "  - 'complete': Final summary with statistics"
  echo "  - 'error': Error message"
  echo ""
  echo "Results will be printed to stdout for individual images or saved to the specified output file for batch processing"
}

# Check for GPU test flag
if [ "$1" = "--test-gpu" ]; then
  echo "Running NudeNet GPU Test..."
  python3 /app/test_gpu.py
  exit $?
fi

# Function to convert Windows path to Linux path
convert_windows_path() {
  local win_path=$1
  
  # Check if it looks like a Windows path (starts with drive letter and colon)
  if [[ $win_path =~ ^[A-Za-z]: ]]; then
    # Extract drive letter, convert to lowercase, and remove colon
    local drive=$(echo "${win_path:0:1}" | tr '[:upper:]' '[:lower:]')
    # Convert backslashes to forward slashes and replace drive with /mnt/drive
    local linux_path=$(echo "${win_path:2}" | tr '\\' '/')
    echo "/mnt/$drive$linux_path"
  else
    # Not a Windows path or already a Linux path
    echo "$win_path"
  fi
}

# Check for batch processing mode
if [ "$1" = "--batch" ]; then
  if [ $# -lt 3 ]; then
    echo "Error: Batch mode requires input JSON file and output file"
    echo "Usage: docker run --gpus all nudenet-gpu --batch /path/to/images.json --output /path/to/results.json"
    exit 1
  fi
  
  # Parse arguments and convert Windows paths if needed
  INPUT_JSON=$(convert_windows_path "$2")
  
  # Look for --output flag
  OUTPUT_FILE=""
  BATCH_SIZE=16
  MODEL_PATH=""
  JSON_PROGRESS=false
  MEMORY_WARNING=85.0
  MEMORY_LIMIT=95.0
  
  shift 2  # Skip the --batch and input file arguments
  
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --output|-o)
        # Convert Windows path to Linux path if needed
        OUTPUT_FILE=$(convert_windows_path "$2")
        shift 2
        ;;
      --batch-size|-b)
        BATCH_SIZE="$2"
        shift 2
        ;;
      --model|-m)
        # Convert Windows path to Linux path if needed
        MODEL_PATH=$(convert_windows_path "$2")
        shift 2
        ;;
      --json-progress)
        JSON_PROGRESS=true
        shift
        ;;
      --memory-warning)
        MEMORY_WARNING="$2"
        shift 2
        ;;
      --memory-limit)
        MEMORY_LIMIT="$2"
        shift 2
        ;;
      *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
  done
  
  if [ -z "$OUTPUT_FILE" ]; then
    echo "Error: --output parameter is required for batch processing"
    exit 1
  fi
  
  echo "Running batch processing..."
  echo "Input JSON: $INPUT_JSON"
  echo "Output file: $OUTPUT_FILE"
  echo "Batch size: $BATCH_SIZE"
  
  # Call the batch processor script with the parsed arguments
  ARGS="--batch-size $BATCH_SIZE"
  if [ ! -z "$MODEL_PATH" ]; then
    ARGS="$ARGS --model $MODEL_PATH"
  fi
  
  if [ "$JSON_PROGRESS" = true ]; then
    ARGS="$ARGS --json-progress"
  fi
  
  # Add memory management parameters
  ARGS="$ARGS --memory-warning $MEMORY_WARNING --memory-limit $MEMORY_LIMIT"
  
  python3 /app/batch_processor.py "$INPUT_JSON" --output "$OUTPUT_FILE" $ARGS
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