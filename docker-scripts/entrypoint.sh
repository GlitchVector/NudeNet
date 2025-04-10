#!/bin/bash
set -e

# Function to show usage
show_usage() {
  echo "NudeNet Image Detector"
  echo "Usage:"
  echo "  * Process individual images: docker run --gpus all nudenet-gpu /path/to/image1.jpg [/path/to/image2.jpg ...]"
  echo "  * Process batch from JSON:   docker run --gpus all -v /d:/mnt/d -v /c:/mnt/c nudenet-gpu --batch /path/to/images.json --output /path/to/results.json"
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
  echo "  --debug                      Enable verbose debug logging (WARNING: may cause buffer overflow with large batches)"
  echo ""
  echo "Important: When processing images with Windows paths (e.g., D:\\path\\to\\images), you MUST mount the drives:"
  echo "  * For Windows paths (WSL2):  -v /d:/mnt/d -v /c:/mnt/c (mount each drive letter you need)"
  echo "  * For Linux paths:          -v /path/on/host:/path/in/container"
  echo ""
  echo "Example for Windows paths in WSL2:"
  echo "  docker run --gpus all -v /d:/mnt/d -v /c:/mnt/c nudenet-gpu --batch /mnt/c/temp/images.json --output /mnt/c/temp/results.json"
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

# Initialize variables for all possible parameters
JSON_PROGRESS=false
DEBUG=false
BATCH_MODE=false
BATCH_SIZE=16
MODEL_PATH=""
MEMORY_WARNING=85.0
MEMORY_LIMIT=95.0
INPUT_JSON=""
OUTPUT_FILE=""

# Array to hold positional arguments (non-flag arguments)
POSITIONAL_ARGS=()

# Parse all arguments, regardless of order
while [ $# -gt 0 ]; do
  case "$1" in
    --test-gpu)
      echo "Running NudeNet GPU Test..."
      python3 /app/test_gpu.py
      exit $?
      ;;
    --batch)
      BATCH_MODE=true
      if [ $# -lt 2 ] || [[ "$2" == --* ]]; then
        echo "Error: --batch requires an input JSON file path"
        exit 1
      fi
      INPUT_JSON=$(convert_windows_path "$2")
      shift 2
      ;;
    --output|-o)
      if [ $# -lt 2 ] || [[ "$2" == --* ]]; then
        echo "Error: --output requires a file path"
        exit 1
      fi
      OUTPUT_FILE=$(convert_windows_path "$2")
      shift 2
      ;;
    --batch-size|-b)
      if [ $# -lt 2 ] || [[ "$2" == --* ]]; then
        echo "Error: --batch-size requires a number"
        exit 1
      fi
      BATCH_SIZE="$2"
      shift 2
      ;;
    --model|-m)
      if [ $# -lt 2 ] || [[ "$2" == --* ]]; then
        echo "Error: --model requires a file path"
        exit 1
      fi
      MODEL_PATH=$(convert_windows_path "$2")
      shift 2
      ;;
    --memory-warning)
      if [ $# -lt 2 ] || [[ "$2" == --* ]]; then
        echo "Error: --memory-warning requires a percentage value"
        exit 1
      fi
      MEMORY_WARNING="$2"
      shift 2
      ;;
    --memory-limit)
      if [ $# -lt 2 ] || [[ "$2" == --* ]]; then
        echo "Error: --memory-limit requires a percentage value"
        exit 1
      fi
      MEMORY_LIMIT="$2"
      shift 2
      ;;
    --json-progress)
      JSON_PROGRESS=true
      shift
      ;;
    --debug)
      DEBUG=true
      shift
      ;;
    --*)
      echo "Error: Unknown option $1"
      show_usage
      exit 1
      ;;
    *)
      # Save any non-flag arguments (these are image paths in single-image mode)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

# Restore positional arguments
set -- "${POSITIONAL_ARGS[@]}"

# Handle batch processing mode
if [ "$BATCH_MODE" = true ]; then
  # Check required parameters
  if [ -z "$INPUT_JSON" ]; then
    echo "Error: No input JSON file specified"
    exit 1
  fi
  
  if [ -z "$OUTPUT_FILE" ]; then
    echo "Error: --output parameter is required for batch processing"
    exit 1
  fi
  
  echo "Running batch processing..."
  echo "Input JSON: $INPUT_JSON"
  echo "Output file: $OUTPUT_FILE"
  echo "Batch size: $BATCH_SIZE"
  
  # Build command arguments
  ARGS="--batch-size $BATCH_SIZE --memory-warning $MEMORY_WARNING --memory-limit $MEMORY_LIMIT"
  
  if [ ! -z "$MODEL_PATH" ]; then
    ARGS="$ARGS --model $MODEL_PATH"
  fi
  
  if [ "$JSON_PROGRESS" = true ]; then
    ARGS="$ARGS --json-progress"
  fi
  
  if [ "$DEBUG" = true ]; then
    ARGS="$ARGS --debug"
    echo "Debug logging enabled"
  fi
  
  # Create a shell script wrapper to ensure real-time output
  TMP_SCRIPT=$(mktemp)
  cat > "$TMP_SCRIPT" << 'EOF'
#!/bin/bash
# This wrapper forces each line of output to be flushed immediately
python3 -u "$@" | while IFS= read -r line; do
  echo "$line"
done
EOF
  chmod +x "$TMP_SCRIPT"
  
  # Run the batch processor through the wrapper
  "$TMP_SCRIPT" /app/batch_processor.py "$INPUT_JSON" --output "$OUTPUT_FILE" $ARGS
  EXIT_CODE=$?
  rm -f "$TMP_SCRIPT"
  exit $EXIT_CODE
fi

# If no arguments provided, show usage
if [ $# -eq 0 ]; then
  show_usage
  exit 0
fi

# Create a shell script wrapper to ensure real-time output for individual mode
TMP_SCRIPT=$(mktemp)
cat > "$TMP_SCRIPT" << 'EOF'
#!/bin/bash
# This wrapper forces each line of output to be flushed immediately
python3 -u "$@" | while IFS= read -r line; do
  echo "$line"
done
EOF
chmod +x "$TMP_SCRIPT"

# Process each image file provided as argument
for img_path in "$@"; do
  echo "Processing: $img_path"
  "$TMP_SCRIPT" python3 -c "
from nudenet import NudeDetector
import sys
import json
import os
import fcntl

# Force stdout to flush immediately
def force_flush_stdout():
    sys.stdout.flush()
    try:
        fd = sys.stdout.fileno()
        fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_SYNC)
        os.fsync(fd)
    except (AttributeError, OSError, ValueError):
        pass

try:
    detector = NudeDetector()
    results = detector.detect('$img_path')
    print(json.dumps(results, indent=2))
    force_flush_stdout()
except Exception as e:
    print(f'Error processing {img_path}: {str(e)}', file=sys.stderr)
"
done

# Clean up
rm -f "$TMP_SCRIPT"