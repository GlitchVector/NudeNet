# NudeNet Docker Scripts Reference

This document provides an overview of the scripts in the `docker-scripts` directory.

## Core Scripts

### 1. entrypoint.sh

Main container entry point that handles commands and environment variables.

- **Usage**: Called automatically when container starts
- **Commands**: detect, benchmark, check-gpu, test, compare, models, api
- **Environment variables**: TRY_ULTRALYTICS

### 2. pytorch_detector.py

Main detector implementation for PyTorch models.

- **Usage**: Used by other tools, typically not called directly
- **Features**: 
  - YOLOv8 model loading with Ultralytics
  - Robust fallback mechanisms
  - Batch processing support
  - Comprehensive error handling

## Utility Scripts

### 3. gpu_diagnostics.py

Comprehensive GPU diagnostics tool that combines multiple testing utilities.

- **Usage**: `python3 gpu_diagnostics.py [--mode MODE]`
- **Modes**: all, torch, onnx, system, detector, pytorch

### 4. model_manager.py

Unified model management tool for downloading, verifying, and listing models.

- **Usage**: `python3 model_manager.py [--action ACTION]`
- **Actions**: download, verify, list, ensure

### 5. test_utils.py

Comprehensive testing and benchmarking utilities.

- **Usage**: `python3 test_utils.py COMMAND [OPTIONS]`
- **Commands**:
  - `test`: Run a single detection test
  - `benchmark`: Run performance benchmarking
  - `batch`: Run batch detection tests
  - `compare`: Compare different detector implementations

### 6. fix_import.py

Utility to fix Python imports in container environments.

- **Usage**: Run automatically by entrypoint.sh if needed

## Auxiliary Files

### 7. debug.sh

Debugging utility for container troubleshooting.

- **Usage**: `bash debug.sh`

### 8. __init__.py

Makes the directory a Python package for easier importing.

### 9. README.md and SUMMARY.md

Documentation for the Docker environment and scripts.

## Usage Examples

### Running a Detection

```bash
# Default model (320n)
python3 test_utils.py test --detector pytorch

# Specific model and image
python3 test_utils.py test --detector pytorch --model /path/to/model.pt --image /path/to/image.jpg
```

### Benchmarking

```bash
# Default benchmark
python3 test_utils.py benchmark

# Custom iterations and warmup
python3 test_utils.py benchmark --iterations 20 --warmup 3
```

### Comparing Detectors

```bash
# Compare all available detectors
python3 test_utils.py compare

# Specific image and iterations
python3 test_utils.py compare --image /path/to/image.jpg --iterations 10
```

### Managing Models

```bash
# List available models
python3 model_manager.py --action list

# Download missing models
python3 model_manager.py --action download

# Verify model integrity
python3 model_manager.py --action verify
```

### GPU Diagnostics

```bash
# Run all diagnostic tests
python3 gpu_diagnostics.py

# Test specific component
python3 gpu_diagnostics.py --mode torch
```