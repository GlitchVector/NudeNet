# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands
- Install Python package (dev): `pip install -e .`
- Install Python package (prod): `pip install nudenet`
- Install with GPU support: `pip install -e ".[gpu]"` or `pip install "nudenet[gpu]"`
- Run browser app (dev): `cd in_browser && yarn install && yarn start`
- Build browser app: `cd in_browser && yarn build`
- Typecheck/lint: N/A (no specific commands for this project)

## Code Style Guidelines
- Python:
  - 4-space indentation
  - CamelCase for classes, snake_case for functions/variables
  - Document function parameters in docstrings
  - Use explicit exception types with descriptive messages
  - Type hints in docstrings, not annotations
- JavaScript:
  - camelCase for variables and functions
  - JSDoc for function documentation
  - Use ES6 features (async/await, destructuring)

## Project Structure
- `nudenet/`: Core YOLO-based detection library
- `in_browser/`: React web implementation
- `NSFWSniffer/`: PySide6 desktop application
- `docker-scripts/`: Docker container support files and utilities
- `fastdeploy_recipe/`: FastDeploy API configuration

## Dependencies
- Python: numpy, onnxruntime, opencv-python-headless
- Python GPU: torch, onnxruntime-gpu
- JS: React, onnxruntime-web, opencv-js

## Docker Usage
- Build GPU image: `docker build -t nudenet-gpu .`
- Run PyTorch detection: `docker run --gpus all nudenet-gpu` or `docker run --gpus all nudenet-gpu detect 640m /path/to/image.jpg`
- Run benchmark: `docker run --gpus all nudenet-gpu benchmark 320n`
- Check GPU: `docker run --gpus all nudenet-gpu check-gpu`
- Run test: `docker run --gpus all nudenet-gpu test /path/to/image.jpg`
- Start API: `docker run --gpus all -p8080:8080 nudenet-gpu api`

## GPU Acceleration
NudeNet supports GPU acceleration with two approaches:
1. PyTorch with CUDA (prioritized when available)
2. ONNX Runtime with CUDAExecutionProvider (fallback)

The Docker configuration prioritizes PyTorch and will automatically use it when CUDA is available.