# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands
- Install Python package (dev): `pip install -e .`
- Install Python package (prod): `pip install nudenet`
- Install with GPU support: `pip install -e ".[gpu]"` or `pip install "nudenet[gpu]"`
- Run browser app (dev): `cd in_browser && yarn install && yarn start`
- Build browser app: `cd in_browser && yarn build`

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

## Dependencies
- Python: numpy, onnxruntime, opencv-python-headless
- Python GPU: onnxruntime-gpu
- JS: React, onnxruntime-web, opencv-js

## Docker Usage
- Build GPU image: `docker build -t nudenet-gpu .`
- Run GPU container: `docker run --gpus all -it nudenet-gpu`
- Start API: `docker run --gpus all -p8080:8080 -it nudenet-gpu api`