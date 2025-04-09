# NudeNet GPU-Accelerated Docker Environment

## Summary of Changes

This document summarizes the optimizations made to the NudeNet Docker environment for GPU acceleration.

### Key Improvements

1. **Simplified Architecture**
   - Streamlined to prioritize PyTorch with direct model loading
   - Eliminated complex model conversion steps
   - Reduced dependencies to essential packages only
   - Created consistent fallback to ONNX Runtime when needed

2. **Enhanced GPU Detection & Performance**
   - Implemented smarter GPU detection in NudeDetector
   - Added symbolic links for CUDA libraries to ensure they're found
   - Set environment variables for optimal GPU utilization
   - Simplified command interface to prioritize GPU operations

3. **Improved Reliability**
   - Added robust model loading with multiple fallback mechanisms
   - Created backup detection when models fail to load
   - Enhanced error handling throughout the codebase
   - Added diagnostic commands for GPU and model verification

### Technical Details

1. **Dockerfile Changes**
   - Reduced unnecessary system dependencies
   - Pinned PyTorch to version 2.0.1 for optimal compatibility
   - Streamlined model installation process
   - Set environment variables for GPU optimization

2. **NudeDetector Enhancements**
   - Added PyTorch-first approach with automatic fallback
   - Improved model path discovery
   - Enhanced error handling and reporting
   - Added logging for better diagnostics

3. **Simple PyTorch Detector**
   - Created robust model loading that handles various formats
   - Implemented tensors-to-GPU conversion with error handling
   - Added backup detection mechanism
   - Ensured consistent output format with ONNX detector

4. **Entrypoint Simplification**
   - Streamlined command interface
   - Made PyTorch detection the default
   - Added simple benchmark capabilities
   - Improved error messages and diagnostics

### Docker Usage Examples

1. **Run detection (default)**
   ```bash
   docker run --gpus all nudenet-gpu
   ```

2. **Run with specific model and image**
   ```bash
   docker run --gpus all -v /path/to/images:/images nudenet-gpu detect 640m /images/your_image.jpg
   ```

3. **Run benchmark**
   ```bash
   docker run --gpus all nudenet-gpu benchmark 320n
   ```

4. **Check GPU status**
   ```bash
   docker run --gpus all nudenet-gpu check-gpu
   ```

5. **Start API server**
   ```bash
   docker run --gpus all -p 8080:8080 nudenet-gpu api
   ```

### Files Modified

1. **Dockerfile** - Simplified with focused dependencies
2. **docker-scripts/entrypoint.sh** - Streamlined commands
3. **docker-scripts/simple_pytorch_detector.py** - Robust model loading
4. **nudenet/nudenet.py** - PyTorch-first approach with fallback
5. **fastdeploy_recipe/predictor.py** - Enhanced GPU detection
6. **docker-scripts/README.md** - New documentation
7. **CLAUDE.md** - Updated with new Docker usage examples

For detailed usage instructions, see the docker-scripts/README.md file.