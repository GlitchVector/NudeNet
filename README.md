**Looking for contributors/ maintainers for this repo**: 
I have become busy with other stuff in the last years, still trying to maintain this repo as it is the current best OSS option for nudity detection,
Looking for interested mainttainer, who can add/ work on more features for this repo (with my help of course)

# NudeNet: lightweight Nudity detection

https://nudenet.notai.tech/ in-browser demo (the detector is run client side, i.e: in your browser, images are not sent to a server)

## Installation

### Basic Installation
```bash
pip install --upgrade "nudenet>=3.4.2"
```

### GPU Support (Recommended for faster processing)
```bash
pip install --upgrade "nudenet[gpu]>=3.4.2"
```

```python
from nudenet import NudeDetector

# Initialize detector (will automatically use GPU if available)
detector = NudeDetector()
# the 320n model included with the package will be used

# For explicit GPU provider selection
# detector = NudeDetector(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

detector.detect('image.jpg') # Returns list of detections

detector.detect_batch(['image_1.jpg', 'image_2.jpg']) # Returns list of [list of detections]
```

- [Python package example in colab](https://colab.research.google.com/drive/1WChIMZ9Yzseije3Oj-Ye-cGCMLw8azvZ?usp=sharing)

- `detect` and `detect_batch` accept file path(s), opencv image(s), image bytes(s), open(image_path, 'rb') (buffereader) objects

#### Available models

| Model | resolution trained | based on | onnx link | pytorch link |
| --- | --- | --- | --- | -- |
| 320n | 320x320 | ultralytics yolov8n | [link](https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/320n.onnx) | [link](https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/320n.pt)
| 640m | 640x640 | ultralytics yolov8m | [link](https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.onnx) | [link](https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.pt)

```python
# To use the 640m model, download the onnx file and pass the path to the model_path argument

detector = NudeDetector(model_path="downloaded_640m.onnx path", inference_resolution=640)

# With explicit GPU providers
detector = NudeDetector(
    model_path="downloaded_640m.onnx path", 
    inference_resolution=640,
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

- 320n is the default model and is included in the `nudenet` python package by default


```python
detection_example = [
 {'class': 'BELLY_EXPOSED',
  'score': 0.799403190612793,
  'box': [64, 182, 49, 51]},
 {'class': 'FACE_FEMALE',
  'score': 0.7881264686584473,
  'box': [82, 66, 36, 43]},
 ]
```

```python
nude_detector.censor('image.jpg') # returns censored image output path

# optional censor(self, image_path, classes=[], output_path=None) classes and output_path can be passed
```

```python
all_labels = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]
```


### GPU Acceleration

NudeNet now automatically uses GPU acceleration if available:

1. Install the GPU dependencies: `pip install "nudenet[gpu]"`
2. The detector will automatically detect and use CUDA if available
3. GPU providers will be prioritized in this order: CUDA → TensorRT → CPU
4. If GPU initialization fails, it will automatically fall back to CPU

You can also specify providers explicitly:
```python
# Use CUDA with specific configuration
detector = NudeDetector(providers=[
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider'
])

# Force CPU only
detector = NudeDetector(providers=['CPUExecutionProvider'])
```

#### Troubleshooting GPU Issues

If you encounter GPU-related issues, try these solutions:

1. **Check GPU Availability**:
   ```python
   import onnxruntime
   print("Available providers:", onnxruntime.get_available_providers())
   
   # For PyTorch
   import torch
   print("CUDA available:", torch.cuda.is_available())
   ```

2. **PyTorch and NumPy Compatibility Issues**:
   - Our container uses PyTorch 2.0.1 with CUDA 11.8 and NumPy 1.x
   - This combination avoids both the PyTorch 2.6+ security restrictions and NumPy 2.0 compatibility issues
   - If you encounter model loading issues, run `docker run --gpus all -it nudenet-gpu fix-models`
   - If you see NumPy-related errors, try the direct ONNX runner: `docker run --gpus all -it nudenet-gpu onnx 320n`
   - You can force re-download of all models with: `docker run --gpus all -it nudenet-gpu download-models force`

3. **ONNX Runtime vs PyTorch**:
   - NudeNet supports both ONNX Runtime and PyTorch for GPU acceleration
   - ONNX Runtime is used by default in the Python package
   - The Docker container supports both backends with comprehensive diagnostics
   - When one backend has issues, try the other:
     ```bash
     # Test ONNX Runtime GPU support
     docker run --gpus all -it nudenet-gpu check-gpu
     
     # Try PyTorch if ONNX Runtime has issues
     docker run --gpus all -it nudenet-gpu pytorch 320n
     
     # Try direct ONNX runner if PyTorch has issues
     docker run --gpus all -it nudenet-gpu onnx 320n
     ```

4. **GPU Memory Issues**:
   - If you're encountering CUDA out of memory errors:
   ```python
   # Reduce GPU memory usage
   detector = NudeDetector(providers=[
       ('CUDAExecutionProvider', {
           'device_id': 0,
           'gpu_mem_limit': 1 * 1024 * 1024 * 1024,  # Limit to 1GB
       }),
       'CPUExecutionProvider'
   ])
   ```

### Docker

#### CPU Version
```bash
docker run -it -p8080:8080 ghcr.io/notai-tech/nudenet:latest
```

#### GPU Version (CUDA 11.8.0 on Ubuntu 22.04)
Build the GPU-enabled container:
```bash
docker build -t nudenet-gpu .
```

Run with NVIDIA Container Toolkit:
```bash
# First run will automatically download all models (ONNX and PyTorch)
docker run --gpus all -it nudenet-gpu

# Check GPU availability with detailed diagnostics
docker run --gpus all -it nudenet-gpu check-gpu

# Run container diagnostics to troubleshoot issues
docker run --gpus all -it nudenet-gpu debug

# If you encounter "Module not found" errors or build issues:
# Use the simplified Dockerfile instead
docker build -t nudenet-gpu -f simple_dockerfile .
docker run --gpus all -it nudenet-gpu check-gpu

# Download all model variants (if not already downloaded)
docker run --gpus all -it nudenet-gpu download-models

# Force re-download all models (if you encounter corrupted model files)
docker run --gpus all -it nudenet-gpu download-models force

# Run benchmarks
docker run --gpus all -it nudenet-gpu benchmark           # Default ONNX 320n benchmark
docker run --gpus all -it nudenet-gpu benchmark 640       # ONNX 640m benchmark
docker run --gpus all -it nudenet-gpu benchmark pytorch   # PyTorch benchmark
docker run --gpus all -it nudenet-gpu benchmark compare   # Compare ONNX vs PyTorch

# Run tests
docker run --gpus all -it nudenet-gpu test                # Test both ONNX and PyTorch models
docker run --gpus all -it nudenet-gpu test /path/to/image.jpg # Test with custom image

# Run with PyTorch models
docker run --gpus all -it nudenet-gpu pytorch 320n        # Run 320n PyTorch model
docker run --gpus all -it nudenet-gpu pytorch 640m        # Run 640m PyTorch model

# Run with direct ONNX runner (useful if PyTorch has issues)
docker run --gpus all -it nudenet-gpu onnx 320n           # Run 320n ONNX model directly
docker run --gpus all -it nudenet-gpu onnx 640m           # Run 640m ONNX model directly

# Fix PyTorch models for compatibility with PyTorch 2.6+
docker run --gpus all -it nudenet-gpu fix-models

# Start API server
docker run --gpus all -p8080:8080 -it nudenet-gpu api

# Run bash shell
docker run --gpus all -it nudenet-gpu bash
```

Example API request:
```bash
curl -F f1=@"images.jpeg" "http://localhost:8080/infer"

{"prediction": [[{"class": "BELLY_EXPOSED", "score": 0.8511635065078735, "box": [71, 182, 31, 50]}, {"class": "FACE_FEMALE", "score": 0.8033977150917053, "box": [83, 69, 21, 37]}, {"class": "FEMALE_BREAST_EXPOSED", "score": 0.7963727712631226, "box": [85, 137, 24, 38]}, {"class": "FEMALE_BREAST_EXPOSED", "score": 0.7709134817123413, "box": [63, 136, 20, 37]}, {"class": "ARMPITS_EXPOSED", "score": 0.7005534172058105, "box": [60, 127, 10, 20]}, {"class": "FEMALE_GENITALIA_EXPOSED", "score": 0.6804671287536621, "box": [81, 241, 14, 24]}]], "success": true}⏎
```

#### Some interesting projects based on NudeNet
1 - by https://github.com/w-e-w, censor extension ps://github.com/notAI-tech/NudeNet/issues/131
