**Looking for contributors/ maintainers for this repo**:
I have become busy with other stuff in the last years, still trying to maintain this repo as it is the current best OSS option for nudity detection,
Looking for interested mainttainer, who can add/ work on more features for this repo (with my help of course)

# NudeNet: lightweight Nudity detection

https://nudenet.notai.tech/ in-browser demo (the detector is run client side, i.e: in your browser, images are not sent to a server)

```bash
pip install --upgrade "nudenet>=3.4.2"
```

```python
from nudenet import NudeDetector
detector = NudeDetector()
# the 320n model included with the package will be used

detector.detect('image.jpg') # Returns list of detections

detector.detect_batch(['image_1.jpg', 'image_2.jpg']) # Returns list of [list of detections]
```

- [Python package example in colab](https://colab.research.google.com/drive/1WChIMZ9Yzseije3Oj-Ye-cGCMLw8azvZ?usp=sharing)

- `detect` and `detect_batch` accept file path(s), opencv image(s), image bytes(s), open(image_path, 'rb') (buffereader) objects

#### Available models

| Model | resolution trained | based on            | onnx link                                                                              | pytorch link                                                                         |
| ----- | ------------------ | ------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| 320n  | 320x320            | ultralytics yolov8n | [link](https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/320n.onnx) | [link](https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/320n.pt) |
| 640m  | 640x640            | ultralytics yolov8m | [link](https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.onnx) | [link](https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.pt) |

```python
# To use the 640m model, download the onnx file and pass the path to the model_path argument

detector = NudeDetector(model_path="downloaded_640m.onnx path", inference_resolution=640)
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

### Docker

```bash
# Build the Docker image with GPU support
docker build -f docker-scripts/Dockerfile -t nudenet-gpu .

# Test GPU capabilities
docker run --gpus all nudenet-gpu --test-gpu
```

#### Processing Single Images

```bash
# Process individual images
docker run --gpus all -v /path/to/images:/images nudenet-gpu /images/image1.jpg /images/image2.jpg
```

#### Batch Processing

```bash
# Process a batch of images from a JSON file
docker run --gpus all -v /path/to/data:/data nudenet-gpu --batch /data/images.json --output /data/results.json

# With real-time progress reporting in JSON format
docker run --gpus all -v /path/to/data:/data nudenet-gpu --batch /data/images.json --output /data/results.json --json-progress

# For Windows paths (using WSL2), mount each drive letter
docker run --gpus all -v /c:/mnt/c -v /d:/mnt/d nudenet-gpu --batch /mnt/c/data/images.json --output /mnt/c/data/results.json
```

The JSON input file should contain a list of image paths:
```json
[
  "/data/image1.jpg",
  "/data/image2.jpg",
  "C:\\Users\\username\\Pictures\\image3.jpg"
]
```

#### JSON Progress Reporting

When using `--json-progress`, the Docker container outputs real-time progress information in JSON format:

```json
{"type": "start", "total_images": 670, "batch_size": 16, "input": "/data/images.json", "output": "/data/results.json", "memory_warning_threshold": 85.0, "memory_limit_threshold": 95.0, "memory_percent": 4.4, "memory_used_gb": 0.94, "memory_total_gb": 31.3}
{"type": "initialized"}
{"type": "progress", "current": 5, "total": 670, "percent": 0.7}
{"type": "progress", "current": 10, "total": 670, "percent": 1.5}
{"type": "progress", "current": 67, "total": 670, "percent": 10.0, "elapsed_seconds": 2.53, "images_per_second": 26.48, "estimated_remaining": 22.77, "memory_percent": 4.6}
// ... more progress updates ...
{"type": "complete", "total_images": 670, "processed_successfully": 670, "errors": 0, "total_time_seconds": 25.1, "average_time_per_image": 0.0375, "output_file": "/data/results.json", "memory_percent": 4.6, "memory_used_gb": 1.01, "memory_total_gb": 31.3}
```

#### Web Server

```bash
docker run -it -p8080:8080 ghcr.io/notai-tech/nudenet:latest
```

```bash
curl -F f1=@"images.jpeg" "http://localhost:8080/infer"

{"prediction": [[{"class": "BELLY_EXPOSED", "score": 0.8511635065078735, "box": [71, 182, 31, 50]}, {"class": "FACE_FEMALE", "score": 0.8033977150917053, "box": [83, 69, 21, 37]}, {"class": "FEMALE_BREAST_EXPOSED", "score": 0.7963727712631226, "box": [85, 137, 24, 38]}, {"class": "FEMALE_BREAST_EXPOSED", "score": 0.7709134817123413, "box": [63, 136, 20, 37]}, {"class": "ARMPITS_EXPOSED", "score": 0.7005534172058105, "box": [60, 127, 10, 20]}, {"class": "FEMALE_GENITALIA_EXPOSED", "score": 0.6804671287536621, "box": [81, 241, 14, 24]}]], "success": true}⏎
```

#### Some interesting projects based on NudeNet

1 - by https://github.com/w-e-w, censor extension ps://github.com/notAI-tech/NudeNet/issues/131
