# NudeNet API Server Documentation

This directory contains the configuration for running NudeNet as an API server using FastDeploy.

## Quick Start

Start the API server with GPU acceleration:

```bash
docker run --gpus all -p 8080:8080 nudenet-gpu api
```

## API Usage

The API server uses HTTP POST requests with JSON payloads. The primary endpoint is:

`POST /api/predict/`

### Request Format

Send a POST request with a JSON body containing:

```json
{
  "image_paths": ["path/to/file1.jpg", "path/to/file2.jpg"],
  "batch_size": 4
}
```

Or for a single image:

```json
{
  "image_paths": ["path/to/file.jpg"]
}
```

For base64-encoded images:

```json
{
  "image_data": ["base64-encoded-image-1", "base64-encoded-image-2"]
}
```

### Response Format

The API returns a JSON array of detection results. Each element in the array corresponds to one image:

```json
[
  [
    {
      "class": "FACE_FEMALE",
      "score": 0.93,
      "box": [100, 50, 200, 200]
    },
    {
      "class": "FEMALE_BREAST_EXPOSED",
      "score": 0.85,
      "box": [300, 200, 100, 100]
    }
  ],
  [
    {
      "class": "FACE_MALE",
      "score": 0.91,
      "box": [120, 60, 180, 180]
    }
  ]
]
```

Where:
- `class`: The detected class label
- `score`: Confidence score (0-1)
- `box`: Bounding box in format [x, y, width, height]

### Example API Requests

#### Python with requests

```python
import requests
import base64

# Image path method
response = requests.post('http://localhost:8080/api/predict/', 
                         json={'image_paths': ['/path/to/image.jpg']})

# Base64 method
with open('image.jpg', 'rb') as f:
    img_data = base64.b64encode(f.read()).decode('utf-8')
    
response = requests.post('http://localhost:8080/api/predict/',
                        json={'image_data': [img_data]})

results = response.json()
print(results)
```

#### cURL

```bash
curl -X POST \
  http://localhost:8080/api/predict/ \
  -H 'Content-Type: application/json' \
  -d '{"image_paths": ["/path/to/image.jpg"]}'
```

### Available Class Labels

The API returns detections with the following class labels:

```
FEMALE_GENITALIA_COVERED
FACE_FEMALE
BUTTOCKS_EXPOSED
FEMALE_BREAST_EXPOSED
FEMALE_GENITALIA_EXPOSED
MALE_BREAST_EXPOSED
ANUS_EXPOSED
FEET_EXPOSED
BELLY_COVERED
FEET_COVERED
ARMPITS_COVERED
ARMPITS_EXPOSED
FACE_MALE
BELLY_EXPOSED
MALE_GENITALIA_EXPOSED
ANUS_COVERED
FEMALE_BREAST_COVERED
BUTTOCKS_COVERED
```

## Advanced Configuration

### Environment Variables

- `NUDENET_LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `PYTORCH_DISABLE`: Set to "1" to disable PyTorch and use only ONNX Runtime
- `CUDA_VISIBLE_DEVICES`: Control which GPU devices to use
- `PYTORCH_CUDA_ALLOC_CONF`: Configure PyTorch CUDA memory allocation

Example:

```bash
docker run --gpus all -p 8080:8080 \
  -e NUDENET_LOG_LEVEL=DEBUG \
  -e CUDA_VISIBLE_DEVICES=0 \
  nudenet-gpu api
```

### Mounting Custom Images

For processing your own images, mount a directory to the container:

```bash
docker run --gpus all -p 8080:8080 -v /path/to/your/images:/images nudenet-gpu api
```

Then use paths like `/images/your_image.jpg` in your API requests.

## Performance Considerations

- GPU acceleration automatically uses PyTorch when available for best performance
- For batch processing of multiple images, use the `batch_size` parameter
- Recommended batch sizes: 4-8 for optimal throughput
- The API server automatically caches models for better performance