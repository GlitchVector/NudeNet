import os
import _io
import math
import cv2
import numpy as np
import onnxruntime
import logging
from onnxruntime.capi import _pybind_state as C

__labels = [
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


def _read_image(image_path, target_size=320):
    if isinstance(image_path, str):
        mat = cv2.imread(image_path)
    elif isinstance(image_path, np.ndarray):
        mat = image_path
    elif isinstance(image_path, bytes):
        mat = cv2.imdecode(np.frombuffer(image_path, np.uint8), -1)
    elif isinstance(image_path, _io.BufferedReader):
        mat = cv2.imdecode(np.frombuffer(image_path.read(), np.uint8), -1)
    else:
        raise ValueError(
            "please make sure the image_path is str or np.ndarray or bytes"
        )

    image_original_width, image_original_height = mat.shape[1], mat.shape[0]

    mat_c3 = cv2.cvtColor(mat, cv2.COLOR_RGBA2BGR)

    max_size = max(mat_c3.shape[:2])  # get max size from width and height
    x_pad = max_size - mat_c3.shape[1]  # set xPadding
    x_ratio = max_size / mat_c3.shape[1]  # set xRatio
    y_pad = max_size - mat_c3.shape[0]  # set yPadding
    y_ratio = max_size / mat_c3.shape[0]  # set yRatio

    mat_pad = cv2.copyMakeBorder(mat_c3, 0, y_pad, 0, x_pad, cv2.BORDER_CONSTANT)

    input_blob = cv2.dnn.blobFromImage(
        mat_pad,
        1 / 255.0,  # normalize
        (target_size, target_size),  # resize to model input size
        (0, 0, 0),  # mean subtraction
        swapRB=True,  # swap red and blue channels
        crop=False,  # don't crop
    )

    return (
        input_blob,
        x_ratio,
        y_ratio,
        x_pad,
        y_pad,
        image_original_width,
        image_original_height,
    )


def _postprocess(
    output,
    x_pad,
    y_pad,
    x_ratio,
    y_ratio,
    image_original_width,
    image_original_height,
    model_width,
    model_height,
):
    outputs = np.transpose(np.squeeze(output[0]))
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)

        if max_score >= 0.2:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][0:4]

            # Convert from center coordinates to top-left corner coordinates
            x = x - w / 2
            y = y - h / 2

            # Scale coordinates to original image size
            x = x * (image_original_width + x_pad) / model_width
            y = y * (image_original_height + y_pad) / model_height
            w = w * (image_original_width + x_pad) / model_width
            h = h * (image_original_height + y_pad) / model_height

            # Remove padding
            x = x
            y = y

            # Clip coordinates to image boundaries
            x = max(0, min(x, image_original_width))
            y = max(0, min(y, image_original_height))
            w = min(w, image_original_width - x)
            h = min(h, image_original_height - y)

            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)

    detections = []
    for i in indices:
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]

        x, y, w, h = box
        detections.append(
            {
                "class": __labels[class_id],
                "score": float(score),
                "box": [int(x), int(y), int(w), int(h)],
            }
        )

    return detections


class NudeDetector:
    def __init__(self, model_path=None, providers=None, inference_resolution=320, use_pytorch=None):
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # For Docker containers with explicit CUDA versions
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first GPU
        os.environ["OMP_NUM_THREADS"] = "1"  # Avoid CPU thread competition
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Prevent GPU OOM
        
        # Set the resolution
        self.input_width = inference_resolution
        self.input_height = inference_resolution
        
        # Try to use PyTorch if available and CUDA is detected
        self.use_pytorch = False
        self.pytorch_detector = None
        
        # Check if we should try PyTorch first
        if use_pytorch is None or use_pytorch:
            try:
                # Import torch and check CUDA availability
                import torch
                if torch.cuda.is_available():
                    self.logger.info("CUDA is available for PyTorch, will try to use PyTorch detector")
                    
                    # Look for models in standard locations
                    pytorch_model_path = None
                    if not model_path:
                        # First check in /app/models/pytorch
                        if os.path.exists("/app/models/pytorch/320n.pt"):
                            pytorch_model_path = f"/app/models/pytorch/320n.pt"
                        elif os.path.exists(f"/app/docker-scripts/pytorch-models/320n.pt"):
                            pytorch_model_path = f"/app/docker-scripts/pytorch-models/320n.pt"
                    else:
                        # If model_path is provided but points to an ONNX model, 
                        # try to find the corresponding PyTorch model
                        if model_path.endswith(".onnx"):
                            base_path = model_path.replace(".onnx", ".pt")
                            if os.path.exists(base_path):
                                pytorch_model_path = base_path
                            else:
                                # Check standard locations with same model name
                                model_name = os.path.basename(model_path).replace(".onnx", "")
                                if os.path.exists(f"/app/models/pytorch/{model_name}.pt"):
                                    pytorch_model_path = f"/app/models/pytorch/{model_name}.pt"
                        else:
                            # If it's already a PyTorch model, use it directly
                            pytorch_model_path = model_path
                    
                    if pytorch_model_path:
                        self.logger.info(f"Loading PyTorch model from {pytorch_model_path}")
                        # Import the PyTorch detector
                        import sys
                        if os.path.exists("/app/docker-scripts/pytorch_detector.py"):
                            sys.path.append("/app/docker-scripts")
                            # Use the cached model implementation for better performance
                            from pytorch_detector import get_cached_model
                            
                            # Get or create the detector using the cached implementation
                            self.pytorch_detector = get_cached_model(pytorch_model_path, 'cuda')
                            self.use_pytorch = True
                            self.logger.info("PyTorch detector initialized successfully")
                        else:
                            self.logger.warning("Could not find simple_pytorch_detector.py, falling back to ONNX")
                    else:
                        self.logger.warning("No PyTorch model found, falling back to ONNX")
                else:
                    self.logger.info("CUDA not available for PyTorch, using ONNX Runtime")
            except ImportError:
                self.logger.warning("PyTorch not installed, using ONNX Runtime")
            except Exception as e:
                self.logger.warning(f"Error initializing PyTorch detector: {e}, falling back to ONNX Runtime")
        
        # Initialize ONNX Runtime as fallback or if PyTorch is not available
        if not self.use_pytorch or use_pytorch is False:
            self.logger.info("Initializing ONNX Runtime detector")
            os.environ["ONNX_BACKEND"] = "CUDAExecutionProvider"  # Force CUDA as backend
            onnxruntime.set_default_logger_severity(0)  # Set to verbose logging
            
            # Set up GPU providers with more aggressive settings
            gpu_providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
                'TensorrtExecutionProvider',
                'CPUExecutionProvider'
            ]
            
            # Try loading CUDA libraries directly to ensure they're found
            try:
                import ctypes
                # Try multiple possible library paths
                cuda_paths = [
                    "libcudart.so",                      # Standard system path
                    "/usr/local/cuda/lib64/libcudart.so", # Default CUDA installation
                    "/usr/lib/libcudart.so",              # Our symlink
                    "/usr/lib/x86_64-linux-gnu/libcudart.so" # Ubuntu specific path
                ]
                
                loaded = False
                for path in cuda_paths:
                    try:
                        ctypes.CDLL(path)
                        self.logger.info(f"Successfully loaded CUDA runtime library from {path}")
                        loaded = True
                        break
                    except Exception as path_e:
                        self.logger.debug(f"Could not load from {path}: {path_e}")
                
                if not loaded:
                    self.logger.warning("Could not load CUDA runtime library from any known path")
            except Exception as e:
                self.logger.warning(f"Error while trying to load CUDA runtime library: {e}")
            
            try:
                # Try to use GPU providers by default
                if providers is None:
                    available_providers = C.get_available_providers()
                    self.logger.info(f"Available providers: {available_providers}")
                    
                    # Check if CUDA is available
                    if 'CUDAExecutionProvider' in available_providers:
                        self.logger.info("CUDA is available, using GPU")
                        providers = gpu_providers
                    else:
                        self.logger.info("CUDA is not available, falling back to CPU")
                        providers = ['CPUExecutionProvider']
                
                self.onnx_session = onnxruntime.InferenceSession(
                    os.path.join(os.path.dirname(__file__), "320n.onnx")
                    if not model_path
                    else model_path,
                    providers=providers,
                )
                
                # Log which providers are actually being used
                self.logger.info(f"Using providers: {self.onnx_session.get_providers()}")
                
            except Exception as e:
                self.logger.warning(f"Error initializing GPU, falling back to CPU: {str(e)}")
                # Fall back to CPU if GPU initialization fails
                self.onnx_session = onnxruntime.InferenceSession(
                    os.path.join(os.path.dirname(__file__), "320n.onnx")
                    if not model_path
                    else model_path,
                    providers=['CPUExecutionProvider'],
                )
                
            model_inputs = self.onnx_session.get_inputs()
            self.input_name = model_inputs[0].name

    def detect(self, image_path):
        # Use PyTorch detector if available
        if self.use_pytorch and self.pytorch_detector:
            try:
                detections = self.pytorch_detector.detect(image_path)
                return detections
            except Exception as e:
                self.logger.warning(f"PyTorch detection failed: {e}, falling back to ONNX Runtime")
        
        # Fall back to ONNX Runtime or use it directly if PyTorch is not available
        (
            preprocessed_image,
            x_ratio,
            y_ratio,
            x_pad,
            y_pad,
            image_original_width,
            image_original_height,
        ) = _read_image(image_path, self.input_width)
        outputs = self.onnx_session.run(None, {self.input_name: preprocessed_image})
        detections = _postprocess(
            outputs,
            x_pad,
            y_pad,
            x_ratio,
            y_ratio,
            image_original_width,
            image_original_height,
            self.input_width,
            self.input_height,
        )

        return detections

    def detect_batch(self, image_paths, batch_size=4):
        """
        Perform batch detection on a list of images.

        Args:
            image_paths (List[Union[str, np.ndarray]]): List of image paths or numpy arrays.
            batch_size (int): Number of images to process in each batch.

        Returns:
            List of detection results for each image.
        """
        # Use PyTorch detector if available (more efficient batch processing)
        if self.use_pytorch and self.pytorch_detector:
            try:
                self.logger.debug(f"Using PyTorch batch detection for {len(image_paths)} images")
                detections = self.pytorch_detector.detect_batch(image_paths, batch_size)
                return detections
            except Exception as e:
                self.logger.warning(f"PyTorch batch detection failed: {e}, falling back to ONNX Runtime")
        
        # Fall back to ONNX Runtime if PyTorch is not available
        all_detections = []

        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i : i + batch_size]
            batch_inputs = []
            batch_metadata = []

            for image_path in batch:
                try:
                    (
                        preprocessed_image,
                        x_ratio,
                        y_ratio,
                        x_pad,
                        y_pad,
                        image_original_width,
                        image_original_height,
                    ) = _read_image(image_path, self.input_width)
                    batch_inputs.append(preprocessed_image)
                    batch_metadata.append(
                        (
                            x_ratio,
                            y_ratio,
                            x_pad,
                            y_pad,
                            image_original_width,
                            image_original_height,
                        )
                    )
                except Exception as e:
                    self.logger.error(f"Error processing image {image_path}: {e}")
                    all_detections.append([])  # Add empty detection for failed image

            # If no valid images in this batch, continue to next batch
            if not batch_inputs:
                continue

            # Stack the preprocessed images into a single numpy array
            try:
                batch_input = np.vstack(batch_inputs)

                # Run inference on the batch
                outputs = self.onnx_session.run(None, {self.input_name: batch_input})

                # Process the outputs for each image in the batch
                for j, metadata in enumerate(batch_metadata):
                    (
                        x_ratio,
                        y_ratio,
                        x_pad,
                        y_pad,
                        image_original_width,
                        image_original_height,
                    ) = metadata
                    detections = _postprocess(
                        [outputs[0][j : j + 1]],  # Select the output for this image
                        x_pad,
                        y_pad,
                        x_ratio,
                        y_ratio,
                        image_original_width,
                        image_original_height,
                        self.input_width,
                        self.input_height,
                    )
                    all_detections.append(detections)
            except Exception as e:
                self.logger.error(f"Error during batch inference: {e}")
                # Add empty detections for all images in this batch
                for _ in range(len(batch_metadata)):
                    all_detections.append([])

        return all_detections

    def censor(self, image_path, classes=[], output_path=None):
        detections = self.detect(image_path)
        if classes:
            detections = [
                detection for detection in detections if detection["class"] in classes
            ]

        img = cv2.imread(image_path)

        for detection in detections:
            box = detection["box"]
            x, y, w, h = box[0], box[1], box[2], box[3]
            # change these pixels to pure black
            img[y : y + h, x : x + w] = (0, 0, 0)

        if not output_path:
            image_path, ext = os.path.splitext(image_path)
            output_path = f"{image_path}_censored{ext}"

        cv2.imwrite(output_path, img)

        return output_path


if __name__ == "__main__":
    detector = NudeDetector()
    # detections = detector.detect("/Users/praneeth.bedapudi/Desktop/cory.jpeg")
    print(
        detector.detect_batch(
            [
                "/Users/praneeth.bedapudi/Desktop/d.jpg",
                "/Users/praneeth.bedapudi/Desktop/a.jpeg",
            ]
        )[0]
    )
    print(detector.detect_batch(["/Users/praneeth.bedapudi/Desktop/d.jpg"])[0])

    print(
        detector.detect_batch(
            [
                "/Users/praneeth.bedapudi/Desktop/d.jpg",
                "/Users/praneeth.bedapudi/Desktop/a.jpeg",
            ]
        )[1]
    )
    print(detector.detect_batch(["/Users/praneeth.bedapudi/Desktop/a.jpeg"])[0])
