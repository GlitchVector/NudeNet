#!/usr/bin/env python3
import sys
import os
import torch
import cv2
import numpy as np
import time
import argparse

# Labels matching the ONNX model
LABELS = [
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

class PyTorchNudeDetector:
    def __init__(self, model_path, inference_resolution=320):
        print(f"Loading PyTorch model from {model_path}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load model with weights_only=False for compatibility with PyTorch 2.6+
        # Check if torch.load supports the weights_only parameter (PyTorch 2.0+)
        weights_only_param_supported = False
        import inspect
        torch_load_params = inspect.signature(torch.load).parameters
        if 'weights_only' in torch_load_params:
            weights_only_param_supported = True
            print("PyTorch supports weights_only parameter")
        else:
            print("Using older PyTorch version without weights_only parameter")

        try:
            # First approach: try with modern PyTorch if supported
            if weights_only_param_supported:
                if self.device == 'cuda':
                    self.model = torch.load(model_path, map_location=torch.device('cuda'), weights_only=False)
                else:
                    self.model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
                print("Model loaded with weights_only=False")
            else:
                # For older PyTorch versions (< 2.0)
                if self.device == 'cuda':
                    self.model = torch.load(model_path, map_location=torch.device('cuda'))
                else:
                    self.model = torch.load(model_path, map_location=torch.device('cpu'))
                print("Model loaded with legacy method (older PyTorch)")
        except Exception as e:
            print(f"Error loading model with first approach: {e}")
            print("Trying alternative loading method...")
            try:
                # Fall back to try without weights_only
                if self.device == 'cuda':
                    self.model = torch.load(model_path, map_location=torch.device('cuda'))
                else:
                    self.model = torch.load(model_path, map_location=torch.device('cpu'))
                print("Model loaded with fallback method")
            except Exception as e2:
                print(f"Error loading model with fallback method: {e2}")
                print("\n---------------------------------------")
                print("MODEL LOADING FAILED: If you're using PyTorch 2.6+, this could be due to")
                print("security restrictions. Try fixing the models with:")
                print("  docker run --gpus all -it nudenet-gpu fix-models")
                print("")
                print("If that doesn't work, try using the direct ONNX runner instead:")
                print("  docker run --gpus all -it nudenet-gpu onnx 320n")
                print("---------------------------------------\n")
                raise
        
        if hasattr(self.model, 'eval'):
            self.model.eval()
            print("Model set to evaluation mode")
            
        self.input_width = inference_resolution
        self.input_height = inference_resolution
        
        # Print model information
        print(f"Model type: {type(self.model)}")
        print(f"Input resolution: {self.input_width}x{self.input_height}")
        
        # GPU info
        if self.device == 'cuda':
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            
    def preprocess_image(self, image_path):
        # Read image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path
            
        # Get original dimensions
        image_original_height, image_original_width = image.shape[:2]
        
        # Convert to RGB (PyTorch models typically use RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize and normalize
        resized = cv2.resize(image_rgb, (self.input_width, self.input_height))
        input_tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
        
        # Add batch dimension
        input_tensor = input_tensor.unsqueeze(0)
        
        # Move to GPU if available
        if self.device == 'cuda':
            input_tensor = input_tensor.cuda()
            
        return input_tensor, image_original_width, image_original_height
    
    def detect(self, image_path):
        input_tensor, image_original_width, image_original_height = self.preprocess_image(image_path)
        
        # Time inference
        start_time = time.time()
        
        with torch.no_grad():
            # Model inference
            if hasattr(self.model, 'forward'):
                outputs = self.model(input_tensor)
            else:
                print("Model doesn't have a standard forward method. This might not be a standard PyTorch model.")
                return []
                
        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time*1000:.2f} ms")
        
        # Process outputs (this depends on the model architecture)
        # For this example, we'll assume a YOLO-style output format similar to ONNX
        if isinstance(outputs, torch.Tensor):
            detections = self.process_yolo_output(
                outputs, 
                image_original_width, 
                image_original_height
            )
        elif isinstance(outputs, (tuple, list)):
            # If multiple outputs, assume first one has detections
            detections = self.process_yolo_output(
                outputs[0], 
                image_original_width, 
                image_original_height
            )
        else:
            print(f"Unexpected output type: {type(outputs)}")
            return []
            
        return detections
    
    def process_yolo_output(self, output, original_width, original_height):
        # Move to CPU for numpy processing
        if self.device == 'cuda':
            output = output.cpu()
            
        # Convert to numpy
        outputs = output.numpy()
        
        # Assuming YOLO format output with shape [batch, num_detections, 4+1+num_classes]
        if len(outputs.shape) == 3:
            outputs = outputs[0]  # Get first batch
            
        # Extract predictions
        boxes = []
        scores = []
        class_ids = []
        
        for detection in outputs:
            # Extract box coordinates, confidence, and class probabilities
            if len(detection) >= 5:  # x, y, w, h, obj_conf + class_probs
                x, y, w, h = detection[0:4]
                confidence = detection[4]
                
                if confidence > 0.25:  # Confidence threshold
                    # Get class with highest probability
                    if len(detection) > 5:
                        class_probs = detection[5:]
                        class_id = np.argmax(class_probs)
                        class_score = class_probs[class_id]
                        final_score = confidence * class_score
                        
                        if final_score > 0.25:  # Final score threshold
                            # Convert from normalized to absolute coordinates
                            x_center = x * original_width
                            y_center = y * original_height
                            box_width = w * original_width
                            box_height = h * original_height
                            
                            # Convert from center to top-left corner format
                            x1 = x_center - box_width / 2
                            y1 = y_center - box_height / 2
                            
                            boxes.append([int(x1), int(y1), int(box_width), int(box_height)])
                            scores.append(float(final_score))
                            class_ids.append(int(class_id))
        
        # Apply NMS (Non-Maximum Suppression)
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)
            
            # Create detections list
            detections = []
            for i in indices:
                if isinstance(i, list) or isinstance(i, tuple) or isinstance(i, np.ndarray):
                    i = i[0]  # Handle OpenCV 4.2+ vs 4.5+ differences
                    
                box = boxes[i]
                score = scores[i]
                class_id = class_ids[i]
                
                label = LABELS[class_id] if class_id < len(LABELS) else f"class_{class_id}"
                
                detections.append({
                    "class": label,
                    "score": float(score),
                    "box": box
                })
                
            return detections
        else:
            return []
        
def main(args):
    detector = PyTorchNudeDetector(args.model, args.resolution)
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return 1
        
    print(f"Running detection on {args.image}")
    detections = detector.detect(args.image)
    
    print("\nDetection Results:")
    if not detections:
        print("No detections found.")
    else:
        for i, detection in enumerate(detections):
            print(f"Detection {i+1}:")
            print(f"  Class: {detection['class']}")
            print(f"  Score: {detection['score']:.4f}")
            print(f"  Box: {detection['box']}")
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch NudeNet Detector")
    parser.add_argument('--model', type=str, default="/app/models/pytorch/320n.pt", 
                        help='Path to PyTorch model file (.pt)')
    parser.add_argument('--image', type=str, default="/app/fastdeploy_recipe/cory_chase.jpeg", 
                        help='Path to input image file')
    parser.add_argument('--resolution', type=int, default=320,
                        help='Input resolution (default: 320)')
    
    args = parser.parse_args()
    sys.exit(main(args))