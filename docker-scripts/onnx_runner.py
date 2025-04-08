#!/usr/bin/env python3
import argparse
import sys
import time
import os
import numpy as np
import onnxruntime as ort
import cv2
from pathlib import Path

class ONNXModelRunner:
    """
    Class to run inference with ONNX models
    """
    def __init__(self, model_path, providers=None, device_id=0):
        self.model_path = model_path
        self.device_id = device_id
        
        # Configure providers
        self.providers = providers or self._get_default_providers()
        
        # Initialize session
        self._init_session()
        
    def _get_default_providers(self):
        """Get default providers with CUDA first if available"""
        available_providers = ort.get_available_providers()
        
        # Check if CUDA is available
        if "CUDAExecutionProvider" in available_providers:
            # Configure CUDA with specific device
            cuda_options = {
                'device_id': self.device_id,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }
            
            return [
                ('CUDAExecutionProvider', cuda_options),
                'CPUExecutionProvider'
            ]
        
        # No CUDA, use CPU
        return ['CPUExecutionProvider']
    
    def _init_session(self):
        """Initialize the ONNX Runtime session"""
        try:
            # Create session options
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 0  # Verbose logging
            sess_options.enable_profiling = True
            
            print(f"Initializing ONNX Runtime session with providers: {self.providers}")
            
            # Create session
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=self.providers
            )
            
            # Get input and output details
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            print(f"Model loaded successfully from {self.model_path}")
            print(f"Input name: {self.input_name}")
            print(f"Input shape: {self.input_shape}")
            print(f"Output names: {self.output_names}")
            print(f"Active providers: {self.session.get_providers()}")
            
        except Exception as e:
            print(f"Error initializing ONNX Runtime session: {e}")
            raise
    
    def preprocess_image(self, image_path, target_size=None):
        """Preprocess image for inference"""
        # Read image
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image from {image_path}")
        elif isinstance(image_path, np.ndarray):
            img = image_path
        else:
            raise ValueError("Image must be a file path or numpy array")
        
        # Get target size from model input shape if not specified
        if target_size is None:
            if len(self.input_shape) == 4:
                # NCHW format
                height, width = self.input_shape[2], self.input_shape[3]
            else:
                # Default size
                height, width = 320, 320
        else:
            height, width = target_size
        
        # Resize and normalize
        img_resized = cv2.resize(img, (width, height))
        
        # Convert to RGB and normalize
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # NHWC to NCHW (if needed)
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        
        # Add batch dimension
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        return img_batch, img
    
    def run_inference(self, image_path, target_size=None):
        """Run inference on an image"""
        # Preprocess image
        input_data, original_img = self.preprocess_image(image_path, target_size)
        
        # Prepare input
        model_inputs = {self.input_name: input_data}
        
        # Run inference
        start_time = time.time()
        outputs = self.session.run(self.output_names, model_inputs)
        inference_time = time.time() - start_time
        
        print(f"Inference time: {inference_time*1000:.2f} ms")
        
        return outputs, inference_time, original_img

def main():
    parser = argparse.ArgumentParser(description="ONNX Model Runner")
    parser.add_argument('--model', type=str, default="/app/models/onnx/320n.onnx",
                        help="Path to ONNX model")
    parser.add_argument('--image', type=str, default="/app/fastdeploy_recipe/cory_chase.jpeg",
                        help="Path to input image")
    parser.add_argument('--device', type=int, default=0,
                        help="GPU device ID (default: 0)")
    parser.add_argument('--size', type=int, default=320,
                        help="Target size for input image (default: 320)")
    parser.add_argument('--cpu', action='store_true',
                        help="Force CPU execution")
    args = parser.parse_args()
    
    try:
        # Select providers
        if args.cpu:
            providers = ['CPUExecutionProvider']
        else:
            providers = None  # Use default provider selection
            
        # Initialize runner
        runner = ONNXModelRunner(args.model, providers, args.device)
        
        # Run inference
        outputs, inference_time, _ = runner.run_inference(args.image, (args.size, args.size))
        
        # Print outputs shape
        for i, output in enumerate(outputs):
            print(f"Output {i} shape: {output.shape}")
        
        # Profiling results
        prof_file = runner.session.end_profiling()
        if os.path.exists(prof_file):
            print(f"Profiling data saved to: {prof_file}")
            
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())