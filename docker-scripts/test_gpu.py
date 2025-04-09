#!/usr/bin/env python3
"""
NudeNet GPU Test Script
-----------------------
This script tests the GPU capabilities of the NudeNet Docker image
by running detection on a test image and measuring performance.
"""

import os
import sys
import time
import json
import torch
import numpy as np
from nudenet import NudeDetector

def format_time(seconds):
    """Format time in seconds to a readable string"""
    if seconds < 0.001:
        return f"{seconds * 1000000:.2f} µs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    else:
        return f"{seconds:.4f} s"

def test_gpu():
    """Test GPU availability and capabilities"""
    print("\n===== GPU AVAILABILITY TEST =====")
    
    # Check for CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("WARNING: CUDA is not available! The container will run in CPU mode.")
        return False
    
    # Get device properties
    device_count = torch.cuda.device_count()
    print(f"Number of CUDA devices: {device_count}")
    
    # Display info for each GPU
    for i in range(device_count):
        device_props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {device_props.name}")
        print(f"  Total memory: {device_props.total_memory / (1024**3):.2f} GB")
        print(f"  CUDA Capability: {device_props.major}.{device_props.minor}")
    
    # Check current device
    current_device = torch.cuda.current_device()
    print(f"\nCurrent device: {torch.cuda.get_device_name(current_device)}")
    
    # Create a small tensor and run a simple operation to verify GPU operation
    print("\nRunning test operation on GPU...")
    x = torch.rand(1000, 1000).cuda()
    start_time = time.time()
    y = torch.matmul(x, x)
    end_time = time.time()
    
    print(f"Matrix multiplication time: {format_time(end_time - start_time)}")
    print("GPU test completed successfully!\n")
    
    return True

def test_nudenet_detection(use_gpu):
    """Test NudeNet detection on the test image"""
    test_image = "/app/docker-scripts/cory_chase.jpeg"
    if not os.path.exists(test_image):
        print(f"Error: Test image not found at {test_image}")
        return False
    
    print("===== NUDENET DETECTION TEST =====")
    
    # Test with default model (320n)
    print("\nTesting with default model (320n)...")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    
    try:
        start_time = time.time()
        detector = NudeDetector(providers=providers)
        init_time = time.time() - start_time
        print(f"Initialization time: {format_time(init_time)}")
        
        # Run detection
        start_time = time.time()
        results = detector.detect(test_image)
        detection_time = time.time() - start_time
        
        print(f"Detection time: {format_time(detection_time)}")
        print(f"Found {len(results)} detections:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['class']} (Score: {result['score']:.4f})")
        
        # Test with 640m model if available
        model_640m = "/app/models/640m.pt"
        if os.path.exists(model_640m):
            print("\nTesting with 640m model...")
            try:
                onnx_model = "/app/models/640m.onnx"
                if not os.path.exists(onnx_model):
                    # If we have the PT file but not ONNX, we would need to convert
                    # This would require ultralytics to be installed
                    print("Note: 640m.onnx not found, can't test directly.")
                else:
                    start_time = time.time()
                    detector_640 = NudeDetector(model_path=onnx_model, inference_resolution=640, providers=providers)
                    init_time = time.time() - start_time
                    print(f"Initialization time: {format_time(init_time)}")
                    
                    start_time = time.time()
                    results_640 = detector_640.detect(test_image)
                    detection_time = time.time() - start_time
                    
                    print(f"Detection time: {format_time(detection_time)}")
                    print(f"Found {len(results_640)} detections:")
                    for i, result in enumerate(results_640, 1):
                        print(f"  {i}. {result['class']} (Score: {result['score']:.4f})")
            except Exception as e:
                print(f"Error testing 640m model: {str(e)}")
        else:
            print("\nNote: 640m model not found, skipping test.")
            
        return True
        
    except Exception as e:
        print(f"Error during NudeNet detection test: {str(e)}")
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("NUDENET GPU TEST SUITE")
    print("=" * 50)
    
    # Get system info
    print("\nSystem Information:")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Test GPU
    gpu_available = test_gpu()
    
    # Test NudeNet detection
    nudenet_success = test_nudenet_detection(gpu_available)
    
    # Overall result
    print("\n" + "=" * 50)
    if gpu_available and nudenet_success:
        print("✅ All tests passed successfully!")
    elif not gpu_available and nudenet_success:
        print("⚠️ Tests completed with CPU only. GPU not available.")
    else:
        print("❌ Some tests failed. Please check the logs.")
    print("=" * 50)
    
    return 0 if nudenet_success else 1

if __name__ == "__main__":
    sys.exit(main())