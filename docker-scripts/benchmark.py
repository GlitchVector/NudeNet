#!/usr/bin/env python3
import time
import os
import argparse
import importlib.util
import sys
from nudenet import NudeDetector

def run_benchmark(args):
    # Find test image
    test_image = args.image
    if not os.path.exists(test_image):
        print(f"Warning: Test image not found at {test_image}")
        print("Searching for any image file to use...")
        
        # Try to find an image file
        for root, dirs, files in os.walk("/app"):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_image = os.path.join(root, file)
                    print(f"Found image: {test_image}")
                    break
            if test_image != args.image:
                break
    
    print(f"Using test image: {test_image}")
    
    # ONNX runtime benchmark
    if not args.pytorch_only:
        run_onnx_benchmark(test_image, args)
    
    # PyTorch benchmark if available
    if args.pytorch and not args.onnx_only:
        run_pytorch_benchmark(test_image, args)

def run_onnx_benchmark(test_image, args):
    print("\n" + "="*50)
    print("ONNX RUNTIME BENCHMARK")
    print("="*50)
    
    # Initialize ONNX model
    if args.model_640:
        model_path = "/app/models/onnx/640m.onnx"
        if os.path.exists(model_path):
            print(f"Using 640m ONNX model: {model_path}")
            detector = NudeDetector(model_path=model_path, inference_resolution=640)
        else:
            print(f"640m model not found at {model_path}, using default model")
            detector = NudeDetector()
    else:
        print("Using default 320n ONNX model")
        detector = NudeDetector()
    
    # Get ONNX runtime providers
    providers = detector.onnx_session.get_providers()
    print(f"ONNX Runtime Providers: {providers}")
    is_using_gpu = "CUDAExecutionProvider" in providers
    print(f"Using GPU acceleration: {is_using_gpu}")
    
    # Warmup phase
    print("\nWarming up...")
    for i in range(5):
        detector.detect(test_image)
    
    # Single image benchmark
    print("\nSingle image benchmark:")
    iterations = args.iterations
    start = time.time()
    for i in range(iterations):
        detector.detect(test_image)
    duration = time.time() - start
    
    print(f"Processed {iterations} images in {duration:.2f} seconds")
    print(f"Average: {(duration/iterations)*1000:.2f} ms per image")
    print(f"Throughput: {iterations/duration:.2f} images per second")
    
    # Batch processing benchmark
    print("\nBatch processing benchmark:")
    batch_size = args.batch_size
    iterations = args.batch_iterations
    batch_images = [test_image] * batch_size
    
    start = time.time()
    for i in range(iterations):
        detector.detect_batch(batch_images)
    duration = time.time() - start
    total_images = iterations * batch_size
    
    print(f"Processed {total_images} images in {duration:.2f} seconds (batch size: {batch_size})")
    print(f"Average: {(duration/total_images)*1000:.2f} ms per image")
    print(f"Throughput: {total_images/duration:.2f} images per second")
    
    # Get memory usage if on GPU
    if is_using_gpu:
        try:
            import subprocess
            print("\nGPU Memory Status:")
            subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv"])
        except Exception as e:
            print(f"Failed to get GPU memory status: {e}")

def run_pytorch_benchmark(test_image, args):
    # Check if PyTorch is available
    if importlib.util.find_spec("torch") is None:
        print("\nPyTorch is not installed. Skipping PyTorch benchmark.")
        return
    
    try:
        print("\n" + "="*50)
        print("PYTORCH BENCHMARK")
        print("="*50)
        
        # Import PyTorch detector
        sys.path.append('/app/docker-scripts')
        from pytorch_detector import PyTorchNudeDetector
        
        # Initialize PyTorch model
        if args.model_640:
            model_path = "/app/models/pytorch/640m.pt"
            resolution = 640
        else:
            model_path = "/app/models/pytorch/320n.pt"
            resolution = 320
            
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"PyTorch model not found at {model_path}")
            print("Please run 'download-models' command first")
            return
            
        print(f"Using PyTorch model: {model_path}")
        detector = PyTorchNudeDetector(model_path, resolution)
        
        # Warmup phase
        print("\nWarming up...")
        for i in range(5):
            detector.detect(test_image)
        
        # Single image benchmark
        print("\nSingle image benchmark:")
        iterations = args.iterations
        start = time.time()
        for i in range(iterations):
            detector.detect(test_image)
        duration = time.time() - start
        
        print(f"Processed {iterations} images in {duration:.2f} seconds")
        print(f"Average: {(duration/iterations)*1000:.2f} ms per image")
        print(f"Throughput: {iterations/duration:.2f} images per second")
        
        # Get GPU memory status
        if detector.device == 'cuda':
            try:
                import torch
                print("\nPyTorch GPU Memory Status:")
                print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
                print(f"Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
                
                import subprocess
                print("\nNVIDIA-SMI GPU Memory Status:")
                subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv"])
            except Exception as e:
                print(f"Failed to get GPU memory status: {e}")
                
    except Exception as e:
        print(f"Error running PyTorch benchmark: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NudeNet Benchmark Tool")
    parser.add_argument('--image', type=str, default="/app/fastdeploy_recipe/cory_chase.jpeg",
                        help='Path to test image')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of iterations for single image benchmark')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for batch processing benchmark')
    parser.add_argument('--batch-iterations', type=int, default=10,
                        help='Number of iterations for batch processing benchmark')
    parser.add_argument('--model-640', action='store_true',
                        help='Use 640m model instead of 320n')
    parser.add_argument('--pytorch', action='store_true',
                        help='Also run PyTorch benchmark')
    parser.add_argument('--pytorch-only', action='store_true',
                        help='Run only PyTorch benchmark')
    parser.add_argument('--onnx-only', action='store_true',
                        help='Run only ONNX Runtime benchmark')
    
    args = parser.parse_args()
    run_benchmark(args)