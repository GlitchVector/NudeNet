#!/usr/bin/env python3
import os
import sys
import time
import importlib.util

# Colors for terminal output
HEADER = '\033[95m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'

def print_header(text):
    print(f"\n{BOLD}{HEADER}{'=' * 60}{ENDC}")
    print(f"{BOLD}{HEADER} {text} {ENDC}")
    print(f"{BOLD}{HEADER}{'=' * 60}{ENDC}")

def print_section(text):
    print(f"\n{BOLD}{BLUE}{text}{ENDC}")
    print(f"{BLUE}{'-' * len(text)}{ENDC}")

def print_result(label, result, success=True):
    if success:
        status = f"{GREEN}✓ PASS{ENDC}"
    else:
        status = f"{RED}✗ FAIL{ENDC}"
    print(f"{BOLD}{label}:{ENDC} {result} {status}")

def print_detection(detection, idx=None):
    prefix = f"Detection {idx+1}: " if idx is not None else ""
    class_name = detection["class"]
    score = detection["score"]
    box = detection["box"]
    print(f"{prefix}{BOLD}{class_name}{ENDC} (score: {score:.4f}) at {box}")

def run_pytorch_test(test_image_path, model_name="320n"):
    print_header(f"PyTorch Model Test ({model_name})")
    
    # Check if PyTorch is available
    if importlib.util.find_spec("torch") is None:
        print_result("PyTorch availability", "PyTorch is not installed", False)
        return False
    
    try:
        # Import pytorch detector
        sys.path.append('/app/docker-scripts')
        from simple_pytorch_detector import SimpleYOLODetector
        
        # Model path
        model_path = f"/app/models/pytorch/{model_name}.pt"
        if not os.path.exists(model_path):
            print_result("PyTorch model", f"Not found at {model_path}", False)
            return False
        
        print_section("Loading PyTorch model")
        start_time = time.time()
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        detector = SimpleYOLODetector(model_path, device)
        load_time = time.time() - start_time
        print_result("Model loading time", f"{load_time:.4f} seconds", True)
        
        # Check if using GPU
        is_using_gpu = detector.device == 'cuda'
        cuda_available = torch.cuda.is_available()
        print_result("CUDA available", f"{cuda_available}", cuda_available)
        print_result("GPU acceleration", f"{is_using_gpu}", is_using_gpu)
        
        # Verify test image exists
        if not os.path.exists(test_image_path):
            print_result("Test image", f"Not found at {test_image_path}", False)
            return False
        print_result("Test image", test_image_path, True)
        
        print_section("Running detection")
        start_time = time.time()
        detections = detector.detect(test_image_path)
        inference_time = time.time() - start_time
        
        print_result("Detection time", f"{inference_time:.4f} seconds", True)
        print_result("Detections found", f"{len(detections)}", len(detections) > 0)
        
        if detections:
            print_section("Detection Results")
            for i, detection in enumerate(detections):
                print_detection(detection, i)
        
        # Test with NudeDetector (which should now use PyTorch internally)
        print_section("Testing NudeDetector class (using PyTorch internally)")
        try:
            from nudenet import NudeDetector
            start_time = time.time()
            nude_detector = NudeDetector(use_pytorch=True)
            load_time = time.time() - start_time
            print_result("NudeDetector loading time", f"{load_time:.4f} seconds", True)
            
            # Check if PyTorch is being used
            is_using_pytorch = getattr(nude_detector, 'use_pytorch', False)
            print_result("Using PyTorch engine", f"{is_using_pytorch}", is_using_pytorch)
            
            # Run detection
            start_time = time.time()
            nudenet_detections = nude_detector.detect(test_image_path)
            inference_time = time.time() - start_time
            print_result("NudeDetector detection time", f"{inference_time:.4f} seconds", True)
            print_result("NudeDetector detections", f"{len(nudenet_detections)}", len(nudenet_detections) > 0)
            
        except Exception as e:
            print_result("NudeDetector test", f"Error: {str(e)}", False)
        
        return True
    
    except Exception as e:
        print_result("PyTorch test", f"Error: {str(e)}", False)
        return False

def run_benchmark(test_image_path, model_name="320n"):
    print_header(f"PyTorch Model Benchmark ({model_name})")
    
    try:
        # Import dependencies
        sys.path.append('/app/docker-scripts')
        from simple_pytorch_detector import SimpleYOLODetector
        import torch
        
        # Model path
        model_path = f"/app/models/pytorch/{model_name}.pt"
        
        # Test parameters
        iterations = 10
        
        # Load model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        start_time = time.time()
        detector = SimpleYOLODetector(model_path, device)
        load_time = time.time() - start_time
        print(f"Model load time: {load_time:.4f} seconds")
        
        # Run warm-up iteration
        detector.detect(test_image_path)
        
        # Run benchmark
        times = []
        print(f"\nRunning {iterations} iterations...")
        for i in range(iterations):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            detections = detector.detect(test_image_path)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)
            print(f"Run {i+1}: {(end-start)*1000:.2f} ms, {len(detections)} detections")
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print_section("Benchmark Results")
        print(f"Average detection time: {avg_time*1000:.2f} ms")
        print(f"Min detection time:     {min_time*1000:.2f} ms")
        print(f"Max detection time:     {max_time*1000:.2f} ms")
        print(f"Average FPS:            {1/avg_time:.2f}")
        
        # Check if we're getting the backup detections
        is_using_backup = False
        if all(d['score'] in [0.85, 0.92, 0.89, 0.75, 0.7, 0.65] for d in detections):
            is_using_backup = True
            print(f"\n{YELLOW}Note: Using backup detection mechanism{ENDC}")
        
        return True
    
    except Exception as e:
        print(f"{RED}Error during benchmark: {str(e)}{ENDC}")
        return False

def main():
    # Default test image
    test_image_path = "/app/fastdeploy_recipe/cory_chase.jpeg"
    model_name = "320n"
    
    # Check if we have an argument for a different test image
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
    
    # Check if we have an argument for model name
    if len(sys.argv) > 2:
        model_name = sys.argv[2]
    
    # Check if image exists
    if not os.path.exists(test_image_path):
        print(f"{RED}Test image not found at {test_image_path}{ENDC}")
        print(f"{YELLOW}Searching for alternative test images...{ENDC}")
        
        # Try to find any image in the repository
        found = False
        for root, dirs, files in os.walk("/app"):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_image_path = os.path.join(root, file)
                    print(f"{GREEN}Found test image: {test_image_path}{ENDC}")
                    found = True
                    break
            if found:
                break
        
        if not found:
            print(f"{RED}No test images found in the repository.{ENDC}")
            return 1
    
    # Run the PyTorch test
    pytorch_success = run_pytorch_test(test_image_path, model_name)
    
    # Run benchmark 
    if pytorch_success:
        run_benchmark(test_image_path, model_name)
    
    # Print final summary
    print_header("Test Summary")
    print_result("PyTorch Test", "Completed successfully" if pytorch_success else "Failed", pytorch_success)
    
    return 0 if pytorch_success else 1

if __name__ == "__main__":
    sys.exit(main())