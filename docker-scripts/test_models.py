#!/usr/bin/env python3
import os
import sys
import json
import time
import importlib.util
from nudenet import NudeDetector

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

def run_onnx_test(test_image_path):
    print_header("ONNX Runtime Model Test (320n)")
    
    try:
        print_section("Loading ONNX model")
        start_time = time.time()
        detector = NudeDetector()  # This loads the default 320n.onnx model
        load_time = time.time() - start_time
        print_result("Model loading time", f"{load_time:.4f} seconds", True)
        
        # Get providers
        providers = detector.onnx_session.get_providers()
        is_using_gpu = "CUDAExecutionProvider" in providers
        print_result("Providers", f"{providers}", True)
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
            
        return True
    
    except Exception as e:
        print_result("ONNX test", f"Error: {str(e)}", False)
        return False

def run_pytorch_test(test_image_path):
    print_header("PyTorch Model Test (320n)")
    
    # Check if PyTorch is available
    if importlib.util.find_spec("torch") is None:
        print_result("PyTorch availability", "PyTorch is not installed", False)
        return False
    
    try:
        # Import pytorch detector
        sys.path.append('/app/docker-scripts')
        from simple_pytorch_detector import SimpleYOLODetector
        
        # Model path
        model_path = "/app/models/pytorch/320n.pt"
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
        
        return True
    
    except Exception as e:
        print_result("PyTorch test", f"Error: {str(e)}", False)
        return False

def compare_results(test_image_path):
    print_header("Comparing ONNX vs PyTorch Results")
    
    try:
        # Run ONNX detection
        onnx_detector = NudeDetector()
        onnx_start = time.time()
        onnx_detections = onnx_detector.detect(test_image_path)
        onnx_time = time.time() - onnx_start
        
        # Import pytorch detector
        sys.path.append('/app/docker-scripts')
        from simple_pytorch_detector import SimpleYOLODetector
        
        # Run PyTorch detection
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pt_detector = SimpleYOLODetector("/app/models/pytorch/320n.pt", device)
        pt_start = time.time()
        pt_detections = pt_detector.detect(test_image_path)
        pt_time = time.time() - pt_start
        
        # Print results
        print_section("Performance Comparison")
        print(f"ONNX Runtime: {onnx_time:.4f} seconds, {len(onnx_detections)} detections")
        print(f"PyTorch: {pt_time:.4f} seconds, {len(pt_detections)} detections")
        
        if onnx_time < pt_time:
            print(f"\n{GREEN}ONNX Runtime is {pt_time/onnx_time:.2f}x faster{ENDC}")
        else:
            print(f"\n{GREEN}PyTorch is {onnx_time/pt_time:.2f}x faster{ENDC}")
        
        # Compare detection results
        print_section("Detection Comparison")
        
        # Sort detections by score for better comparison
        onnx_detections.sort(key=lambda x: (x["class"], -x["score"]))
        pt_detections.sort(key=lambda x: (x["class"], -x["score"]))
        
        print(f"{BOLD}ONNX Detections:{ENDC}")
        for detection in onnx_detections:
            print(f"  {detection['class']} (score: {detection['score']:.4f})")
            
        print(f"\n{BOLD}PyTorch Detections:{ENDC}")
        for detection in pt_detections:
            print(f"  {detection['class']} (score: {detection['score']:.4f})")
        
        # Count matches
        onnx_classes = {d["class"] for d in onnx_detections}
        pt_classes = {d["class"] for d in pt_detections}
        common_classes = onnx_classes.intersection(pt_classes)
        
        print(f"\n{BOLD}Common detections:{ENDC} {len(common_classes)} of {len(onnx_classes.union(pt_classes))}")
        for cls in common_classes:
            print(f"  {cls}")
        
        return True
    
    except Exception as e:
        print(f"{RED}Error comparing results: {str(e)}{ENDC}")
        return False

def main():
    # Default test image
    test_image_path = "/app/fastdeploy_recipe/cory_chase.jpeg"
    
    # Check if we have an argument for a different test image
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
    
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
    
    # Run the tests
    onnx_success = run_onnx_test(test_image_path)
    pytorch_success = run_pytorch_test(test_image_path)
    
    # If both tests succeeded, compare results
    if onnx_success and pytorch_success:
        compare_results(test_image_path)
    
    # Print final summary
    print_header("Test Summary")
    print_result("ONNX Test", "Completed successfully" if onnx_success else "Failed", onnx_success)
    print_result("PyTorch Test", "Completed successfully" if pytorch_success else "Failed", pytorch_success)
    
    return 0 if onnx_success and pytorch_success else 1

if __name__ == "__main__":
    sys.exit(main())