#!/usr/bin/env python3
"""
Unified GPU diagnostics script for NudeNet
Combines the functionality of multiple diagnostic scripts
"""
import os
import sys
import subprocess
import time
import argparse
from pathlib import Path
import importlib.util

# Color codes for pretty printing
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'

def print_section_header(title):
    """Print a formatted section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 50}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {title}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 50}{Colors.ENDC}")

def print_result(message, success):
    """Print a formatted result message"""
    status = f"{Colors.GREEN}[SUCCESS]{Colors.ENDC}" if success else f"{Colors.RED}[FAILED]{Colors.ENDC}"
    print(f"{status} {message}")

def print_info(message):
    """Print a formatted info message"""
    print(f"{Colors.BLUE}[INFO]{Colors.ENDC} {message}")

def run_command(cmd, print_output=True):
    """Run a shell command and return the output"""
    try:
        result = subprocess.run(cmd, shell=True, check=False, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
        if print_output:
            print(result.stdout)
        return result.stdout.strip(), result.returncode == 0
    except Exception as e:
        print(f"{Colors.RED}Error executing command: {e}{Colors.ENDC}")
        return str(e), False

def check_torch_cuda():
    """Check CUDA availability with PyTorch"""
    print_section_header("PyTorch CUDA Check")
    
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"Device {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Capability: {torch.cuda.get_device_capability(i)}")
                
            # Test tensor creation on GPU
            try:
                x = torch.rand(5, 5).cuda()
                y = torch.rand(5, 5).cuda()
                z = x + y
                print_result("GPU tensor test", True)
            except Exception as e:
                print_result(f"GPU tensor test failed: {e}", False)
        else:
            print(f"{Colors.YELLOW}CUDA not available. Check NVIDIA drivers and CUDA installation.{Colors.ENDC}")
            print("Possible reasons:")
            print("1. NVIDIA drivers are not properly installed")
            print("2. CUDA libraries are not properly installed")
            print("3. PyTorch was not compiled with CUDA support")
            print("4. Your GPU is not compatible with this CUDA version")
            
        return cuda_available
    except ImportError:
        print(f"{Colors.RED}PyTorch not installed - cannot check CUDA support{Colors.ENDC}")
        return False
    except Exception as e:
        print(f"{Colors.RED}Error checking PyTorch CUDA: {e}{Colors.ENDC}")
        return False

def check_onnx_providers():
    """Check ONNX Runtime providers"""
    print_section_header("ONNX Runtime Providers Check")
    
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime version: {ort.__version__}")
        providers = ort.get_available_providers()
        print("Available providers:")
        for idx, provider in enumerate(providers):
            print(f"  {idx+1}. {provider}")
        
        gpu_available = 'CUDAExecutionProvider' in providers
        print_result("CUDA execution provider available", gpu_available)
        
        return gpu_available
    except ImportError:
        print(f"{Colors.RED}ONNX Runtime not installed - cannot check providers{Colors.ENDC}")
        return False
    except Exception as e:
        print(f"{Colors.RED}Error checking ONNX Runtime: {e}{Colors.ENDC}")
        return False

def check_system_nvidia():
    """Check system NVIDIA setup (drivers, CUDA)"""
    print_section_header("System NVIDIA Setup")
    
    print_info("Checking NVIDIA drivers...")
    nvidia_smi, success = run_command("nvidia-smi")
    print_result("NVIDIA drivers installed and working", success)
    
    print_info("Checking CUDA installation...")
    nvcc_version, success = run_command("nvcc --version", False)
    if success:
        print(nvcc_version)
    print_result("NVCC (CUDA compiler) available", success)
    
    print_info("Checking GPU libraries...")
    libs, _ = run_command("ldconfig -p | grep -E 'libcuda|libnvidia|libcudnn'", False)
    if libs:
        print("Found GPU-related libraries:")
        print(libs)
        print_result("GPU libraries found", True)
    else:
        print_result("No GPU libraries found", False)
    
    return success

def check_detector_gpu():
    """Test NudeDetector with GPU"""
    print_section_header("NudeDetector GPU Test")
    
    try:
        start_time = time.time()
        from nudenet import NudeDetector
        print(f"NudeDetector imported in {time.time() - start_time:.2f}s")
        
        # Try to init with PyTorch (GPU)
        start_time = time.time()
        detector = NudeDetector(use_pytorch=True)
        init_time = time.time() - start_time
        print(f"Detector initialized in {init_time:.2f}s")
        
        # Check if using PyTorch
        is_using_pytorch = getattr(detector, 'use_pytorch', False)
        print(f"Using PyTorch: {is_using_pytorch}")
        
        # Find a test image
        test_image = None
        if os.path.exists("/app/fastdeploy_recipe/cory_chase.jpeg"):
            test_image = "/app/fastdeploy_recipe/cory_chase.jpeg"
        else:
            # Try to find any image
            for ext in ['.jpg', '.jpeg', '.png']:
                for path in ['/app', os.path.expanduser('~')]:
                    if test_image:
                        break
                    for root, _, files in os.walk(path, topdown=True, followlinks=False):
                        if root.startswith('/app/models'):  # Skip model directories
                            continue
                        for filename in files:
                            if filename.lower().endswith(ext):
                                test_image = os.path.join(root, filename)
                                break
                        if test_image:
                            break
                        
        if test_image:
            print(f"Using test image: {test_image}")
            start_time = time.time()
            result = detector.detect(test_image)
            inference_time = time.time() - start_time
            
            print(f"Detection completed in {inference_time:.2f}s with {len(result)} detections")
            print_result("NudeDetector working with GPU acceleration", inference_time < 1.0)
            
            return True
        else:
            print(f"{Colors.YELLOW}No test image found - skipping detector test{Colors.ENDC}")
            return False
    except ImportError:
        print(f"{Colors.RED}NudeDetector not installed - cannot test GPU detection{Colors.ENDC}")
        return False
    except Exception as e:
        print(f"{Colors.RED}Error testing NudeDetector: {e}{Colors.ENDC}")
        return False

def test_pytorch_detector():
    """Test PyTorch detector"""
    print_section_header("PyTorch Detector Test")
    
    try:
        # Import the PyTorch detector
        detector_path = "/app/docker-scripts/pytorch_detector.py"
        if not os.path.exists(detector_path):
            print(f"{Colors.RED}PyTorch detector not found at {detector_path}{Colors.ENDC}")
            return False
            
        import_name = "pytorch_detector"
        spec = importlib.util.spec_from_file_location(import_name, detector_path)
        detector_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(detector_module)
        
        # Find a model
        model_path = None
        for path in ["/app/models/pytorch/320n.pt", "/app/docker-scripts/pytorch-models/320n.pt"]:
            if os.path.exists(path):
                model_path = path
                break
                
        if not model_path:
            print(f"{Colors.RED}No PyTorch model found{Colors.ENDC}")
            return False
            
        print(f"Using model: {model_path}")
        
        # Create detector
        start_time = time.time()
        detector = detector_module.SimpleYOLODetector(model_path)
        init_time = time.time() - start_time
        print(f"Detector initialized in {init_time:.2f}s")
        
        # Find a test image
        test_image = None
        if os.path.exists("/app/fastdeploy_recipe/cory_chase.jpeg"):
            test_image = "/app/fastdeploy_recipe/cory_chase.jpeg"
            
        if test_image:
            print(f"Using test image: {test_image}")
            start_time = time.time()
            result = detector.detect(test_image)
            inference_time = time.time() - start_time
            
            print(f"Detection completed in {inference_time:.2f}s with {len(result)} detections")
            print_result("PyTorch detector working with GPU acceleration", inference_time < 1.0)
            
            return True
        else:
            print(f"{Colors.YELLOW}No test image found - skipping detector test{Colors.ENDC}")
            return False
    except Exception as e:
        print(f"{Colors.RED}Error testing PyTorch detector: {e}{Colors.ENDC}")
        return False

def main():
    parser = argparse.ArgumentParser(description="GPU Diagnostics Tool")
    parser.add_argument('--mode', type=str, default='all',
                      choices=['all', 'torch', 'onnx', 'system', 'detector', 'pytorch'],
                      help='Which diagnostic mode to run')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    
    args = parser.parse_args()
    
    print(f"{Colors.BOLD}{Colors.GREEN}NudeNet GPU Diagnostics Tool{Colors.ENDC}")
    print(f"Running in {args.mode} mode\n")
    
    results = {}
    
    if args.mode in ['all', 'torch']:
        results['torch'] = check_torch_cuda()
        
    if args.mode in ['all', 'onnx']:
        results['onnx'] = check_onnx_providers()
        
    if args.mode in ['all', 'system']:
        results['system'] = check_system_nvidia()
        
    if args.mode in ['all', 'detector']:
        results['detector'] = check_detector_gpu()
        
    if args.mode in ['all', 'pytorch']:
        results['pytorch'] = test_pytorch_detector()
    
    # Print summary
    print_section_header("Diagnostics Summary")
    for test, result in results.items():
        status = f"{Colors.GREEN}[PASS]{Colors.ENDC}" if result else f"{Colors.RED}[FAIL]{Colors.ENDC}"
        print(f"{status} {test.capitalize()} test")
    
    # Overall result
    if all(results.values()):
        print(f"\n{Colors.GREEN}All tests passed! GPU acceleration should be working correctly.{Colors.ENDC}")
    else:
        print(f"\n{Colors.YELLOW}Some tests failed. GPU acceleration may not be fully functional.{Colors.ENDC}")
        failed_tests = [test for test, result in results.items() if not result]
        print(f"Failed tests: {', '.join(failed_tests)}")
        
        if 'system' in failed_tests:
            print(f"\n{Colors.YELLOW}Recommendation: Check NVIDIA drivers and CUDA installation{Colors.ENDC}")
        if 'torch' in failed_tests:
            print(f"\n{Colors.YELLOW}Recommendation: Verify PyTorch CUDA support{Colors.ENDC}")
        if 'onnx' in failed_tests:
            print(f"\n{Colors.YELLOW}Recommendation: Verify ONNX Runtime GPU providers{Colors.ENDC}")
    
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())