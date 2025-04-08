#!/usr/bin/env python3
import os
import sys
import time
import importlib.util
import logging

# Set up colorful output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_section_header(title):
    """Print a formatted section header"""
    width = len(title) + 4
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * width}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}= {title} ={Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * width}{Colors.ENDC}")

def print_subsection(title):
    """Print a formatted subsection"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{title}{Colors.ENDC}")
    print(f"{Colors.BLUE}{'-' * len(title)}{Colors.ENDC}")

def print_status(message, success=True):
    """Print a status message with color indicator"""
    status = f"{Colors.GREEN}✓ PASS{Colors.ENDC}" if success else f"{Colors.RED}✗ FAIL{Colors.ENDC}"
    print(f"{Colors.BOLD}{message}:{Colors.ENDC} {status}")

def check_onnx_runtime():
    """Check ONNX Runtime GPU support"""
    print_section_header("ONNX RUNTIME GPU CHECK")
    
    try:
        import onnxruntime
        from onnxruntime.capi import _pybind_state as C
        
        print_subsection("ONNX Runtime Information")
        print(f"ONNX Runtime version: {onnxruntime.__version__}")
        
        # Get available providers
        available_providers = C.get_available_providers()
        print(f"\nAvailable providers: {available_providers}")
        
        # Check if CUDA provider is available
        is_cuda_available = "CUDAExecutionProvider" in available_providers
        is_tensorrt_available = "TensorrtExecutionProvider" in available_providers
        
        print_status("CUDA provider available", is_cuda_available)
        print_status("TensorRT provider available", is_tensorrt_available)
        print_status("GPU acceleration available", is_cuda_available or is_tensorrt_available)
        
        if not is_cuda_available:
            print(f"\n{Colors.YELLOW}Possible reasons for missing CUDA provider:{Colors.ENDC}")
            print("1. CUDA is not installed or not detected")
            print("2. ONNX Runtime was not built with CUDA support")
            print("3. The CUDA version is incompatible with ONNX Runtime")
            print("4. Required CUDA libraries are not in the system path")
        
        # Try to load a model to test actual provider usage
        print_subsection("Testing ONNX Runtime Model Loading")
        try:
            from nudenet import NudeDetector
            
            print("Initializing NudeDetector...")
            start_time = time.time()
            detector = NudeDetector()
            init_time = time.time() - start_time
            
            actual_providers = detector.onnx_session.get_providers()
            print(f"Initialization time: {init_time:.4f} seconds")
            print(f"Active providers: {actual_providers}")
            
            is_using_cuda = "CUDAExecutionProvider" in actual_providers
            print_status("Using CUDA for inference", is_using_cuda)
            
            return is_using_cuda
            
        except Exception as e:
            print(f"{Colors.RED}Error loading ONNX model: {str(e)}{Colors.ENDC}")
            return False
        
    except ImportError as e:
        print(f"{Colors.RED}Error: ONNX Runtime not found - {str(e)}{Colors.ENDC}")
        return False
    except Exception as e:
        print(f"{Colors.RED}Error checking ONNX Runtime: {str(e)}{Colors.ENDC}")
        return False

def check_pytorch():
    """Check PyTorch GPU support"""
    print_section_header("PYTORCH GPU CHECK")
    
    try:
        import torch
        
        print_subsection("PyTorch Information")
        print(f"PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print_status("CUDA available", cuda_available)
        
        if not cuda_available:
            print(f"\n{Colors.YELLOW}Possible reasons for CUDA unavailability:{Colors.ENDC}")
            print("1. NVIDIA drivers are not properly installed")
            print("2. CUDA libraries are not properly installed")
            print("3. PyTorch was not compiled with CUDA support")
            print("4. Your GPU is not compatible with this CUDA version")
            return False
        
        # Get CUDA version and device info
        print(f"CUDA version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"CUDA device count: {device_count}")
        
        # List all available devices
        print_subsection("Available GPUs")
        for i in range(device_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  - Compute capability: {torch.cuda.get_device_capability(i)}")
            print(f"  - Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        # Test actual GPU functionality
        print_subsection("Testing GPU Computation")
        try:
            # Create a small tensor and move it to GPU
            start_time = time.time()
            x = torch.ones(1000, 1000, device='cuda')
            y = x + x
            torch.cuda.synchronize()  # Wait for the operation to complete
            init_time = time.time() - start_time
            
            print(f"Tensor creation and addition time: {init_time:.4f} seconds")
            print_status("Basic GPU computation", True)
            
            # Test loading a model if available
            if os.path.exists("/app/models/pytorch/320n.pt"):
                print_subsection("Testing PyTorch Model Loading")
                try:
                    sys.path.append('/app/docker-scripts')
                    from pytorch_detector import PyTorchNudeDetector
                    
                    model_path = "/app/models/pytorch/320n.pt"
                    print(f"Loading model from {model_path}...")
                    
                    start_time = time.time()
                    detector = PyTorchNudeDetector(model_path, 320)
                    load_time = time.time() - start_time
                    
                    print(f"Model loading time: {load_time:.4f} seconds")
                    print_status("Model loaded successfully", True)
                    
                except Exception as e:
                    print(f"{Colors.RED}Error loading PyTorch model: {str(e)}{Colors.ENDC}")
            
            return True
            
        except Exception as e:
            print(f"{Colors.RED}Error in GPU computation test: {str(e)}{Colors.ENDC}")
            return False
            
    except ImportError:
        print(f"{Colors.RED}Error: PyTorch not found{Colors.ENDC}")
        return False
    except Exception as e:
        print(f"{Colors.RED}Error checking PyTorch: {str(e)}{Colors.ENDC}")
        return False

def check_system_gpu():
    """Check system GPU information"""
    print_section_header("SYSTEM GPU INFORMATION")
    
    try:
        import subprocess
        
        # Check NVIDIA driver version
        try:
            driver_version = subprocess.check_output(
                "nvidia-smi --query-gpu=driver_version --format=csv,noheader", 
                shell=True
            ).decode().strip()
            print(f"NVIDIA driver version: {driver_version}")
        except:
            print(f"{Colors.RED}NVIDIA driver not found or nvidia-smi not available{Colors.ENDC}")
        
        # Check GPU information
        try:
            gpu_info = subprocess.check_output(
                "nvidia-smi --query-gpu=name,memory.total,compute_capability --format=csv,noheader",
                shell=True
            ).decode().strip()
            print("\nGPU Information:")
            print(gpu_info)
        except:
            print(f"{Colors.RED}Could not retrieve GPU information{Colors.ENDC}")
        
        # Show GPU processes
        try:
            processes = subprocess.check_output(
                "nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv",
                shell=True
            ).decode().strip()
            print("\nGPU Processes:")
            print(processes)
        except:
            print(f"{Colors.RED}Could not retrieve GPU processes{Colors.ENDC}")
            
        return True
    except Exception as e:
        print(f"{Colors.RED}Error checking system GPU: {str(e)}{Colors.ENDC}")
        return False

def main():
    """Run the complete GPU check"""
    print_section_header("GPU AVAILABILITY CHECK")
    print(f"{Colors.YELLOW}Checking GPU capabilities with multiple frameworks...{Colors.ENDC}")
    
    # Check system GPU
    system_gpu_ok = check_system_gpu()
    
    # Space between sections
    print("\n")
    
    # Check ONNX Runtime
    onnx_gpu_ok = check_onnx_runtime()
    
    # Space between sections
    print("\n")
    
    # Check PyTorch
    pytorch_gpu_ok = check_pytorch()
    
    # Print summary
    print_section_header("GPU CHECK SUMMARY")
    print_status("System GPU detected", system_gpu_ok)
    print_status("ONNX Runtime GPU acceleration", onnx_gpu_ok)
    print_status("PyTorch GPU acceleration", pytorch_gpu_ok)
    
    if pytorch_gpu_ok and not onnx_gpu_ok:
        print(f"\n{Colors.YELLOW}RECOMMENDATION:{Colors.ENDC}")
        print("CUDA appears to be working with PyTorch but not with ONNX Runtime.")
        print("Consider using the PyTorch implementation for GPU acceleration:")
        print("  docker run --gpus all -it nudenet-gpu pytorch 320n")
    elif onnx_gpu_ok and not pytorch_gpu_ok:
        print(f"\n{Colors.YELLOW}RECOMMENDATION:{Colors.ENDC}")
        print("CUDA appears to be working with ONNX Runtime but not with PyTorch.")
        print("If you're experiencing issues with PyTorch model loading, try:")
        print("  docker run --gpus all -it nudenet-gpu fix-models")
        print("  docker run --gpus all -it nudenet-gpu onnx 320n")
    elif not pytorch_gpu_ok and not onnx_gpu_ok:
        print(f"\n{Colors.YELLOW}RECOMMENDATION:{Colors.ENDC}")
        print("Neither PyTorch nor ONNX Runtime detected GPU acceleration.")
        print("Try fixing your models and using the direct ONNX runner:")
        print("  docker run --gpus all -it nudenet-gpu fix-models")
        print("  docker run --gpus all -it nudenet-gpu onnx 320n")
        
    return 0 if (pytorch_gpu_ok or onnx_gpu_ok) else 1

if __name__ == "__main__":
    sys.exit(main())