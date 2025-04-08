#!/usr/bin/env python3
import os
import subprocess
import sys
from nudenet import NudeDetector

def print_banner(text):
    width = len(text) + 4
    print("=" * width)
    print(f"= {text} =")
    print("=" * width)

def print_section(title):
    print(f"\n{title}")
    print("-" * len(title))

def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {cmd}")
        print(f"Error message: {e.stderr}")
        return None

def check_gpu():
    print_banner("GPU ENVIRONMENT CHECK")
    
    print_section("System Information")
    run_command("uname -a | cat")
    
    print_section("CUDA Version")
    cuda_version = run_command("nvcc --version | grep 'release' | awk '{print $6}'")
    if cuda_version:
        print(f"CUDA Compiler: {cuda_version.strip()}")
    else:
        print("CUDA compiler not found or not in PATH")
    
    # Check CUDA runtime
    cuda_runtime = run_command("cat /usr/local/cuda/version.txt 2>/dev/null || echo 'CUDA runtime version file not found'")
    print(f"CUDA Runtime: {cuda_runtime.strip()}")
    
    print_section("GPU Information")
    gpu_info = run_command("nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader")
    if gpu_info:
        print(gpu_info)
    else:
        print("No NVIDIA GPU found or nvidia-smi not available")
    
    print_section("GPU Memory")
    run_command("nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv")
    
    print_section("GPU Processes")
    run_command("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv")
    
    print_section("ONNX Runtime Information")
    try:
        import onnxruntime
        print(f"ONNX Runtime version: {onnxruntime.__version__}")
        
        # Initialize detector to check providers
        print("\nInitializing NudeDetector to check providers...")
        detector = NudeDetector()
        available_providers = detector.onnx_session.get_providers()
        
        print(f"\nAvailable ONNX Runtime Providers: {available_providers}")
        
        if "CUDAExecutionProvider" in available_providers:
            print("\n✅ GPU acceleration is ENABLED")
            
            # Try to get CUDA device properties
            try:
                cuda_device_props = run_command("python3 -c \"import onnxruntime as ort; print(ort.get_device())\"")
                if cuda_device_props:
                    print(f"\nONNX Runtime CUDA Device: {cuda_device_props.strip()}")
            except Exception as e:
                print(f"Could not get CUDA device properties: {e}")
                
        else:
            print("\n❌ GPU acceleration is NOT ENABLED")
            print("Available providers are CPU only")
            print("\nPossible issues:")
            print("1. CUDA drivers not properly installed")
            print("2. ONNX Runtime built without CUDA support")
            print("3. GPU is not compatible with this CUDA version")
    
    except ImportError:
        print("ONNX Runtime not installed")
    except Exception as e:
        print(f"Error checking ONNX Runtime: {e}")
    
    print("\nCheck complete.")

if __name__ == "__main__":
    check_gpu()