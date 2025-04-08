#!/usr/bin/env python3
import torch
import sys

def check_cuda():
    print("PyTorch CUDA Diagnostics")
    print("-----------------------")
    
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if not cuda_available:
        print("CUDA is not available. Possible reasons:")
        print("1. NVIDIA drivers are not properly installed")
        print("2. CUDA libraries are not properly installed")
        print("3. PyTorch was not compiled with CUDA support")
        print("4. Your GPU is not compatible with this CUDA version")
        return False
    
    # Get CUDA version
    print(f"CUDA version: {torch.version.cuda}")
    
    # Get device count
    device_count = torch.cuda.device_count()
    print(f"CUDA device count: {device_count}")
    
    # List all available devices
    print("\nAvailable GPUs:")
    for i in range(device_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  - Compute capability: {torch.cuda.get_device_capability(i)}")
        print(f"  - Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    # Test actual GPU functionality
    print("\nTesting GPU computation...")
    try:
        # Create a small tensor and move it to GPU
        x = torch.ones(10, 10, device='cuda')
        y = x + x
        
        # Run a small benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(1000):
            y = x + x
        end_event.record()
        
        # Wait for completion
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        
        print(f"GPU computation successful: {elapsed_time:.2f} ms for 1000 additions")
        print("CUDA is working correctly.")
        return True
    
    except Exception as e:
        print(f"Error testing GPU computation: {e}")
        return False

if __name__ == "__main__":
    success = check_cuda()
    sys.exit(0 if success else 1)