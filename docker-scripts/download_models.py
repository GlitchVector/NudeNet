#!/usr/bin/env python3
import os
import urllib.request
import hashlib
import sys

# Base URL for models
BASE_URL = "https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/"

# Model definitions
MODELS = [
    {
        "name": "320n.onnx",
        "url": f"{BASE_URL}320n.onnx",
        "size": "17.3 MB",
        "type": "onnx",
        "resolution": "320x320"
    },
    {
        "name": "320n.pt",
        "url": f"{BASE_URL}320n.pt",
        "size": "6.5 MB",
        "type": "pytorch",
        "resolution": "320x320"
    },
    {
        "name": "640m.onnx",
        "url": f"{BASE_URL}640m.onnx",
        "size": "49.5 MB",
        "type": "onnx",
        "resolution": "640x640"
    },
    {
        "name": "640m.pt",
        "url": f"{BASE_URL}640m.pt",
        "size": "25.6 MB",
        "type": "pytorch",
        "resolution": "640x640"
    }
]

# Create directory structure
def create_dirs():
    os.makedirs("/app/models/onnx", exist_ok=True)
    os.makedirs("/app/models/pytorch", exist_ok=True)

# Download progress hook
def download_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    progress = min(int(count * block_size), total_size)
    sys.stdout.write(f"\r{progress}/{total_size} bytes ({percent}%)")
    sys.stdout.flush()

# Download models
def download_models(force=False):
    create_dirs()
    
    print("=== Downloading NudeNet Models ===")
    print(f"{'Model':<15} {'Type':<10} {'Resolution':<12} {'Size':<10} {'Status':<10}")
    print("-" * 60)
    
    for model in MODELS:
        model_name = model["name"]
        model_type = model["type"]
        
        # Determine target directory and path
        if model_type == "onnx":
            target_dir = "/app/models/onnx"
        else:
            target_dir = "/app/models/pytorch"
            
        target_path = os.path.join(target_dir, model_name)
        
        # Check if model exists and its size
        should_download = force
        if os.path.exists(target_path):
            file_size = os.path.getsize(target_path)
            if file_size < 1000:  # File is too small, likely corrupted
                print(f"{model_name:<15} {model_type:<10} {model['resolution']:<12} {model['size']:<10} {'Corrupted - re-downloading':<30}")
                should_download = True
            elif not force:
                print(f"{model_name:<15} {model_type:<10} {model['resolution']:<12} {model['size']:<10} {'Already exists':<10}")
                continue
        else:
            should_download = True
            
        # Download model
        if should_download:
            try:
                print(f"{model_name:<15} {model_type:<10} {model['resolution']:<12} {model['size']:<10} {'Downloading...':<10}")
                urllib.request.urlretrieve(model["url"], target_path, download_progress)
                print(f"\n{model_name:<15} {model_type:<10} {model['resolution']:<12} {model['size']:<10} {'✓ Done':<10}")
                
                # Verify file size after download
                if os.path.getsize(target_path) < 1000:
                    print(f"Warning: Downloaded file {model_name} is suspiciously small. It may be corrupted.")
                
                # Create symlink for default model if it's 320n.onnx
                if model_name == "320n.onnx":
                    nudenet_dir = "/app/nudenet"
                    if os.path.exists(nudenet_dir):
                        symlink_path = os.path.join(nudenet_dir, "320n.onnx")
                        # Only create symlink if original exists but symlink doesn't
                        if os.path.exists(target_path) and not os.path.exists(symlink_path):
                            print(f"Creating symlink for default model at {symlink_path}")
                            # If file exists but isn't a symlink, rename it first
                            if os.path.isfile(symlink_path) and not os.path.islink(symlink_path):
                                os.rename(symlink_path, f"{symlink_path}.original")
                                print(f"Renamed existing file to {symlink_path}.original")
                            # Create relative symlink
                            os.symlink(os.path.relpath(target_path, nudenet_dir), symlink_path)
                
            except Exception as e:
                print(f"\nError downloading {model_name}: {str(e)}")
                # If download fails, attempt to use a direct GitHub URL as a backup
                try:
                    print("Retrying with direct GitHub URL...")
                    direct_url = f"https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/{model_name}"
                    urllib.request.urlretrieve(direct_url, target_path, download_progress)
                    print(f"\n{model_name:<15} {model_type:<10} {model['resolution']:<12} {model['size']:<10} {'✓ Done (backup URL)':<20}")
                except Exception as e2:
                    print(f"\nError with backup download: {str(e2)}")
    
    print("\nDownload complete!")

if __name__ == "__main__":
    force = False
    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        force = True
        print("Force re-download enabled")
    download_models(force)