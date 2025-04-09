#!/usr/bin/env python3
"""
Model Manager for NudeNet
Handles downloading, verification, and management of model files
"""
import os
import sys
import argparse
import urllib.request
import hashlib
import shutil
import json
import time
from pathlib import Path

# Base model directories
MODELS_DIR = "/app/models" if os.path.exists("/app") else os.path.join(os.path.dirname(__file__), "models")
ONNX_DIR = os.path.join(MODELS_DIR, "onnx")
PYTORCH_DIR = os.path.join(MODELS_DIR, "pytorch")

# Base URL for models - direct from official source
BASE_URL = "https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/"

# Model definitions
MODELS = [
    {
        "name": "320n.onnx",
        "url": f"{BASE_URL}320n.onnx",
        "size": "17.3 MB",
        "type": "onnx",
        "resolution": "320x320",
        "description": "Smaller, faster model (320x320)",
        "output_path": os.path.join(ONNX_DIR, "320n.onnx"),
        "md5": "6715c42f99ce4b2c27f82fcdb175ab1a"
    },
    {
        "name": "320n.pt",
        "url": f"{BASE_URL}320n.pt",
        "size": "6.2 MB",
        "type": "pytorch",
        "resolution": "320x320",
        "description": "Smaller, faster model (320x320) in PyTorch format",
        "output_path": os.path.join(PYTORCH_DIR, "320n.pt"),
        "md5": "df4a8d7fc0bb7a7f4de1db3e8e03b8d9"
    },
    {
        "name": "640m.onnx",
        "url": f"{BASE_URL}640m.onnx",
        "size": "51.9 MB",
        "type": "onnx",
        "resolution": "640x640",
        "description": "Larger, more accurate model (640x640)",
        "output_path": os.path.join(ONNX_DIR, "640m.onnx"),
        "md5": "aacc2ae0be4f85677b80be2a199c5c3d"
    },
    {
        "name": "640m.pt",
        "url": f"{BASE_URL}640m.pt",
        "size": "18.3 MB",
        "type": "pytorch",
        "resolution": "640x640",
        "description": "Larger, more accurate model (640x640) in PyTorch format",
        "output_path": os.path.join(PYTORCH_DIR, "640m.pt"),
        "md5": "ce0c5b3d7f5f45d87deb65c9126ad4c5"
    }
]

def create_directories():
    """Create model directories if they don't exist"""
    os.makedirs(ONNX_DIR, exist_ok=True)
    os.makedirs(PYTORCH_DIR, exist_ok=True)
    print(f"Model directories created/verified: {ONNX_DIR}, {PYTORCH_DIR}")

def calculate_md5(filename):
    """Calculate the MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    try:
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error calculating MD5 for {filename}: {e}")
        return None

def download_file(url, output_path, expected_md5=None):
    """Download a file with progress reporting and optional MD5 verification"""
    try:
        if os.path.exists(output_path):
            print(f"File already exists: {output_path}")
            if expected_md5:
                file_md5 = calculate_md5(output_path)
                if file_md5 == expected_md5:
                    print(f"MD5 verified ✅: {file_md5}")
                    return True
                else:
                    print(f"MD5 mismatch ❌: expected {expected_md5}, got {file_md5}")
                    print(f"Re-downloading file...")
            else:
                return True
        
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"Downloading {url} to {output_path}...")
        
        # Simple progress reporting
        def report_progress(count, block_size, total_size):
            percent = min(100, count * block_size * 100 // total_size)
            sys.stdout.write(f"\rProgress: {percent}% {count*block_size}/{total_size} bytes")
            sys.stdout.flush()
        
        # Download with progress
        urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
        print("")  # New line after progress
        
        # Verify MD5 if provided
        if expected_md5:
            file_md5 = calculate_md5(output_path)
            if file_md5 == expected_md5:
                print(f"MD5 verified ✅: {file_md5}")
            else:
                print(f"MD5 mismatch ❌: expected {expected_md5}, got {file_md5}")
                return False
        
        print(f"Download completed: {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def download_models(model_types=None, force=False):
    """Download all or specific types of models"""
    create_directories()
    
    if model_types is None:
        model_types = ["onnx", "pytorch"]
    
    success_count = 0
    failure_count = 0
    
    for model in MODELS:
        if model["type"] in model_types:
            output_path = model["output_path"]
            
            # Skip if file exists and not forcing download
            if os.path.exists(output_path) and not force:
                print(f"Model already exists: {output_path}")
                # Verify MD5 if available
                if "md5" in model:
                    file_md5 = calculate_md5(output_path)
                    if file_md5 == model["md5"]:
                        print(f"MD5 verified ✅: {file_md5}")
                        success_count += 1
                    else:
                        print(f"MD5 mismatch ❌: expected {model['md5']}, got {file_md5}")
                        print(f"Will re-download model...")
                        if download_file(model["url"], output_path, model.get("md5")):
                            success_count += 1
                        else:
                            failure_count += 1
                else:
                    success_count += 1
            else:
                # Download the model
                if download_file(model["url"], output_path, model.get("md5")):
                    success_count += 1
                else:
                    failure_count += 1
    
    print(f"\nDownload summary: {success_count} successful, {failure_count} failed")
    return success_count > 0 and failure_count == 0

def verify_models(model_types=None):
    """Verify that models exist and have correct MD5 if known"""
    if model_types is None:
        model_types = ["onnx", "pytorch"]
    
    all_verified = True
    for model in MODELS:
        if model["type"] in model_types:
            output_path = model["output_path"]
            if os.path.exists(output_path):
                print(f"Model exists: {output_path}")
                # Verify MD5 if available
                if "md5" in model:
                    file_md5 = calculate_md5(output_path)
                    if file_md5 == model["md5"]:
                        print(f"MD5 verified ✅: {file_md5}")
                    else:
                        print(f"MD5 mismatch ❌: expected {model['md5']}, got {file_md5}")
                        all_verified = False
            else:
                print(f"Model missing ❌: {output_path}")
                all_verified = False
    
    return all_verified

def list_models():
    """List all available models with details"""
    print("\nAvailable Models:")
    print("----------------")
    
    for model in MODELS:
        output_path = model["output_path"]
        exists = "✅" if os.path.exists(output_path) else "❌"
        
        print(f"{exists} {model['name']}")
        print(f"   Type: {model['type']}")
        print(f"   Resolution: {model['resolution']}")
        print(f"   Size: {model['size']}")
        print(f"   Description: {model['description']}")
        
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024*1024)
            print(f"   Actual size: {size_mb:.1f} MB")
            
            # Verify MD5 if available
            if "md5" in model:
                file_md5 = calculate_md5(output_path)
                if file_md5 == model["md5"]:
                    print(f"   MD5: verified ✅")
                else:
                    print(f"   MD5: mismatch ❌")
        
        print("")

def main():
    parser = argparse.ArgumentParser(description="NudeNet Model Manager")
    parser.add_argument('--action', type=str, default='ensure',
                      choices=['download', 'verify', 'list', 'ensure'],
                      help='Action to perform (download, verify, list, ensure)')
    parser.add_argument('--type', type=str, default='all',
                      choices=['all', 'onnx', 'pytorch'],
                      help='Type of models to operate on')
    parser.add_argument('--force', action='store_true',
                      help='Force download even if models exist')
    
    args = parser.parse_args()
    
    print(f"NudeNet Model Manager - {args.action.capitalize()} Models")
    
    # Determine model types
    model_types = None
    if args.type != 'all':
        model_types = [args.type]
    
    # Perform requested action
    if args.action == 'download':
        success = download_models(model_types, args.force)
        return 0 if success else 1
        
    elif args.action == 'verify':
        success = verify_models(model_types)
        if success:
            print("All models verified successfully! ✅")
        else:
            print("Some models failed verification ❌")
        return 0 if success else 1
        
    elif args.action == 'list':
        list_models()
        return 0
        
    elif args.action == 'ensure':
        # First verify, then download missing
        if verify_models(model_types):
            print("All models already present and verified! ✅")
            return 0
        else:
            print("Some models are missing or failed verification, downloading...")
            success = download_models(model_types, args.force)
            return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())