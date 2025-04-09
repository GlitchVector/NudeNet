#!/usr/bin/env python3
"""
NudeNet Batch Image Processor for JSON lists
--------------------------------------------
Process images listed in a JSON file with paths from various locations
"""

import os
import sys
import json
import time
import argparse
import psutil
from pathlib import Path, PurePath
from nudenet import NudeDetector

def normalize_path(path):
    """
    Normalize Windows paths to Linux paths for Docker
    Examples:
    - C:\\Users\\name\\images\\pic.jpg -> /mnt/c/Users/name/images/pic.jpg
    - D:/Photos/vacation/beach.png -> /mnt/d/Photos/vacation/beach.png
    """
    # Convert Windows path to PurePath for manipulation
    pure_path = PurePath(path)
    
    # Check if it's a Windows path with drive letter
    if pure_path.drive:
        # Extract drive letter without colon
        drive = pure_path.drive.replace(':', '').lower()
        # Create new path with /mnt/drive_letter/ prefix
        parts = list(pure_path.parts[1:])  # Skip the drive part
        linux_path = os.path.join('/mnt', drive, *parts)
        return linux_path
    
    # If not a Windows path or no drive letter, return as is
    return path

# Global detector instance cache
DETECTOR_INSTANCE = None

def get_detector(model_path=None):
    """Get or initialize the detector instance (singleton pattern)"""
    global DETECTOR_INSTANCE
    if DETECTOR_INSTANCE is None:
        # Initialize detector with the specified model
        detector_args = {}
        if model_path:
            detector_args['model_path'] = model_path
        DETECTOR_INSTANCE = NudeDetector(**detector_args)
    return DETECTOR_INSTANCE

def check_memory_usage(threshold=90.0):
    """
    Check memory usage and return True if it's above the threshold.
    """
    # Get virtual memory usage
    vm = psutil.virtual_memory()
    percent_used = vm.percent
    return percent_used > threshold

def normalize_windows_path(path):
    """
    More aggressive Windows path normalization for paths inside the JSON file.
    Handles backslashes and properly converts to Docker-mounted paths.
    """
    # Replace backslashes with forward slashes
    path = path.replace('\\', '/')
    
    # Check if it's a Windows path with drive letter (e.g., D:/path/...)
    if len(path) > 1 and path[1] == ':' and path[0].isalpha():
        # Extract drive letter
        drive = path[0].lower()
        # Create mount path (/mnt/d/...)
        path = f"/mnt/{drive}/{path[3:]}"
    
    return path

def check_file_exists(path):
    """Check if a file exists, with helpful logging for debugging path issues"""
    exists = os.path.exists(path)
    if not exists:
        parent_dir = os.path.dirname(path)
        if not os.path.exists(parent_dir):
            print(f"Warning: Parent directory doesn't exist: {parent_dir}", file=sys.stderr)
        print(f"Warning: File not found: {path}", file=sys.stderr)
    return exists

def process_image_batch(image_paths):
    """Process a batch of images and return results"""
    results = {}
    valid_paths = []
    original_to_normalized = {}
    
    # Normalize paths more aggressively for Windows file paths
    for path in image_paths:
        normalized_path = normalize_windows_path(path)
        original_to_normalized[path] = normalized_path
        exists = check_file_exists(normalized_path)
        
        if exists:
            valid_paths.append(path)
        else:
            # File doesn't exist - record error without trying to process
            results[path] = {
                'error': f"File not found: {normalized_path}",
                'success': False
            }
    
    # If no valid files, just return the errors
    if not valid_paths:
        return results
    
    try:
        # Get the detector instance
        detector = get_detector()
        
        # Check memory before processing
        memory_before = psutil.virtual_memory().percent
        
        # Collect all normalized paths for valid files
        paths_to_process = [original_to_normalized[p] for p in valid_paths]
        
        # Process batch with normalized paths
        start_time = time.time()
        detections_batch = detector.detect_batch(paths_to_process)
        batch_time = time.time() - start_time
        
        # Map results back to original paths
        for i, path in enumerate(valid_paths):
            try:
                results[path] = {
                    'detections': detections_batch[i],
                    'success': True
                }
            except Exception as e:
                results[path] = {
                    'error': str(e),
                    'success': False
                }
    
    except Exception as e:
        # If batch processing fails, process one by one as fallback
        print(f"Batch processing failed: {str(e)}. Falling back to individual processing.")
        for path in valid_paths:
            try:
                detector = get_detector()
                normalized_path = original_to_normalized[path]
                
                if os.path.exists(normalized_path):
                    start_time = time.time()
                    detections = detector.detect(normalized_path)
                    
                    results[path] = {
                        'detections': detections,
                        'success': True
                    }
                else:
                    results[path] = {
                        'error': f"File not found: {normalized_path}",
                        'success': False
                    }
            except Exception as e:
                results[path] = {
                    'error': f"Error processing {path}: {str(e)}",
                    'success': False
                }
    
    # Check memory after processing
    memory_after = psutil.virtual_memory().percent
    memory_change = memory_after - memory_before
    
    # Add memory warning if usage is high
    if check_memory_usage(threshold=85.0):
        print(f"Warning: High memory usage detected ({memory_after:.1f}%). Consider reducing batch size.", 
              file=sys.stderr)
    
    return results

def print_progress(current, total, elapsed, json_format=False):
    """Print progress information in plain text or JSON format"""
    # Get memory information
    vm = psutil.virtual_memory()
    memory_info = {
        "memory_percent": round(vm.percent, 1),
        "memory_used_gb": round(vm.used / (1024**3), 2),
        "memory_total_gb": round(vm.total / (1024**3), 2)
    }
    
    if json_format:
        progress_data = {
            "type": "progress",
            "current": current,
            "total": total,
            "percent": round(current/total*100, 1) if total > 0 else 0,
            "elapsed_seconds": round(elapsed, 2),
            "images_per_second": round(current/elapsed, 2) if elapsed > 0 else 0,
            "estimated_remaining": round((total-current) / (current/elapsed) if current > 0 and elapsed > 0 else 0, 2)
        }
        # Add memory information
        progress_data.update(memory_info)
        print(json.dumps(progress_data), flush=True)
    else:
        images_per_sec = current / elapsed if elapsed > 0 else 0
        print(f"Progress: {current}/{total} images processed "
              f"({current/total*100:.1f}%, {images_per_sec:.2f} images/sec)")
        print(f"Memory usage: {memory_info['memory_percent']}% - "
              f"{memory_info['memory_used_gb']}GB / {memory_info['memory_total_gb']}GB")

def main():
    parser = argparse.ArgumentParser(description='Process images listed in a JSON file with NudeNet')
    parser.add_argument('input_json', help='JSON file containing list of image paths')
    parser.add_argument('--output', '-o', help='Output JSON file for results', required=True)
    parser.add_argument('--batch-size', '-b', type=int, default=16,
                        help='Number of images to process in each batch (default: 16)')
    parser.add_argument('--model', '-m', help='Path to model file (default: use built-in model)')
    parser.add_argument('--json-progress', action='store_true',
                        help='Output progress information in JSON format')
    parser.add_argument('--memory-warning', type=float, default=85.0,
                        help='Memory usage percentage at which to issue warnings (default: 85.0)')
    parser.add_argument('--memory-limit', type=float, default=95.0,
                        help='Memory usage percentage at which to abort processing (default: 95.0)')
    
    args = parser.parse_args()
    
    # Check input file exists
    input_json_path = Path(args.input_json)
    if not input_json_path.exists():
        error_msg = f"Error: Input file {args.input_json} does not exist"
        if args.json_progress:
            print(json.dumps({"type": "error", "message": error_msg}), flush=True)
        else:
            print(error_msg)
        return 1
    
    # Load list of image paths from JSON
    try:
        with open(input_json_path, 'r') as f:
            data = json.load(f)
            
        # Handle different possible JSON structures
        if isinstance(data, list):
            # JSON is a simple list of paths
            image_paths = data
        elif isinstance(data, dict) and 'images' in data:
            # JSON has an 'images' key with the list
            image_paths = data['images']
        elif isinstance(data, dict) and 'paths' in data:
            # JSON has a 'paths' key with the list
            image_paths = data['paths']
        elif isinstance(data, dict) and 'files' in data:
            # JSON has a 'files' key with the list
            image_paths = data['files']
        else:
            # Try to extract any list values
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    image_paths = value
                    print(f"Found image paths in key: {key}")
                    break
            else:
                print("Error: Could not find list of image paths in JSON file")
                return 1
                
    except json.JSONDecodeError:
        print(f"Error: {args.input_json} is not a valid JSON file")
        return 1
    except Exception as e:
        print(f"Error loading input file: {str(e)}")
        return 1
    
    if not image_paths:
        error_msg = "No image paths found in the input JSON"
        if args.json_progress:
            print(json.dumps({"type": "error", "message": error_msg}), flush=True)
        else:
            print(error_msg)
        return 1
    
    # Get memory information
    vm = psutil.virtual_memory()
    memory_info = {
        "memory_percent": round(vm.percent, 1),
        "memory_used_gb": round(vm.used / (1024**3), 2),
        "memory_total_gb": round(vm.total / (1024**3), 2)
    }
    
    # Print startup information
    if args.json_progress:
        start_info = {
            "type": "start",
            "total_images": len(image_paths),
            "batch_size": args.batch_size,
            "input": args.input_json,
            "output": args.output,
            "memory_warning_threshold": args.memory_warning,
            "memory_limit_threshold": args.memory_limit
        }
        # Add memory information
        start_info.update(memory_info)
        print(json.dumps(start_info), flush=True)
    else:
        print(f"Found {len(image_paths)} images to process")
        print(f"Memory usage: {memory_info['memory_percent']}% - "
              f"{memory_info['memory_used_gb']}GB / {memory_info['memory_total_gb']}GB")
    
    # Initialize NudeDetector as a singleton
    if not args.json_progress:
        print("Initializing NudeDetector (cached instance)...")
    
    # Initialize or get cached detector
    if args.model:
        detector = get_detector(model_path=args.model)
    else:
        detector = get_detector()
    
    if args.json_progress:
        print(json.dumps({"type": "initialized"}), flush=True)
    
    # Process images in batches
    results = {}
    start_time = time.time()
    total_images = len(image_paths)
    processed_images = 0
    
    # Process in batches
    for i in range(0, len(image_paths), args.batch_size):
        # Check if memory usage is above limit before processing batch
        if check_memory_usage(threshold=args.memory_limit):
            error_msg = f"Memory usage exceeded limit ({psutil.virtual_memory().percent:.1f}% > {args.memory_limit}%). Aborting processing."
            if args.json_progress:
                print(json.dumps({
                    "type": "error", 
                    "message": error_msg,
                    "memory_percent": psutil.virtual_memory().percent
                }), flush=True)
            else:
                print(f"ERROR: {error_msg}", file=sys.stderr)
            return 1
            
        batch = image_paths[i:i + args.batch_size]
        
        # Process batch (uses cached detector internally)
        batch_results = process_image_batch(batch)
        results.update(batch_results)
        
        # Update progress
        processed_images += len(batch)
        elapsed = time.time() - start_time
        
        # Print progress in appropriate format
        if processed_images % 5 == 0 or processed_images == total_images:  # Report every 5 images
            print_progress(processed_images, total_images, elapsed, args.json_progress)
            
        # Check if memory usage is above warning threshold
        if check_memory_usage(threshold=args.memory_warning):
            warn_msg = f"High memory usage detected ({psutil.virtual_memory().percent:.1f}%). Consider reducing batch size."
            if args.json_progress:
                print(json.dumps({
                    "type": "warning", 
                    "message": warn_msg,
                    "memory_percent": psutil.virtual_memory().percent
                }), flush=True)
            else:
                print(f"WARNING: {warn_msg}", file=sys.stderr)
    
    # Save results
    if not args.json_progress:
        print(f"Saving results to {args.output}")
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calculate summary statistics
    total_time = time.time() - start_time
    success_count = sum(1 for _, r in results.items() if r.get('success', False))
    error_count = sum(1 for _, r in results.items() if not r.get('success', False))
    
    # Get final memory information
    vm = psutil.virtual_memory()
    memory_info = {
        "memory_percent": round(vm.percent, 1),
        "memory_used_gb": round(vm.used / (1024**3), 2),
        "memory_total_gb": round(vm.total / (1024**3), 2)
    }
    
    # Print summary in appropriate format
    if args.json_progress:
        complete_data = {
            "type": "complete",
            "total_images": total_images,
            "processed_successfully": success_count,
            "errors": error_count,
            "total_time_seconds": round(total_time, 2),
            "average_time_per_image": round(total_time/total_images, 4) if total_images > 0 else 0,
            "output_file": args.output
        }
        # Add memory information
        complete_data.update(memory_info)
        
        print(json.dumps(complete_data), flush=True)
    else:
        print(f"\nSummary:")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average time per image: {total_time/total_images:.4f} seconds")
        print(f"Images processed successfully: {success_count}")
        print(f"Images with errors: {error_count}")
        print(f"Final memory usage: {memory_info['memory_percent']}% - "
              f"{memory_info['memory_used_gb']}GB / {memory_info['memory_total_gb']}GB")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())