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

def process_image_batch(detector, image_paths):
    """Process a batch of images and return results"""
    results = {}
    # Normalize paths for Docker environment
    normalized_paths = {path: normalize_path(path) for path in image_paths}
    
    try:
        # Process a batch of images with NudeDetector
        start_time = time.time()
        detections_batch = detector.detect_batch([normalized_paths[p] for p in image_paths])
        batch_time = time.time() - start_time
        
        # Map results back to original paths
        for i, path in enumerate(image_paths):
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
        for path in image_paths:
            try:
                start_time = time.time()
                detections = detector.detect(normalized_paths[path])
                
                results[path] = {
                    'detections': detections,
                    'success': True
                }
            except Exception as e:
                results[path] = {
                    'error': str(e),
                    'success': False
                }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Process images listed in a JSON file with NudeNet')
    parser.add_argument('input_json', help='JSON file containing list of image paths')
    parser.add_argument('--output', '-o', help='Output JSON file for results', required=True)
    parser.add_argument('--batch-size', '-b', type=int, default=16,
                        help='Number of images to process in each batch (default: 16)')
    parser.add_argument('--model', '-m', help='Path to model file (default: use built-in model)')
    
    args = parser.parse_args()
    
    # Check input file exists
    input_json_path = Path(args.input_json)
    if not input_json_path.exists():
        print(f"Error: Input file {args.input_json} does not exist")
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
        print("No image paths found in the input JSON")
        return 1
    
    print(f"Found {len(image_paths)} images to process")
    
    # Initialize NudeDetector
    detector_args = {}
    if args.model:
        detector_args['model_path'] = args.model
    
    print("Initializing NudeDetector...")
    detector = NudeDetector(**detector_args)
    
    # Process images in batches
    results = {}
    start_time = time.time()
    total_images = len(image_paths)
    processed_images = 0
    
    # Process in batches
    for i in range(0, len(image_paths), args.batch_size):
        batch = image_paths[i:i + args.batch_size]
        
        # Process batch
        batch_results = process_image_batch(detector, batch)
        results.update(batch_results)
        
        # Update progress
        processed_images += len(batch)
        elapsed = time.time() - start_time
        images_per_sec = processed_images / elapsed if elapsed > 0 else 0
        print(f"Progress: {processed_images}/{total_images} images processed "
              f"({processed_images/total_images*100:.1f}%, {images_per_sec:.2f} images/sec)")
    
    # Save results
    print(f"Saving results to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    total_time = time.time() - start_time
    success_count = sum(1 for _, r in results.items() if r.get('success', False))
    error_count = sum(1 for _, r in results.items() if not r.get('success', False))
    
    print(f"\nSummary:")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time/total_images:.4f} seconds")
    print(f"Images processed successfully: {success_count}")
    print(f"Images with errors: {error_count}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())