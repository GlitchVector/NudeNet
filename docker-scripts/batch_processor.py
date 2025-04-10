#!/usr/bin/env python3
"""
NudeNet Batch Image Processor for JSON lists
--------------------------------------------
Process images listed in a JSON file with paths from various locations

This script is optimized for performance and memory efficiency:
- Uses stderr for logging to prevent buffer overflow issues
- Minimizes log verbosity in non-debug mode
- Adapts progress reporting frequency based on batch size
- Implements batched processing with memory monitoring
"""

import os
import sys
import json
import time
import argparse
import psutil
import logging
import fcntl
import threading
from pathlib import Path, PurePath
from nudenet import NudeDetector

# Configure logging
# Use stderr for regular logging so it doesn't interfere with JSON progress output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("nudenet-batch")

# Default to a higher log level to reduce verbosity, will be adjusted if --debug is used
logger.setLevel(logging.WARNING)

def force_flush_stdout():
    """
    Force stdout to flush immediately using low-level operations.
    This is more reliable than Python's flush() in containerized environments.
    """
    # Try multiple flush methods to ensure output gets through
    sys.stdout.flush()  # Python's built-in flush
    
    try:
        # Unix-specific: Use fsync to force OS to write data
        fd = sys.stdout.fileno()
        fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_SYNC)
        os.fsync(fd)
    except (AttributeError, OSError, ValueError):
        # Not all environments support these operations
        pass
    
    # Last resort: Try to write directly to /dev/stdout device file
    try:
        with open('/dev/stdout', 'w') as stdout_dev:
            stdout_dev.write('\n')  # Empty line to force flush
            stdout_dev.flush()
    except:
        pass

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
    original_path = path
    
    # Replace backslashes with forward slashes
    path = path.replace('\\', '/')
    
    # Check if it's a Windows path with drive letter (e.g., D:/path/...)
    if len(path) > 1 and path[1] == ':' and path[0].isalpha():
        # Extract drive letter
        drive = path[0].lower()
        # Create mount path (/mnt/d/...)
        path = f"/mnt/{drive}/{path[3:]}"
        logger.debug(f"Converted Windows path: {original_path} → {path}")
    else:
        logger.debug(f"Path appears to be non-Windows or already normalized: {path}")
    
    return path

def check_file_exists(path):
    """Check if a file exists, with helpful logging for debugging path issues"""
    exists = os.path.exists(path)
    if exists:
        # Only log detailed file info in debug mode
        logger.debug(f"File exists: {path}")
        try:
            if logger.level <= logging.DEBUG:
                size = os.path.getsize(path)
                logger.debug(f"File size: {size} bytes")
        except Exception as e:
            logger.warning(f"Error getting file size: {str(e)}")
    else:
        parent_dir = os.path.dirname(path)
        if not os.path.exists(parent_dir):
            logger.warning(f"Parent directory doesn't exist: {parent_dir}")
            # Only check mounts in debug mode to reduce verbosity
            if logger.level <= logging.DEBUG:
                # Try to list mountpoints for debugging
                try:
                    mounts = []
                    if os.path.exists('/mnt'):
                        mounts = os.listdir('/mnt')
                    logger.debug(f"Available mounts in /mnt: {mounts}")
                except Exception as e:
                    logger.warning(f"Error checking mounts: {str(e)}")
        logger.warning(f"File not found: {path}")
    return exists

def process_image_batch(image_paths, json_progress=False, current_count=0, total_count=0, start_time=None):
    """Process a batch of images and return results"""
    logger.debug(f"Processing batch of {len(image_paths)} images")
    results = {}
    valid_paths = []
    original_to_normalized = {}
    
    # Normalize paths more aggressively for Windows file paths
    logger.debug("Normalizing paths and checking file existence")
    for path in image_paths:
        normalized_path = normalize_windows_path(path)
        original_to_normalized[path] = normalized_path
        logger.debug(f"Checking existence of: {normalized_path}")
        exists = check_file_exists(normalized_path)
        
        if exists:
            logger.debug(f"Valid file found: {normalized_path}")
            valid_paths.append(path)
        else:
            logger.warning(f"File not found, skipping: {normalized_path}")
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
        
        # For individual image processing with progress reporting
        if json_progress and start_time is not None:
            # Process images one by one with immediate progress updates
            for i, (original_path, normalized_path) in enumerate(zip(valid_paths, paths_to_process)):
                try:
                    # Update progress for individual image
                    current_image = current_count + i + 1
                    elapsed = time.time() - start_time
                    
                    # Update global progress for the background reporter thread
                    global current_progress
                    current_progress = current_image
                    
                    # Process individual image
                    detections = detector.detect(normalized_path)
                    results[original_path] = {
                        'detections': detections,
                        'success': True
                    }
                    
                    # Print more detailed progress on milestones
                    if current_image % 20 == 0 or current_image == total_count:
                        print_progress(current_image, total_count, elapsed, json_progress)
                except Exception as e:
                    results[original_path] = {
                        'error': str(e),
                        'success': False
                    }
        else:
            # Process batch with normalized paths
            start_time_batch = time.time()
            detections_batch = detector.detect_batch(paths_to_process)
            batch_time = time.time() - start_time_batch
            
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
        print(f"Batch processing failed: {str(e)}. Falling back to individual processing.", file=sys.stderr)
        for i, path in enumerate(valid_paths):
            try:
                detector = get_detector()
                normalized_path = original_to_normalized[path]
                
                # We already checked existence, no need to check again
                # Update progress for individual image in fallback mode
                if json_progress and start_time is not None:
                    current_image = current_count + i + 1
                    elapsed = time.time() - start_time
                    
                    # Update global progress for the background reporter thread
                    global current_progress
                    current_progress = current_image
                
                # Process individual image
                detections = detector.detect(normalized_path)
                
                results[path] = {
                    'detections': detections,
                    'success': True
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
    # Set update frequency based on batch size to prevent too many updates
    if total <= 100:
        min_update_frequency = 1  # Every image for very small batches
    elif total <= 500:
        min_update_frequency = 5  # Every 5 images for small batches
    elif total <= 2000:
        min_update_frequency = 10  # Every 10 images for medium batches
    else:
        min_update_frequency = 20  # Every 20 images for large batches
        
    # Check if we should print an update based on frequency
    should_update = (current % min_update_frequency == 0) or (current == total)
    if not should_update and current > 0:  # Skip intermediate updates
        return
        
    # Get memory information (only when actually printing)
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
            "percent": round(current/total*100, 1) if total > 0 else 0
        }
        
        # Major milestone is every ~10% or at the end
        is_major_milestone = (current % max(1, min(total // 10, 100)) == 0) or current == total
        
        # For major milestones, include detailed metrics
        if is_major_milestone:
            progress_data.update({
                "elapsed_seconds": round(elapsed, 2),
                "images_per_second": round(current/elapsed, 2) if elapsed > 0 else 0,
                "estimated_remaining": round((total-current) / (current/elapsed) if current > 0 and elapsed > 0 else 0, 2),
                "memory_percent": memory_info["memory_percent"]
            })
        
        print(json.dumps(progress_data), flush=True)
        # Use aggressive flush methods
        force_flush_stdout()
    else:
        images_per_sec = current / elapsed if elapsed > 0 else 0
        print(f"Progress: {current}/{total} images processed "
              f"({current/total*100:.1f}%, {images_per_sec:.2f} images/sec)")
        print(f"Memory usage: {memory_info['memory_percent']}% - "
              f"{memory_info['memory_used_gb']}GB / {memory_info['memory_total_gb']}GB")
        force_flush_stdout()

# Global variables for tracking progress
current_progress = 0
total_progress = 0
start_time_global = 0
exit_flag = False
is_json_progress = False

def progress_reporter_thread():
    """A separate thread that reports progress every second regardless of batch processing"""
    interval = 0.5  # Half second update interval
    last_progress = 0
    
    while not exit_flag:
        if is_json_progress and current_progress > 0 and current_progress <= total_progress:
            # Only report if progress has changed
            if current_progress > last_progress:
                elapsed = time.time() - start_time_global
                
                # Create minimal progress update
                progress_data = {
                    "type": "progress",
                    "current": current_progress,
                    "total": total_progress,
                    "percent": round(current_progress/total_progress*100, 1) if total_progress > 0 else 0
                }
                
                # Add detailed metrics every 10% or at completion
                is_milestone = (current_progress % max(1, min(total_progress // 10, 50)) == 0) or current_progress == total_progress
                if is_milestone:
                    # Memory info
                    vm = psutil.virtual_memory()
                    
                    # Add extra info for milestone
                    progress_data.update({
                        "elapsed_seconds": round(elapsed, 2),
                        "images_per_second": round(current_progress/elapsed, 2) if elapsed > 0 else 0,
                        "estimated_remaining": round((total_progress-current_progress) / (current_progress/elapsed) if current_progress > 0 and elapsed > 0 else 0, 2),
                        "memory_percent": round(vm.percent, 1)
                    })
                
                # Print directly to device file to bypass all buffering
                try:
                    with open('/dev/stdout', 'w') as f:
                        f.write(json.dumps(progress_data) + '\n')
                        f.flush()
                except:
                    # Fallback to standard methods
                    print(json.dumps(progress_data), flush=True)
                    force_flush_stdout()
                
                last_progress = current_progress
        
        # Sleep briefly
        time.sleep(interval)

def main():
    global current_progress, total_progress, start_time_global, exit_flag, is_json_progress
    
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
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set logging level based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    else:
        # In non-debug mode, keep logging minimal
        logger.setLevel(logging.WARNING)
    
    # Log environment information only in debug mode
    if logger.level <= logging.DEBUG:
        logger.debug(f"Starting NudeNet batch processor")
        logger.debug(f"Python version: {sys.version}")
        logger.debug(f"Running as user: {os.getuid()}")
        logger.debug(f"Current working directory: {os.getcwd()}")
        
        # Log mount points for debugging
        try:
            if os.path.exists('/mnt'):
                mounts = os.listdir('/mnt')
                logger.debug(f"Available mounts in /mnt: {mounts}")
        except Exception as e:
            logger.warning(f"Error checking mounts: {str(e)}")
    else:
        # Just log basic info in non-debug mode
        logger.info(f"Starting NudeNet batch processor")
    
    # Log command line arguments at the appropriate level
    if logger.level <= logging.DEBUG:
        logger.debug(f"Input JSON: {args.input_json}")
        logger.debug(f"Output path: {args.output}")
        logger.debug(f"Batch size: {args.batch_size}")
        if args.model:
            logger.debug(f"Model path: {args.model}")
        else:
            logger.debug("Using default model")
    
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
        logger.info(f"Reading JSON file: {input_json_path}")
        
        try:
            # Get file size and permissions for debugging
            file_size = os.path.getsize(input_json_path)
            file_perms = oct(os.stat(input_json_path).st_mode & 0o777)
            logger.info(f"Input JSON file size: {file_size} bytes, permissions: {file_perms}")
        except Exception as e:
            logger.warning(f"Error getting file info: {str(e)}")
        
        with open(input_json_path, 'r') as f:
            file_content = f.read()
            logger.debug(f"File content (first 1000 chars): {file_content[:1000]}")
            
            # Parse JSON
            data = json.loads(file_content)
            logger.info(f"JSON parsed successfully. Type: {type(data).__name__}")
            
        # Handle different possible JSON structures
        if isinstance(data, list):
            # JSON is a simple list of paths
            image_paths = data
            logger.info(f"Found list of paths directly in JSON root")
        elif isinstance(data, dict) and 'images' in data:
            # JSON has an 'images' key with the list
            image_paths = data['images']
            logger.info(f"Found image paths in 'images' key")
        elif isinstance(data, dict) and 'paths' in data:
            # JSON has a 'paths' key with the list
            image_paths = data['paths']
            logger.info(f"Found image paths in 'paths' key")
        elif isinstance(data, dict) and 'files' in data:
            # JSON has a 'files' key with the list
            image_paths = data['files']
            logger.info(f"Found image paths in 'files' key")
        else:
            # Try to extract any list values
            found_paths = False
            if isinstance(data, dict):
                logger.debug(f"JSON keys: {list(data.keys())}")
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0:
                        image_paths = value
                        logger.info(f"Found image paths in key: {key}")
                        found_paths = True
                        break
            
            if not found_paths:
                logger.error(f"Could not find list of image paths in JSON file")
                logger.debug(f"JSON content structure: {type(data)}")
                return 1
                
    except json.JSONDecodeError as e:
        logger.error(f"Error: {args.input_json} is not a valid JSON file: {str(e)}")
        return 1
    except Exception as e:
        logger.error(f"Error loading input file: {str(e)}")
        return 1
    
    if not image_paths:
        error_msg = "No image paths found in the input JSON"
        logger.error(error_msg)
        if args.json_progress:
            print(json.dumps({"type": "error", "message": error_msg}), flush=True)
        else:
            print(error_msg)
        return 1
        
    # Log summary info at INFO level
    logger.info(f"Found {len(image_paths)} image paths in JSON")
    
    # Log detailed sample paths only in debug mode
    if logger.level <= logging.DEBUG and len(image_paths) > 0:
        logger.debug(f"First few paths:")
        for i, path in enumerate(image_paths[:min(5, len(image_paths))]):
            logger.debug(f"  Path {i+1}: {path}")
        
        # Try to normalize a sample path only in debug mode
        sample_path = image_paths[0]
        normalized = normalize_windows_path(sample_path)
        logger.debug(f"Sample path normalization:")
        logger.debug(f"  Original: {sample_path}")
        logger.debug(f"  Normalized: {normalized}")
        logger.debug(f"  Exists: {os.path.exists(normalized)}")
    
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
        force_flush_stdout()
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
    
    # Setup global progress tracking
    if args.json_progress:
        is_json_progress = True
        total_progress = len(image_paths)
        start_time_global = time.time()
        
        # Start the background progress reporter thread
        reporter = threading.Thread(target=progress_reporter_thread)
        reporter.daemon = True  # Make the thread exit when main thread exits
        reporter.start()
        
        print(json.dumps({"type": "initialized"}), flush=True)
        force_flush_stdout()
    
    # Process images in batches
    results = {}
    start_time = time.time()
    total_images = len(image_paths)
    processed_images = 0
    
    # Process in batches or individually, depending on use case
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
                force_flush_stdout()
            else:
                print(f"ERROR: {error_msg}", file=sys.stderr)
            return 1
            
        batch = image_paths[i:i + args.batch_size]
        
        # Process batch with real-time progress updates if JSON progress is enabled
        # This allows per-image progress updates during batch processing
        batch_results = process_image_batch(
            batch, 
            json_progress=args.json_progress,
            current_count=processed_images,
            total_count=total_images,
            start_time=start_time
        )
        results.update(batch_results)
        
        # Update progress counter
        processed_images += len(batch)
        elapsed = time.time() - start_time
        
        # Update global progress for the background reporter thread
        current_progress = processed_images
        
        # For non-JSON progress mode (which won't get real-time updates), 
        # print batch-level progress updates
        if not args.json_progress:
            # Determine reporting frequency based on total images to reduce output volume
            if total_images <= 100:
                report_frequency = 5  # Every 5 images for small batches
            elif total_images <= 1000:
                report_frequency = 20  # Every 20 images for medium batches
            else:
                report_frequency = 50  # Every 50 images for large batches
                
            # Print progress at calculated frequency or on completion
            if processed_images % report_frequency == 0 or processed_images == total_images:
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
                force_flush_stdout()
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
        force_flush_stdout()
    else:
        print(f"\nSummary:")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average time per image: {total_time/total_images:.4f} seconds")
        print(f"Images processed successfully: {success_count}")
        print(f"Images with errors: {error_count}")
        print(f"Final memory usage: {memory_info['memory_percent']}% - "
              f"{memory_info['memory_used_gb']}GB / {memory_info['memory_total_gb']}GB")
    
    # Signal thread to exit
    exit_flag = True
    
    # Allow time for final progress update
    time.sleep(0.1)
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        exit_flag = True  # Signal thread to terminate
        sys.exit(1)