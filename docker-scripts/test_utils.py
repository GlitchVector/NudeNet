#!/usr/bin/env python3
"""
Testing and benchmarking utilities for NudeNet
Combines functionality of multiple testing scripts
"""
import time
import os
import sys
import argparse
import importlib.util
from pathlib import Path
import json

# Try to import detector modules
try:
    from nudenet import NudeDetector
    NUDENET_AVAILABLE = True
except ImportError:
    NUDENET_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Class for colored output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 50}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 50}{Colors.ENDC}")

def find_test_image(default_paths=None):
    """Find a test image to use"""
    if default_paths is None:
        default_paths = [
            "/app/fastdeploy_recipe/cory_chase.jpeg",
            "/app/images/test.jpg",
            "/app/test.jpg"
        ]
    
    # First check default paths
    for path in default_paths:
        if os.path.exists(path):
            return path
    
    # If no default image found, search for any image
    print("Default test images not found, searching for any image file...")
    for ext in ['.jpg', '.jpeg', '.png']:
        if os.path.exists("/app"):
            search_dirs = ["/app"]
        else:
            search_dirs = [os.path.expanduser('~'), os.path.dirname(__file__)]
            
        for search_dir in search_dirs:
            for root, _, files in os.walk(search_dir):
                if root.startswith('/app/models'):  # Skip model directories
                    continue
                for file in files:
                    if file.lower().endswith(ext):
                        return os.path.join(root, file)
    
    return None

def load_pytorch_detector(model_path=None):
    """Load the PyTorch detector"""
    # Default model paths
    default_paths = [
        "/app/models/pytorch/320n.pt",
        "/app/docker-scripts/pytorch-models/320n.pt",
    ]
    
    # If model_path is None, try default paths
    if model_path is None:
        for path in default_paths:
            if os.path.exists(path):
                model_path = path
                break
    
    if model_path is None or not os.path.exists(model_path):
        print(f"{Colors.RED}No model found at {model_path}{Colors.ENDC}")
        return None
    
    # Import the PyTorch detector
    detector_path = "/app/docker-scripts/pytorch_detector.py"
    if not os.path.exists(detector_path):
        print(f"{Colors.RED}PyTorch detector not found at {detector_path}{Colors.ENDC}")
        return None
    
    # Use importlib to import the module
    import_name = "pytorch_detector"
    spec = importlib.util.spec_from_file_location(import_name, detector_path)
    detector_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(detector_module)
    
    # Create detector
    print(f"Loading PyTorch detector with model {model_path}...")
    try:
        detector = detector_module.SimpleYOLODetector(model_path)
        return detector
    except Exception as e:
        print(f"{Colors.RED}Error loading PyTorch detector: {e}{Colors.ENDC}")
        return None

def load_onnx_detector(model_path=None):
    """Load the ONNX Runtime detector"""
    # Default model paths
    default_paths = [
        "/app/models/onnx/320n.onnx",
        "/app/nudenet/320n.onnx",
    ]
    
    # If model_path is None, try default paths
    if model_path is None:
        for path in default_paths:
            if os.path.exists(path):
                model_path = path
                break
    
    if model_path is None or not os.path.exists(model_path):
        print(f"{Colors.RED}No ONNX model found at {model_path}{Colors.ENDC}")
        return None
    
    # Import the ONNX runner
    runner_path = "/app/docker-scripts/onnx_runner.py"
    if not os.path.exists(runner_path):
        print(f"{Colors.RED}ONNX runner not found at {runner_path}{Colors.ENDC}")
        return None
    
    # Use importlib to import the module
    import_name = "onnx_runner"
    spec = importlib.util.spec_from_file_location(import_name, runner_path)
    runner_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner_module)
    
    # Create detector
    print(f"Loading ONNX detector with model {model_path}...")
    try:
        detector = runner_module.ONNXDetector(model_path)
        return detector
    except Exception as e:
        print(f"{Colors.RED}Error loading ONNX detector: {e}{Colors.ENDC}")
        return None

def run_single_test(detector, image_path, confidence=0.25):
    """Run a single test with the given detector"""
    if detector is None:
        print(f"{Colors.RED}Detector not loaded{Colors.ENDC}")
        return None
    
    if not os.path.exists(image_path):
        print(f"{Colors.RED}Test image not found at {image_path}{Colors.ENDC}")
        return None
    
    print(f"Running detection on {image_path}...")
    start_time = time.time()
    
    try:
        results = detector.detect(image_path, confidence)
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Detection completed in {duration:.4f} seconds")
        print(f"Found {len(results)} detections:")
        
        # Print first few results
        for i, detection in enumerate(results[:5]):
            print(f"  {i+1}. {detection['class']} (score: {detection['score']:.4f}) at {detection['box']}")
        
        if len(results) > 5:
            print(f"  ... and {len(results) - 5} more detections")
        
        return {
            "duration": duration,
            "num_detections": len(results),
            "results": results
        }
    except Exception as e:
        print(f"{Colors.RED}Error running detection: {e}{Colors.ENDC}")
        return None

def run_benchmark(detector, image_path, iterations=10, warmup=1, confidence=0.25):
    """Run a benchmark with the given detector"""
    if detector is None:
        print(f"{Colors.RED}Detector not loaded{Colors.ENDC}")
        return None
    
    if not os.path.exists(image_path):
        print(f"{Colors.RED}Test image not found at {image_path}{Colors.ENDC}")
        return None
    
    print(f"Running benchmark on {image_path} with {iterations} iterations...")
    
    # Warmup
    print("Warming up detector...")
    for _ in range(warmup):
        detector.detect(image_path, confidence)
    
    # Run benchmark
    times = []
    detection_counts = []
    
    for i in range(iterations):
        # Synchronize CUDA if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()
            
        start_time = time.time()
        results = detector.detect(image_path, confidence)
        
        # Synchronize CUDA if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.time()
        duration = end_time - start_time
        
        times.append(duration)
        detection_counts.append(len(results))
        print(f"  Run {i+1}: {duration*1000:.2f} ms, {len(results)} detections")
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    fps = 1 / avg_time
    
    print("\nBenchmark Results:")
    print(f"  Average detection time: {avg_time*1000:.2f} ms")
    print(f"  Min detection time:     {min_time*1000:.2f} ms")
    print(f"  Max detection time:     {max_time*1000:.2f} ms")
    print(f"  FPS:                    {fps:.2f}")
    print(f"  Detections:             {sum(detection_counts)/len(detection_counts):.1f} (avg)")
    
    return {
        "iterations": iterations,
        "times": times,
        "detection_counts": detection_counts,
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "fps": fps
    }

def run_batch_test(detector, image_paths, batch_size=4, confidence=0.25):
    """Run a batch test with the given detector"""
    if detector is None:
        print(f"{Colors.RED}Detector not loaded{Colors.ENDC}")
        return None
    
    # Verify images exist
    valid_paths = []
    for path in image_paths:
        if os.path.exists(path):
            valid_paths.append(path)
        else:
            print(f"{Colors.YELLOW}Warning: Image not found at {path}{Colors.ENDC}")
    
    if not valid_paths:
        print(f"{Colors.RED}No valid images found{Colors.ENDC}")
        return None
    
    print(f"Running batch detection on {len(valid_paths)} images with batch size {batch_size}...")
    
    # Check if detector supports batch detection
    if not hasattr(detector, 'detect_batch'):
        print(f"{Colors.YELLOW}Detector doesn't support batch detection, using sequential detection{Colors.ENDC}")
        
        start_time = time.time()
        all_results = []
        for i, path in enumerate(valid_paths):
            print(f"  Processing image {i+1}/{len(valid_paths)}: {path}")
            results = detector.detect(path, confidence)
            all_results.append(results)
        end_time = time.time()
        
        total_duration = end_time - start_time
        print(f"Sequential detection completed in {total_duration:.4f} seconds")
        print(f"Average time per image: {total_duration/len(valid_paths):.4f} seconds")
        
        return {
            "batch_size": 1,
            "num_images": len(valid_paths),
            "total_duration": total_duration,
            "avg_per_image": total_duration/len(valid_paths),
            "results": all_results
        }
    else:
        # Use batch detection
        start_time = time.time()
        batch_results = detector.detect_batch(valid_paths, batch_size=batch_size)
        end_time = time.time()
        
        total_duration = end_time - start_time
        print(f"Batch detection completed in {total_duration:.4f} seconds")
        print(f"Average time per image: {total_duration/len(valid_paths):.4f} seconds")
        
        # Print summary
        total_detections = sum(len(results) for results in batch_results)
        print(f"Total detections: {total_detections} ({total_detections/len(valid_paths):.1f} per image)")
        
        return {
            "batch_size": batch_size,
            "num_images": len(valid_paths),
            "total_duration": total_duration,
            "avg_per_image": total_duration/len(valid_paths),
            "results": batch_results
        }

def compare_detectors(image_path, iterations=5):
    """Compare different detectors on the same image"""
    print_header("Detector Comparison")
    
    if not os.path.exists(image_path):
        print(f"{Colors.RED}Test image not found at {image_path}{Colors.ENDC}")
        return None
    
    results = {}
    
    # Test NudeDetector with PyTorch
    if NUDENET_AVAILABLE:
        print("\nTesting NudeDetector with PyTorch:")
        try:
            detector = NudeDetector(use_pytorch=True)
            if hasattr(detector, 'use_pytorch') and detector.use_pytorch:
                start_time = time.time()
                for i in range(iterations):
                    detections = detector.detect(image_path)
                    print(f"  Run {i+1}: {len(detections)} detections")
                duration = (time.time() - start_time) / iterations
                fps = 1 / duration
                print(f"  Average: {duration*1000:.2f} ms ({fps:.2f} FPS)")
                results["NudeDetector_PyTorch"] = {
                    "duration": duration,
                    "fps": fps,
                    "detections": len(detections)
                }
            else:
                print(f"{Colors.YELLOW}PyTorch not available, skipped{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
    
    # Test NudeDetector with ONNX
    if NUDENET_AVAILABLE:
        print("\nTesting NudeDetector with ONNX:")
        try:
            detector = NudeDetector(use_pytorch=False)
            start_time = time.time()
            for i in range(iterations):
                detections = detector.detect(image_path)
                print(f"  Run {i+1}: {len(detections)} detections")
            duration = (time.time() - start_time) / iterations
            fps = 1 / duration
            print(f"  Average: {duration*1000:.2f} ms ({fps:.2f} FPS)")
            results["NudeDetector_ONNX"] = {
                "duration": duration,
                "fps": fps,
                "detections": len(detections)
            }
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
    
    # Test PyTorch detector
    print("\nTesting PyTorch detector:")
    pytorch_detector = load_pytorch_detector()
    if pytorch_detector:
        try:
            start_time = time.time()
            for i in range(iterations):
                detections = pytorch_detector.detect(image_path)
                print(f"  Run {i+1}: {len(detections)} detections")
            duration = (time.time() - start_time) / iterations
            fps = 1 / duration
            print(f"  Average: {duration*1000:.2f} ms ({fps:.2f} FPS)")
            results["PyTorch_Detector"] = {
                "duration": duration,
                "fps": fps,
                "detections": len(detections)
            }
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
    
    # Test ONNX detector
    print("\nTesting ONNX detector:")
    onnx_detector = load_onnx_detector()
    if onnx_detector:
        try:
            start_time = time.time()
            for i in range(iterations):
                detections = onnx_detector.detect(image_path)
                print(f"  Run {i+1}: {len(detections)} detections")
            duration = (time.time() - start_time) / iterations
            fps = 1 / duration
            print(f"  Average: {duration*1000:.2f} ms ({fps:.2f} FPS)")
            results["ONNX_Detector"] = {
                "duration": duration,
                "fps": fps,
                "detections": len(detections)
            }
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
    
    # Print comparison
    if results:
        print("\nDetector Comparison Summary:")
        print(f"{'Detector':<20} {'Time (ms)':<10} {'FPS':<10} {'Detections':<10}")
        print("-" * 50)
        for name, result in results.items():
            print(f"{name:<20} {result['duration']*1000:<10.2f} {result['fps']:<10.2f} {result['detections']:<10}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="NudeNet Testing Utilities")
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run a single test')
    test_parser.add_argument('--detector', type=str, default='pytorch',
                           choices=['pytorch', 'onnx', 'nudenet'],
                           help='Detector to use')
    test_parser.add_argument('--model', type=str, default=None,
                           help='Path to model file')
    test_parser.add_argument('--image', type=str, default=None,
                           help='Path to test image')
    test_parser.add_argument('--confidence', type=float, default=0.25,
                           help='Confidence threshold')
    test_parser.add_argument('--output-json', type=str, default=None,
                           help='Write results to JSON file')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run benchmark')
    bench_parser.add_argument('--detector', type=str, default='pytorch',
                            choices=['pytorch', 'onnx', 'nudenet'],
                            help='Detector to use')
    bench_parser.add_argument('--model', type=str, default=None,
                            help='Path to model file')
    bench_parser.add_argument('--image', type=str, default=None,
                            help='Path to test image')
    bench_parser.add_argument('--iterations', type=int, default=10,
                            help='Number of iterations')
    bench_parser.add_argument('--warmup', type=int, default=1,
                            help='Number of warmup iterations')
    bench_parser.add_argument('--confidence', type=float, default=0.25,
                            help='Confidence threshold')
    bench_parser.add_argument('--output-json', type=str, default=None,
                            help='Write results to JSON file')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Run batch test')
    batch_parser.add_argument('--detector', type=str, default='pytorch',
                            choices=['pytorch', 'onnx', 'nudenet'],
                            help='Detector to use')
    batch_parser.add_argument('--model', type=str, default=None,
                            help='Path to model file')
    batch_parser.add_argument('--images', type=str, required=True,
                            help='Comma-separated list of image paths')
    batch_parser.add_argument('--batch-size', type=int, default=4,
                            help='Batch size')
    batch_parser.add_argument('--confidence', type=float, default=0.25,
                            help='Confidence threshold')
    batch_parser.add_argument('--output-json', type=str, default=None,
                            help='Write results to JSON file')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare detectors')
    compare_parser.add_argument('--image', type=str, default=None,
                              help='Path to test image')
    compare_parser.add_argument('--iterations', type=int, default=5,
                              help='Number of iterations')
    compare_parser.add_argument('--output-json', type=str, default=None,
                              help='Write results to JSON file')
    
    args = parser.parse_args()
    
    # Find a test image if not specified
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command in ['test', 'benchmark', 'compare'] and not hasattr(args, 'image') or args.image is None:
        args.image = find_test_image()
        if args.image is None:
            print(f"{Colors.RED}No test image found. Please specify one with --image{Colors.ENDC}")
            return 1
        else:
            print(f"Using test image: {args.image}")
    
    results = None
    
    # Run the requested command
    if args.command == 'test':
        print_header(f"Running Test with {args.detector.capitalize()} Detector")
        
        # Load the requested detector
        detector = None
        if args.detector == 'pytorch':
            detector = load_pytorch_detector(args.model)
        elif args.detector == 'onnx':
            detector = load_onnx_detector(args.model)
        elif args.detector == 'nudenet':
            if NUDENET_AVAILABLE:
                detector = NudeDetector(use_pytorch=True)
            else:
                print(f"{Colors.RED}NudeDetector not available{Colors.ENDC}")
                return 1
        
        if detector is None:
            print(f"{Colors.RED}Failed to load detector{Colors.ENDC}")
            return 1
        
        # Run the test
        results = run_single_test(detector, args.image, args.confidence)
        
    elif args.command == 'benchmark':
        print_header(f"Running Benchmark with {args.detector.capitalize()} Detector")
        
        # Load the requested detector
        detector = None
        if args.detector == 'pytorch':
            detector = load_pytorch_detector(args.model)
        elif args.detector == 'onnx':
            detector = load_onnx_detector(args.model)
        elif args.detector == 'nudenet':
            if NUDENET_AVAILABLE:
                detector = NudeDetector(use_pytorch=True)
            else:
                print(f"{Colors.RED}NudeDetector not available{Colors.ENDC}")
                return 1
        
        if detector is None:
            print(f"{Colors.RED}Failed to load detector{Colors.ENDC}")
            return 1
        
        # Run the benchmark
        results = run_benchmark(detector, args.image, args.iterations, args.warmup, args.confidence)
        
    elif args.command == 'batch':
        print_header(f"Running Batch Test with {args.detector.capitalize()} Detector")
        
        # Load the requested detector
        detector = None
        if args.detector == 'pytorch':
            detector = load_pytorch_detector(args.model)
        elif args.detector == 'onnx':
            detector = load_onnx_detector(args.model)
        elif args.detector == 'nudenet':
            if NUDENET_AVAILABLE:
                detector = NudeDetector(use_pytorch=True)
            else:
                print(f"{Colors.RED}NudeDetector not available{Colors.ENDC}")
                return 1
        
        if detector is None:
            print(f"{Colors.RED}Failed to load detector{Colors.ENDC}")
            return 1
        
        # Parse image paths
        image_paths = args.images.split(',')
        
        # Run the batch test
        results = run_batch_test(detector, image_paths, args.batch_size, args.confidence)
        
    elif args.command == 'compare':
        results = compare_detectors(args.image, args.iterations)
    
    # Save results to JSON if requested
    if results and args.output_json:
        try:
            # Convert non-serializable values
            results_copy = {}
            for key, value in results.items():
                if isinstance(value, (list, dict, int, float, str, bool)) or value is None:
                    results_copy[key] = value
                else:
                    results_copy[key] = str(value)
            
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
            
            with open(args.output_json, 'w') as f:
                json.dump(results_copy, f, indent=2)
            print(f"Results saved to {args.output_json}")
        except Exception as e:
            print(f"{Colors.RED}Error saving results to JSON: {e}{Colors.ENDC}")
    
    return 0 if results else 1

if __name__ == "__main__":
    sys.exit(main())