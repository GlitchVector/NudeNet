from nudenet import NudeDetector
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Attempt to use PyTorch with CUDA if available
use_pytorch = True
try:
    # Check if the PYTORCH_DISABLE environment variable is set
    if os.environ.get("PYTORCH_DISABLE", "0").lower() in ("1", "true", "yes"):
        logger.info("PyTorch detector disabled by environment variable")
        use_pytorch = False
    else:
        # Try to import torch to check availability
        import torch
        if torch.cuda.is_available():
            logger.info("CUDA is available - Using PyTorch detector with GPU acceleration")
            # Log CUDA device info
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
        else:
            logger.info("CUDA not available - Using ONNX Runtime")
            use_pytorch = False
except ImportError:
    logger.info("PyTorch not installed - Using ONNX Runtime")
    use_pytorch = False
except Exception as e:
    logger.warning(f"Error checking PyTorch: {e} - Using ONNX Runtime")
    use_pytorch = False

# Initialize the detector with PyTorch if available
detector = NudeDetector(use_pytorch=use_pytorch)

# Log which detector is being used
if hasattr(detector, 'use_pytorch') and detector.use_pytorch:
    logger.info("Using PyTorch detector for predictions")
else:
    logger.info("Using ONNX Runtime detector for predictions")

def predictor(image_paths, batch_size=1):
    """
    Process images with nudenet detector
    
    Args:
        image_paths: List of paths to images to process
        batch_size: Batch size for processing
        
    Returns:
        List of detection results for each image
    """
    return detector.detect_batch(image_paths, batch_size=batch_size)

