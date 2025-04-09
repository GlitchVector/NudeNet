"""
Docker scripts and utilities for NudeNet with GPU acceleration
"""

from .simple_pytorch_detector import (
    SimpleYOLODetector,
    get_cached_model,
    detect_image,
    detect_batch,
    LABELS
)

__all__ = [
    'SimpleYOLODetector',
    'get_cached_model',
    'detect_image',
    'detect_batch',
    'LABELS'
]