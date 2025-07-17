# src/vintageoptics/core/__init__.py

"""
Core pipeline and configuration management
"""

# Import utilities that don't depend on pipeline
from .config_manager import ConfigManager
from .performance_monitor import PerformanceMonitor

# Data classes for pipeline (moved here to avoid circular imports)
import numpy as np
from typing import Dict, Optional, Any

class ImageData:
    """Container for image and metadata"""
    
    def __init__(self, image, depth_map=None, metadata=None):
        self.image = image
        self.depth_map = depth_map
        self.metadata = metadata or {}

class ProcessingResult:
    """Container for processing results"""
    
    def __init__(self, image, mode=None, **kwargs):
        self.image = image
        self.mode = mode
        for k, v in kwargs.items():
            setattr(self, k, v)

class BatchResult:
    """Container for batch processing results"""
    
    def __init__(self, results, summary):
        self.results = results
        self.summary = summary

# NOTE: Pipeline imports moved to avoid circular dependency
# Import pipeline classes separately when needed

__all__ = [
    'ConfigManager', 
    'PerformanceMonitor',
    'ImageData',
    'ProcessingResult',
    'BatchResult'
]
