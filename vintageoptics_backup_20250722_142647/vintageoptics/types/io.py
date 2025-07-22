# src/vintageoptics/types/io.py

from dataclasses import dataclass
from typing import Dict, Optional, Any
import numpy as np

@dataclass
class ImageData:
    """Container for image and associated data"""
    image: np.ndarray
    depth_map: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ProcessingResult:
    """Container for processing results"""
    image: np.ndarray
    mode: str
    metadata: Dict
    quality_metrics: Optional[Dict] = None
    
    def __init__(self, image, mode, **kwargs):
        self.image = image
        self.mode = mode
        self.metadata = {}
        for k, v in kwargs.items():
            setattr(self, k, v)

@dataclass
class BatchResult:
    """Container for batch processing results"""
    results: list
    report: Dict
    
    def __init__(self, results, report):
        self.results = results
        self.report = report
