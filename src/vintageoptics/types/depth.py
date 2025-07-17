# src/vintageoptics/types/depth.py

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

@dataclass
class FocusPoint:
    """Represents a focus point in the image"""
    x: int
    y: int
    depth: float
    confidence: float
    
    def to_dict(self):
        return {
            'x': self.x,
            'y': self.y, 
            'depth': self.depth,
            'confidence': self.confidence
        }

@dataclass
class DepthLayer:
    """Represents a depth layer for processing"""
    layer_id: int
    mask: np.ndarray
    mean_depth: float
    depth_range: Tuple[float, float]
    blur_characteristics: Dict
    processing_priority: float

@dataclass 
class DepthMap:
    """Complete depth map with analysis results"""
    depth_map: np.ndarray
    confidence_map: np.ndarray
    focus_points: List[FocusPoint]
    depth_layers: List[DepthLayer]
