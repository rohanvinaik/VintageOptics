# src/vintageoptics/types/__init__.py

"""
Type definitions for VintageOptics
"""

from .depth import DepthMap, DepthLayer, FocusPoint
from .optics import LensProfile, OpticalParameters
from .io import ImageData, ProcessingResult, BatchResult

__all__ = [
    'DepthMap', 'DepthLayer', 'FocusPoint',
    'LensProfile', 'OpticalParameters', 
    'ImageData', 'ProcessingResult', 'BatchResult'
]
