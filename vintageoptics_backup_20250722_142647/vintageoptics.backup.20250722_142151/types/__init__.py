# src/vintageoptics/types/__init__.py

"""
Type definitions for VintageOptics
"""

from .depth import DepthMap, DepthLayer, FocusPoint
from .optics import LensProfile, OpticalParameters, ImageMetadata, LensParameters, OpticalDefect, LensType
from .io import ImageData, ProcessingResult, BatchResult

__all__ = [
    'DepthMap', 'DepthLayer', 'FocusPoint',
    'LensProfile', 'OpticalParameters', 'ImageMetadata', 'LensParameters', 'OpticalDefect', 'LensType',
    'ImageData', 'ProcessingResult', 'BatchResult'
]
