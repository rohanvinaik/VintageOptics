# src/vintageoptics/detection/__init__.py

"""
Unified lens detection system with comprehensive lens identification
"""

from .unified_detector import UnifiedLensDetector, LensDetectionResult
from .metadata_extractor import MetadataExtractor
from .lens_fingerprinting import LensFingerprinting, OpticalFingerprint
from .base_detector import BaseLensDetector
from .electronic_detector import ElectronicLensDetector
from .vintage_detector import VintageLensDetector

__all__ = [
    'UnifiedLensDetector',
    'LensDetectionResult',
    'MetadataExtractor', 
    'LensFingerprinting',
    'OpticalFingerprint',
    'BaseLensDetector',
    'ElectronicLensDetector',
    'VintageLensDetector'
]
