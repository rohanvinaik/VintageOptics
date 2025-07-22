# src/vintageoptics/detection/__init__.py

"""
Unified lens detection system with comprehensive lens identification
"""

from .unified_detector import UnifiedLensDetector, LensDetectionResult
from .metadata_extractor import MetadataExtractor
from .lens_fingerprinting import LensFingerprinter, OpticalFingerprint
from .base_detector import BaseLensDetector
from .electronic_detector import ElectronicDetector
from .vintage_detector import VintageDetector

__all__ = [
    'UnifiedLensDetector',
    'LensDetectionResult',
    'MetadataExtractor', 
    'LensFingerprinter',
    'OpticalFingerprint',
    'BaseLensDetector',
    'ElectronicDetector',
    'VintageDetector'
]
