"""
Compatibility module for old import paths.
Maps old class names to new implementations.
"""

# Import the actual implementations
from .unified_detector import UnifiedLensDetector
from ..vintageml.detector import VintageMLDefectDetector

# Compatibility aliases
UnifiedDetector = UnifiedLensDetector  # Old name compatibility

__all__ = ['UnifiedDetector', 'UnifiedLensDetector', 'VintageMLDefectDetector']
