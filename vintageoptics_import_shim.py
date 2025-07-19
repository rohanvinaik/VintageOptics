"""
VintageOptics Compatibility Shim
This module provides compatibility mappings for import mismatches
"""

# Import what's actually available
from vintageoptics.detection import VintageLensDetector, ElectronicLensDetector
from vintageoptics.constraints import (
    ConstraintSpecification,
    OrthogonalErrorAnalyzer,
    UncertaintyTracker,
    UncertaintyEstimate
)

# Create compatibility aliases
VintageDetector = VintageLensDetector
ElectronicDetector = ElectronicLensDetector

# Make them available for import
__all__ = [
    'VintageDetector',
    'ElectronicDetector',
    'ConstraintSpecification',
    'OrthogonalErrorAnalyzer',
    'UncertaintyTracker',
    'UncertaintyEstimate'
]
