"""
Hyperdimensional computing module for advanced error correction and pattern recognition.

This module implements hyperdimensional vector representations for lens defects
and sensor errors, enabling orthogonal error separation and robust feature encoding.
"""

from .hd_encoder import HyperdimensionalEncoder
from .error_separator import OrthogonalErrorSeparator
from .defect_topology import TopologicalDefectAnalyzer, TopologicalFeature
from .constraint_solver import ConstraintBasedCorrector
from .integration import (
    HyperdimensionalLensAnalyzer,
    quick_hd_correction,
    analyze_lens_defects,
    separate_vintage_digital_errors
)

__all__ = [
    'HyperdimensionalEncoder',
    'OrthogonalErrorSeparator', 
    'TopologicalDefectAnalyzer',
    'TopologicalFeature',
    'ConstraintBasedCorrector',
    'HyperdimensionalLensAnalyzer',
    'quick_hd_correction',
    'analyze_lens_defects',
    'separate_vintage_digital_errors'
]
