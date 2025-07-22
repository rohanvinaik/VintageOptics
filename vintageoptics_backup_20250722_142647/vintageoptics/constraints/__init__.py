"""
Constraint-based orchestration system for VintageOptics.

This module provides the foundation for constraint-oriented AI task orchestration,
allowing physical constraints to guide lens correction and synthesis operations.
"""

from .constraint_spec import ConstraintSpecification, PhysicalConstraint
from .error_analyzer import OrthogonalErrorAnalyzer
from .task_graph import OpticalTaskGraph, TaskNode
from .uncertainty import UncertaintyTracker, UncertaintyEstimate

__all__ = [
    'ConstraintSpecification',
    'PhysicalConstraint',
    'OrthogonalErrorAnalyzer',
    'OpticalTaskGraph',
    'TaskNode',
    'UncertaintyTracker',
    'UncertaintyEstimate'
]
