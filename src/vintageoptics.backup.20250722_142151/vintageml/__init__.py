# src/vintageoptics/vintageml/__init__.py
"""
Vintage Machine Learning Module - AI Winter-era algorithms for lens defect detection
Implements classic ML approaches as the primary detection layer before modern methods
"""

from .perceptron import Perceptron, Adaline, MulticlassPerceptron
from .clustering import SelfOrganizingMap, KMeansVintage
from .dimensionality import PCAVintage, LDAVintage
from .dimensionality_enhanced import (
    AdaptiveDimensionalityReducer, RandomProjection,
    SparsePCAVintage, IncrementalPCAVintage, QuantizedPCA
)
from .neighbors import KNNVintage, KNNInpainter
from .detector import VintageMLDefectDetector, VintageDefectResult
from .trainer import VintageMLTrainer
from .compression import (
    LowRankApproximation, PrunedNetwork, 
    DistilledVintageML, CompressedVintageMLDetector,
    compress_vintage_ml_models
)

__all__ = [
    # Classic algorithms
    'Perceptron',
    'Adaline',
    'MulticlassPerceptron',
    'SelfOrganizingMap',
    'KMeansVintage',
    'PCAVintage', 
    'LDAVintage',
    'KNNVintage',
    'KNNInpainter',
    
    # Main components
    'VintageMLDefectDetector',
    'VintageDefectResult',
    'VintageMLTrainer',
    
    # Enhanced dimensionality reduction
    'AdaptiveDimensionalityReducer',
    'RandomProjection',
    'SparsePCAVintage',
    'IncrementalPCAVintage',
    'QuantizedPCA',
    
    # Compression techniques
    'LowRankApproximation',
    'PrunedNetwork',
    'DistilledVintageML',
    'CompressedVintageMLDetector',
    'compress_vintage_ml_models'
]
