"""
VintageOptics - Advanced Lens Correction and Synthesis Library

A comprehensive library for vintage lens correction, defect removal,
and characteristic synthesis using physical optics simulation and
hyperdimensional computing.
"""

from .__version__ import __version__

# Core functionality
from .core.pipeline import (
    VintageOpticsPipeline,
    PipelineConfig,
    PipelineResult,
    quick_process,
    process_with_profile
)

# Hyperdimensional computing features
from .hyperdimensional import (
    HyperdimensionalLensAnalyzer,
    HyperdimensionalEncoder,
    OrthogonalErrorSeparator,
    TopologicalDefectAnalyzer,
    ConstraintBasedCorrector,
    quick_hd_correction,
    analyze_lens_defects,
    separate_vintage_digital_errors
)

# Analysis tools
from .analysis import (
    LensCharacterizer,
    LensCharacteristics,
    QualityAnalyzer,
    QualityMetrics,
    quick_lens_analysis,
    quick_quality_check,
    detailed_quality_report
)

# Detection
from .detection import (
    UnifiedDetector,
    DetectionResult,
    VintageDetector,
    ElectronicDetector,
    detect_lens
)

# Synthesis
from .synthesis import (
    LensSynthesizer,
    SynthesisConfig,
    synthesize_lens_effect,
    CharacteristicLibrary,
    BokehSynthesizer
)

# Physics engine
from .physics import (
    OpticsEngine,
    BrownConradyModel,
    ChromaticAberrationModel,
    VignettingModel,
    DiffractionModel
)

# Depth processing
from .depth import (
    DepthAnalyzer,
    BokehAnalyzer,
    FocusMapper
)

# Types
from .types import (
    ProcessingMode,
    LensType,
    LensProfile,
    BokehShape,
    CorrectionParameters,
    ImageMetadata
)

# Utilities
from .utils import (
    ImageIO,
    ColorManager,
    PerformanceMonitor,
    setup_logging
)

# Statistical tools
from .statistical import (
    AdaptiveCleanup,
    DefectDetector,
    PreservationEngine
)

# API
from .api import app as rest_api

__all__ = [
    # Version
    '__version__',
    
    # Core
    'VintageOpticsPipeline',
    'PipelineConfig',
    'PipelineResult',
    'quick_process',
    'process_with_profile',
    
    # HD Computing
    'HyperdimensionalLensAnalyzer',
    'HyperdimensionalEncoder',
    'OrthogonalErrorSeparator',
    'TopologicalDefectAnalyzer',
    'ConstraintBasedCorrector',
    'quick_hd_correction',
    'analyze_lens_defects',
    'separate_vintage_digital_errors',
    
    # Analysis
    'LensCharacterizer',
    'LensCharacteristics',
    'QualityAnalyzer',
    'QualityMetrics',
    'quick_lens_analysis',
    'quick_quality_check',
    'detailed_quality_report',
    
    # Detection
    'UnifiedDetector',
    'DetectionResult',
    'VintageDetector',
    'ElectronicDetector',
    'detect_lens',
    
    # Synthesis
    'LensSynthesizer',
    'SynthesisConfig',
    'synthesize_lens_effect',
    'CharacteristicLibrary',
    'BokehSynthesizer',
    
    # Physics
    'OpticsEngine',
    'BrownConradyModel',
    'ChromaticAberrationModel',
    'VignettingModel',
    'DiffractionModel',
    
    # Depth
    'DepthAnalyzer',
    'BokehAnalyzer',
    'FocusMapper',
    
    # Types
    'ProcessingMode',
    'LensType',
    'LensProfile',
    'BokehShape',
    'CorrectionParameters',
    'ImageMetadata',
    
    # Utils
    'ImageIO',
    'ColorManager',
    'PerformanceMonitor',
    'setup_logging',
    
    # Statistical
    'AdaptiveCleanup',
    'DefectDetector',
    'PreservationEngine',
    
    # API
    'rest_api'
]

# Set up default logging
from .utils import setup_logging
setup_logging()

# Welcome message
import logging
logger = logging.getLogger(__name__)
logger.info(f"VintageOptics v{__version__} initialized")
logger.info("Hyperdimensional computing features enabled")
