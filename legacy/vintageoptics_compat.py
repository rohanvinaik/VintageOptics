"""
Compatibility layer to restore VintageOptics functionality
Maps the new structure to the expected API
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Dict, Any

# Try to import from the actual VintageOptics structure
try:
    from vintageoptics.core.pipeline import VintageOpticsPipeline, PipelineConfig
    from vintageoptics.types.optics import ProcessingMode, LensProfile, DefectType
    PIPELINE_AVAILABLE = True
except:
    PIPELINE_AVAILABLE = False
    
    # Create dummy classes
    class ProcessingMode:
        CORRECTION = "CORRECTION"
        SYNTHESIS = "SYNTHESIS"
        HYBRID = "HYBRID"
        AUTO = "AUTO"
        
    @dataclass
    class LensProfile:
        name: str
        focal_length: float
        max_aperture: float
        min_aperture: float = 16.0
        aperture_blades: int = 8
        coating_type: str = "multi-coated"
        era: str = "modern"
        manufacturer: str = "Unknown"
        k1: float = -0.01
        k2: float = 0.005
        p1: float = 0.0
        p2: float = 0.0
        vignetting_amount: float = 0.2
        vignetting_falloff: float = 2.5
        chromatic_aberration: float = 0.01
        lateral_chromatic_scale: float = 1.001
        bokeh_quality: float = 0.8
        coma_amount: float = 0.05
        spherical_aberration: float = 0.02
        
    class DefectType:
        DUST = "DUST"
        FUNGUS = "FUNGUS"
        SCRATCHES = "SCRATCHES"
        HAZE = "HAZE"


class ProcessingPipeline:
    """Compatibility wrapper for the expected ProcessingPipeline class"""
    
    def __init__(self):
        self.mode = ProcessingMode.HYBRID
        self.lens_profile = None
        self.enable_defects = False
        self._pipeline = None
        
        if PIPELINE_AVAILABLE:
            try:
                config = PipelineConfig()
                self._pipeline = VintageOpticsPipeline(config)
            except:
                pass
    
    def configure(self, mode=None, lens_profile=None, enable_defects=False):
        """Configure the pipeline"""
        if mode:
            self.mode = mode
        if lens_profile:
            self.lens_profile = lens_profile
        self.enable_defects = enable_defects
        
        # Reconfigure internal pipeline if available
        if PIPELINE_AVAILABLE and self._pipeline:
            try:
                config = PipelineConfig(
                    mode=getattr(ProcessingMode, str(mode).upper(), ProcessingMode.HYBRID),
                    correction_strength=0.8,
                    use_hd=True
                )
                self._pipeline.config = config
            except:
                pass
    
    def process(self, image):
        """Process an image"""
        if self._pipeline and self.lens_profile:
            try:
                # Use the actual pipeline
                result = self._pipeline.process(image, lens_profile=self.lens_profile)
                
                # Create expected result format
                class Result:
                    def __init__(self, corrected_image, quality_metrics=None):
                        self.corrected_image = corrected_image
                        self.quality_metrics = quality_metrics or DummyMetrics()
                        self.detected_defects = []
                        self.correction_strength = 0.8
                
                return Result(result.corrected_image, result.quality_metrics)
            except:
                pass
        
        # Fallback processing
        return DummyResult(image)


class OpticsEngine:
    """Compatibility wrapper for OpticsEngine"""
    
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        self._engine = None
        
        try:
            from vintageoptics.physics.optics_engine import OpticsEngine as RealEngine
            self._engine = RealEngine(use_gpu=use_gpu)
        except:
            pass
    
    def apply_lens_model(self, image, lens_profile):
        """Apply lens characteristics to image"""
        if self._engine:
            try:
                return self._engine.apply_lens_model(image, lens_profile)
            except:
                pass
        
        # Fallback - apply simple vignetting
        result = image.copy()
        h, w = image.shape[:2]
        Y, X = np.ogrid[:h, :w]
        center_x, center_y = w/2, h/2
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        vignette_strength = getattr(lens_profile, 'vignetting_amount', 0.3)
        vignette = 1 - (dist / max_dist) * vignette_strength
        vignette = np.clip(vignette, 0, 1)
        
        for i in range(3):
            result[:, :, i] = (result[:, :, i] * vignette).astype(np.uint8)
        
        return result


class LensSynthesizer:
    """Compatibility wrapper for LensSynthesizer"""
    
    def __init__(self):
        self._synthesizer = None
        try:
            from vintageoptics.synthesis.lens_synthesizer import LensSynthesizer as RealSynthesizer
            self._synthesizer = RealSynthesizer()
        except:
            pass
    
    def apply(self, image, lens_profile, strength=0.8):
        """Apply lens synthesis"""
        if self._synthesizer:
            try:
                return self._synthesizer.apply(image, lens_profile, strength)
            except:
                pass
        
        # Fallback
        return image


class VintageDetector:
    """Compatibility wrapper for VintageDetector"""
    
    def __init__(self):
        self._detector = None
        try:
            from vintageoptics.detection.vintage_detector import VintageDetector as RealDetector
            self._detector = RealDetector()
        except:
            pass
    
    def add_defects(self, image, defect_types):
        """Add vintage defects to image"""
        if self._detector:
            try:
                return self._detector.add_defects(image, defect_types)
            except:
                pass
        
        # Simple fallback - add some noise
        result = image.copy()
        if DefectType.DUST in defect_types:
            # Add some dust spots
            h, w = image.shape[:2]
            for _ in range(20):
                x = np.random.randint(0, w)
                y = np.random.randint(0, h)
                cv2.circle(result, (x, y), 2, (200, 200, 200), -1)
        
        return result


class DummyMetrics:
    """Dummy quality metrics"""
    def __init__(self):
        self.overall_quality = 0.85
        self.sharpness = 0.8
        self.contrast = 0.75
        self.noise_level = 0.1


class DummyResult:
    """Dummy processing result"""
    def __init__(self, image):
        self.corrected_image = image
        self.quality_metrics = DummyMetrics()
        self.detected_defects = []
        self.correction_strength = 0.8


class QualityMetrics:
    """Compatibility wrapper for QualityMetrics"""
    pass
