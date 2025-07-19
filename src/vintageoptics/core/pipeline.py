"""
Enhanced core pipeline with hyperdimensional computing integration.

This module provides the main processing pipeline that orchestrates
all components including HD-based correction and analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import cv2
import time
import logging
from dataclasses import dataclass, field

from ..hyperdimensional import (
    HyperdimensionalLensAnalyzer,
    quick_hd_correction,
    separate_vintage_digital_errors
)
from ..analysis import LensCharacterizer, QualityAnalyzer
from ..detection import UnifiedDetector
from ..physics.optics_engine import OpticsEngine
from ..synthesis import LensSynthesizer
from ..depth import DepthAnalyzer, BokehAnalyzer
from ..statistical import AdaptiveCleanup
from ..utils import ImageIO, ColorManager, PerformanceMonitor
from ..types.optics import ProcessingMode, LensProfile

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the processing pipeline."""
    # Processing modes
    mode: ProcessingMode = ProcessingMode.HYBRID
    use_hd: bool = True
    hd_dimension: int = 10000
    
    # Detection settings
    auto_detect: bool = True
    detection_confidence: float = 0.7
    
    # Correction settings
    correction_strength: float = 0.8
    preserve_character: bool = True
    adaptive_strength: bool = True
    
    # Quality settings
    target_quality: float = 0.8
    max_iterations: int = 3
    
    # Performance settings
    use_gpu: bool = True
    enable_caching: bool = True
    parallel_processing: bool = True
    
    # Output settings
    save_intermediate: bool = False
    generate_report: bool = True
    compute_quality_maps: bool = False


@dataclass
class PipelineResult:
    """Results from pipeline processing."""
    # Main outputs
    corrected_image: np.ndarray
    synthesis_result: Optional[np.ndarray] = None
    
    # Analysis results
    lens_characteristics: Optional[Any] = None
    quality_metrics: Optional[Any] = None
    depth_map: Optional[np.ndarray] = None
    
    # HD results
    hd_analysis: Optional[Dict] = None
    vintage_errors: Optional[np.ndarray] = None
    digital_errors: Optional[np.ndarray] = None
    
    # Metadata
    processing_time: float = 0.0
    iterations_used: int = 1
    mode_used: ProcessingMode = ProcessingMode.HYBRID
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    # Reports
    quality_report: Optional[str] = None
    lens_report: Optional[str] = None


class VintageOpticsPipeline:
    """
    Main processing pipeline integrating all VintageOptics components
    including hyperdimensional computing features.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        
        # Initialize components
        self._init_components()
        
        # Performance monitoring
        self.monitor = PerformanceMonitor()
        
        # Cache for processed images
        self.cache = {} if self.config.enable_caching else None
        
    def _init_components(self):
        """Initialize all pipeline components."""
        # HD components
        if self.config.use_hd:
            self.hd_analyzer = HyperdimensionalLensAnalyzer(self.config.hd_dimension)
        else:
            self.hd_analyzer = None
        
        # Traditional components
        self.lens_characterizer = LensCharacterizer(use_hd=self.config.use_hd)
        self.quality_analyzer = QualityAnalyzer(use_hd=self.config.use_hd)
        self.unified_detector = UnifiedDetector()
        
        # Physics and synthesis
        self.optics_engine = OpticsEngine(use_gpu=self.config.use_gpu)
        self.lens_synthesizer = LensSynthesizer()
        
        # Depth analysis
        self.depth_analyzer = DepthAnalyzer()
        self.bokeh_analyzer = BokehAnalyzer()
        
        # Cleanup
        self.adaptive_cleanup = AdaptiveCleanup()
        
        # I/O and color
        self.image_io = ImageIO()
        self.color_manager = ColorManager()
        
    def process(self, 
                image: Union[str, Path, np.ndarray],
                lens_profile: Optional[LensProfile] = None,
                reference_image: Optional[np.ndarray] = None) -> PipelineResult:
        """
        Process an image through the complete pipeline.
        
        Args:
            image: Input image (path or array)
            lens_profile: Optional lens profile to apply
            reference_image: Optional reference for comparison
            
        Returns:
            Complete processing results
        """
        start_time = time.time()
        
        # Load image
        if isinstance(image, (str, Path)):
            image_array = self.image_io.load(str(image))
        else:
            image_array = image.copy()
        
        # Check cache
        cache_key = self._get_cache_key(image_array, lens_profile)
        if self.cache and cache_key in self.cache:
            logger.info("Using cached result")
            return self.cache[cache_key]
        
        # Initialize result
        result = PipelineResult(corrected_image=image_array)
        
        # Step 1: Detection and characterization
        if self.config.auto_detect:
            self._detect_and_characterize(image_array, result)
        
        # Step 2: HD analysis and separation
        if self.config.use_hd:
            self._perform_hd_analysis(image_array, result)
        
        # Step 3: Determine processing mode
        mode = self._determine_processing_mode(result)
        result.mode_used = mode
        
        # Step 4: Apply corrections based on mode
        if mode == ProcessingMode.CORRECTION:
            self._apply_correction_mode(image_array, result)
        elif mode == ProcessingMode.SYNTHESIS:
            self._apply_synthesis_mode(image_array, lens_profile, result)
        else:  # HYBRID
            self._apply_hybrid_mode(image_array, lens_profile, result)
        
        # Step 5: Depth-aware processing
        if self.config.mode in [ProcessingMode.HYBRID, ProcessingMode.SYNTHESIS]:
            self._apply_depth_processing(result)
        
        # Step 6: Quality enhancement iterations
        if self.config.target_quality > 0:
            self._iterative_enhancement(result, reference_image)
        
        # Step 7: Final cleanup and color correction
        self._final_processing(result)
        
        # Step 8: Generate reports
        if self.config.generate_report:
            self._generate_reports(result)
        
        # Update timing
        result.processing_time = time.time() - start_time
        
        # Cache result
        if self.cache:
            self.cache[cache_key] = result
        
        # Log summary
        logger.info(f"Processing completed in {result.processing_time:.2f}s")
        logger.info(f"Mode: {result.mode_used.value}, Quality: {result.quality_metrics.overall_quality:.1%}")
        
        return result
    
    def _detect_and_characterize(self, image: np.ndarray, result: PipelineResult):
        """Detect lens type and characterize properties."""
        logger.info("Detecting and characterizing lens...")
        
        # Unified detection
        detection = self.unified_detector.detect(image)
        result.confidence_scores['lens_detection'] = detection.confidence
        
        # Full characterization
        result.lens_characteristics = self.lens_characterizer.analyze(
            image,
            full_analysis=True
        )
        
        # Store confidence scores
        result.confidence_scores['vintage'] = detection.vintage_probability
        result.confidence_scores['electronic'] = detection.electronic_probability
        
    def _perform_hd_analysis(self, image: np.ndarray, result: PipelineResult):
        """Perform hyperdimensional analysis."""
        logger.info("Performing HD analysis...")
        
        # Full HD analysis
        hd_results = self.hd_analyzer.analyze_and_correct(
            image,
            mode='auto',
            strength=0.0  # Just analyze, don't correct yet
        )
        
        result.hd_analysis = hd_results
        
        # Separate errors
        separation = separate_vintage_digital_errors(image)
        result.vintage_errors = separation['vintage_errors']
        result.digital_errors = separation['digital_errors']
        
        # Update confidence scores
        result.confidence_scores['vintage_errors'] = separation['vintage_confidence']
        result.confidence_scores['digital_errors'] = separation['digital_confidence']
        
    def _determine_processing_mode(self, result: PipelineResult) -> ProcessingMode:
        """Determine the best processing mode based on analysis."""
        if self.config.mode != ProcessingMode.AUTO:
            return self.config.mode
        
        # Auto mode selection based on confidence scores
        vintage_conf = result.confidence_scores.get('vintage', 0)
        defect_count = 0
        
        if result.lens_characteristics:
            defect_count = (
                result.lens_characteristics.dust_spots +
                result.lens_characteristics.scratches +
                result.lens_characteristics.fungus_areas
            )
        
        # Decision logic
        if defect_count > 10 or result.lens_characteristics.haze_level > 0.3:
            # Heavy defects - use correction mode
            return ProcessingMode.CORRECTION
        elif vintage_conf > 0.8 and defect_count < 3:
            # Clear vintage lens with minimal defects - synthesis
            return ProcessingMode.SYNTHESIS
        else:
            # Mixed case - use hybrid
            return ProcessingMode.HYBRID
    
    def _apply_correction_mode(self, image: np.ndarray, result: PipelineResult):
        """Apply correction mode processing."""
        logger.info("Applying correction mode...")
        
        if self.config.use_hd:
            # Use HD correction
            corrected = self.hd_analyzer.corrector.correct_image(
                image,
                correction_strength=self.config.correction_strength
            )
            result.corrected_image = corrected['corrected']
        else:
            # Traditional correction
            # Apply physics-based corrections
            params = self.optics_engine.estimate_parameters(image)
            result.corrected_image = self.optics_engine.correct(image, params)
            
            # Adaptive cleanup
            result.corrected_image = self.adaptive_cleanup.process(
                result.corrected_image,
                preserve_details=self.config.preserve_character
            )
    
    def _apply_synthesis_mode(self, 
                            image: np.ndarray,
                            lens_profile: Optional[LensProfile],
                            result: PipelineResult):
        """Apply synthesis mode processing."""
        logger.info("Applying synthesis mode...")
        
        # Use provided profile or detect from image
        if lens_profile is None and result.lens_characteristics:
            # Create profile from detected characteristics
            lens_profile = self._create_lens_profile(result.lens_characteristics)
        
        if lens_profile:
            # Synthesize lens characteristics
            result.synthesis_result = self.lens_synthesizer.apply(
                image,
                lens_profile,
                strength=self.config.correction_strength
            )
            result.corrected_image = result.synthesis_result
        else:
            logger.warning("No lens profile available for synthesis")
            result.corrected_image = image
    
    def _apply_hybrid_mode(self,
                          image: np.ndarray,
                          lens_profile: Optional[LensProfile],
                          result: PipelineResult):
        """Apply hybrid mode processing."""
        logger.info("Applying hybrid mode...")
        
        # First correct defects
        if self.config.use_hd:
            # HD correction with lower strength
            corrected = self.hd_analyzer.corrector.correct_image(
                image,
                correction_strength=self.config.correction_strength * 0.7
            )
            intermediate = corrected['corrected']
        else:
            # Traditional correction
            params = self.optics_engine.estimate_parameters(image)
            intermediate = self.optics_engine.correct(image, params)
        
        # Then synthesize desired characteristics
        if lens_profile:
            result.synthesis_result = self.lens_synthesizer.apply(
                intermediate,
                lens_profile,
                strength=self.config.correction_strength * 0.5
            )
            result.corrected_image = result.synthesis_result
        else:
            result.corrected_image = intermediate
    
    def _apply_depth_processing(self, result: PipelineResult):
        """Apply depth-aware processing."""
        logger.info("Applying depth-aware processing...")
        
        # Estimate depth map
        result.depth_map = self.depth_analyzer.estimate_depth(result.corrected_image)
        
        # Apply depth-aware bokeh
        if result.synthesis_result is not None:
            bokeh_result = self.bokeh_analyzer.enhance_bokeh(
                result.corrected_image,
                result.depth_map,
                quality='smooth'
            )
            result.corrected_image = bokeh_result
    
    def _iterative_enhancement(self, 
                              result: PipelineResult,
                              reference: Optional[np.ndarray]):
        """Apply iterative quality enhancement."""
        logger.info("Applying iterative enhancement...")
        
        current = result.corrected_image
        
        for iteration in range(self.config.max_iterations):
            # Measure quality
            quality = self.quality_analyzer.analyze(current, reference)
            
            if quality.overall_quality >= self.config.target_quality:
                logger.info(f"Target quality reached at iteration {iteration + 1}")
                break
            
            # Adaptive strength based on quality gap
            quality_gap = self.config.target_quality - quality.overall_quality
            adaptive_strength = min(1.0, quality_gap * 2)
            
            if self.config.use_hd:
                # HD enhancement
                enhanced = self.hd_analyzer.analyze_and_correct(
                    current,
                    mode='hybrid',
                    strength=adaptive_strength
                )
                current = enhanced['corrected']
            else:
                # Traditional enhancement
                current = self.adaptive_cleanup.process(
                    current,
                    strength=adaptive_strength
                )
            
            result.iterations_used = iteration + 1
        
        result.corrected_image = current
        
        # Final quality assessment
        result.quality_metrics = self.quality_analyzer.analyze(
            result.corrected_image,
            reference,
            compute_maps=self.config.compute_quality_maps
        )
    
    def _final_processing(self, result: PipelineResult):
        """Apply final processing steps."""
        logger.info("Applying final processing...")
        
        # Color correction
        result.corrected_image = self.color_manager.correct_color(
            result.corrected_image,
            method='adaptive'
        )
        
        # Ensure proper range
        if result.corrected_image.dtype == np.float32 or result.corrected_image.dtype == np.float64:
            result.corrected_image = np.clip(result.corrected_image * 255, 0, 255).astype(np.uint8)
        
        # Final sharpening if needed
        if result.quality_metrics and result.quality_metrics.sharpness < 0.7:
            kernel = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]]) / 9
            result.corrected_image = cv2.filter2D(result.corrected_image, -1, kernel)
    
    def _generate_reports(self, result: PipelineResult):
        """Generate quality and lens reports."""
        # Quality report
        if result.quality_metrics:
            result.quality_report = self._format_quality_report(result.quality_metrics)
        
        # Lens report
        if result.lens_characteristics:
            result.lens_report = self.lens_characterizer.generate_report(
                result.lens_characteristics
            )
    
    def _format_quality_report(self, metrics: Any) -> str:
        """Format quality metrics into report."""
        report = []
        report.append("=== Quality Analysis Report ===\n")
        report.append(f"Overall Quality: {metrics.overall_quality:.1%}")
        report.append(f"Sharpness: {metrics.sharpness:.1%}")
        report.append(f"Contrast: {metrics.contrast:.1%}")
        report.append(f"Noise Level: {metrics.noise_level:.1%}")
        
        if metrics.hd_quality_score > 0:
            report.append(f"\nHD Quality Score: {metrics.hd_quality_score:.1%}")
            report.append(f"Defect Impact: {metrics.defect_impact:.1%}")
        
        report.append(f"\nPerceptual Quality: {metrics.perceptual_quality:.1%}")
        report.append(f"Aesthetic Score: {metrics.aesthetic_score:.1%}")
        
        return "\n".join(report)
    
    def _create_lens_profile(self, characteristics: Any) -> LensProfile:
        """Create lens profile from characteristics."""
        return LensProfile(
            name=characteristics.lens_model or "Unknown",
            focal_length=characteristics.focal_length,
            max_aperture=characteristics.max_aperture,
            vignetting_amount=1 - np.min(characteristics.vignetting_profile)
            if characteristics.vignetting_profile is not None else 0.2,
            distortion_amount=abs(characteristics.distortion_coefficients.get('k1', 0))
            if characteristics.distortion_coefficients else 0.0,
            chromatic_aberration=characteristics.chromatic_aberration.get('lateral', 0)
            if characteristics.chromatic_aberration else 0.0,
            bokeh_quality=characteristics.bokeh_quality.get('roundness', 0.8)
            if characteristics.bokeh_quality else 0.8
        )
    
    def _get_cache_key(self, image: np.ndarray, lens_profile: Optional[LensProfile]) -> str:
        """Generate cache key for image and settings."""
        # Simple hash based on image shape and mean
        image_hash = f"{image.shape}_{np.mean(image):.2f}"
        profile_hash = lens_profile.name if lens_profile else "none"
        config_hash = f"{self.config.mode.value}_{self.config.correction_strength:.2f}"
        
        return f"{image_hash}_{profile_hash}_{config_hash}"
    
    def batch_process(self, 
                     images: List[Union[str, Path, np.ndarray]],
                     lens_profile: Optional[LensProfile] = None,
                     output_dir: Optional[Path] = None) -> List[PipelineResult]:
        """
        Process multiple images.
        
        Args:
            images: List of images to process
            lens_profile: Optional lens profile to apply
            output_dir: Optional directory to save results
            
        Returns:
            List of processing results
        """
        results = []
        
        for i, image in enumerate(images):
            logger.info(f"Processing image {i+1}/{len(images)}")
            
            try:
                result = self.process(image, lens_profile)
                results.append(result)
                
                # Save if output directory provided
                if output_dir:
                    output_path = output_dir / f"corrected_{i:04d}.jpg"
                    self.image_io.save(str(output_path), result.corrected_image)
                    
            except Exception as e:
                logger.error(f"Failed to process image {i}: {e}")
                results.append(None)
        
        return results


# Convenience functions
def quick_process(image_path: str, 
                 mode: str = 'auto',
                 strength: float = 0.8) -> np.ndarray:
    """
    Quick processing with minimal configuration.
    
    Args:
        image_path: Path to input image
        mode: Processing mode ('correction', 'synthesis', 'hybrid', 'auto')
        strength: Correction strength (0-1)
        
    Returns:
        Processed image
    """
    config = PipelineConfig(
        mode=ProcessingMode(mode.upper()),
        correction_strength=strength,
        use_hd=True
    )
    
    pipeline = VintageOpticsPipeline(config)
    result = pipeline.process(image_path)
    
    return result.corrected_image


def process_with_profile(image_path: str,
                        lens_name: str,
                        strength: float = 0.8) -> np.ndarray:
    """
    Process image with a specific lens profile.
    
    Args:
        image_path: Path to input image
        lens_name: Name of lens profile to apply
        strength: Effect strength (0-1)
        
    Returns:
        Processed image
    """
    # Load lens profile (would come from database in production)
    profile = LensProfile(
        name=lens_name,
        focal_length=50.0,
        max_aperture=1.4,
        vignetting_amount=0.3,
        distortion_amount=0.05,
        chromatic_aberration=1.5,
        bokeh_quality=0.9
    )
    
    config = PipelineConfig(
        mode=ProcessingMode.SYNTHESIS,
        correction_strength=strength
    )
    
    pipeline = VintageOpticsPipeline(config)
    result = pipeline.process(image_path, lens_profile=profile)
    
    return result.corrected_image
