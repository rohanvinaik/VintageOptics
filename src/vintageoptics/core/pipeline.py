# src/vintageoptics/core/pipeline.py

from typing import Dict, Optional, Union, List
import numpy as np
import cv2
import os
from dataclasses import dataclass
from enum import Enum

from ..detection import UnifiedLensDetector
from ..physics import OpticsEngine
from ..depth import DepthAwareProcessor
from ..synthesis import LensSynthesizer
from ..statistical import AdaptiveCleanup
from ..analysis import QualityAnalyzer
from .config_manager import ConfigManager
from .performance_monitor import PerformanceMonitor
from . import ImageData, ProcessingResult, BatchResult


class ProcessingMode(Enum):
    CORRECT = "correct"
    SYNTHESIZE = "synthesize"
    HYBRID = "hybrid"


@dataclass
class ProcessingRequest:
    """Unified processing request"""
    image_path: str
    mode: ProcessingMode
    output_path: Optional[str] = None
    source_lens: Optional[str] = None
    target_lens: Optional[str] = None
    settings: Optional[Dict] = None
    preserve_metadata: bool = True
    use_depth: bool = True
    gpu_acceleration: bool = True


class VintageOpticsPipeline:
    """Main processing pipeline handling both correction and synthesis"""
    
    def __init__(self, config_path: str):
        self.config = ConfigManager.load(config_path)
        self._initialize_components()
        self._setup_plugins()
    
    def _initialize_components(self):
        """Initialize all processing components"""
        # Detection
        self.detector = UnifiedLensDetector(self.config)
        
        # Processing engines
        self.optics_engine = OpticsEngine(self.config)
        self.depth_processor = DepthAwareProcessor(self.config)
        self.synthesizer = LensSynthesizer(self.config)
        self.cleanup_engine = AdaptiveCleanup(self.config)
        
        # Analysis
        self.quality_analyzer = QualityAnalyzer()
        
        # Performance
        self.performance_monitor = PerformanceMonitor()
    
    def process(self, request: ProcessingRequest) -> ProcessingResult:
        """Unified processing entry point"""
        
        with self.performance_monitor.track("total_processing"):
            # Load and analyze image
            image_data = self._load_and_analyze(request.image_path)
            
            # Route to appropriate processor
            if request.mode == ProcessingMode.CORRECT:
                result = self._process_correction(image_data, request)
            elif request.mode == ProcessingMode.SYNTHESIZE:
                result = self._process_synthesis(image_data, request)
            else:  # HYBRID
                result = self._process_hybrid(image_data, request)
            
            # Post-processing and save
            if request.output_path:
                self._save_result(result, request)
            
            return result
    
    def _process_correction(self, image_data: ImageData, 
                          request: ProcessingRequest) -> ProcessingResult:
        """Correction pipeline with depth awareness"""
        
        # Detect lens and corrections already applied
        lens_profile = self.detector.detect_comprehensive(image_data)
        
        # Check for existing corrections
        applied_corrections = self._detect_applied_corrections(image_data)
        
        # Get correction parameters
        params = self._calculate_correction_params(
            lens_profile, 
            applied_corrections,
            request.settings
        )
        
        # Apply corrections based on depth
        if request.use_depth and self._should_use_depth(image_data):
            corrected = self.depth_processor.process_with_layers(
                image_data.image,
                image_data.depth_map,
                params
            )
        else:
            corrected = self.optics_engine.apply_corrections(
                image_data.image,
                params
            )
        
        # Statistical cleanup
        final = self.cleanup_engine.clean_with_preservation(
            corrected,
            lens_profile,
            image_data.depth_map
        )
        
        return ProcessingResult(
            image=final,
            mode=ProcessingMode.CORRECT,
            lens_profile=lens_profile,
            quality_metrics=self.quality_analyzer.analyze(
                image_data.image, final
            )
        )
    
    def _process_synthesis(self, image_data: ImageData,
                         request: ProcessingRequest) -> ProcessingResult:
        """Synthesis pipeline for applying lens characteristics"""
        
        # Detect source lens if not specified
        if not request.source_lens:
            source_profile = self.detector.detect_comprehensive(image_data)
        else:
            source_profile = self._load_lens_profile(request.source_lens)
        
        # Load target lens profile
        target_profile = self._load_lens_profile(request.target_lens)
        
        # Synthesize characteristics
        synthesized = self.synthesizer.apply_lens_character(
            image_data.image,
            source_profile,
            target_profile,
            image_data.depth_map,
            request.settings
        )
        
        return ProcessingResult(
            image=synthesized,
            mode=ProcessingMode.SYNTHESIZE,
            source_lens=source_profile,
            target_lens=target_profile,
            synthesis_params=self.synthesizer.get_synthesis_report()
        )
    
    def batch_process(self, input_dir: str, output_dir: str,
                     mode: ProcessingMode = ProcessingMode.CORRECT,
                     **kwargs) -> BatchResult:
        """Process multiple images efficiently"""
        
        # Discover images
        images = self._discover_images(input_dir)
        
        # Group by lens for efficiency
        grouped = self._group_by_lens(images)
        
        # Process with optimizations
        results = []
        with self.performance_monitor.track("batch_processing"):
            for lens_group, group_images in grouped.items():
                # Cache lens profile
                self._cache_lens_profile(lens_group)
                
                # Process group in parallel if enabled
                if self.config.get('parallel_processing'):
                    group_results = self._parallel_process(
                        group_images, mode, **kwargs
                    )
                else:
                    group_results = [
                        self.process(ProcessingRequest(
                            img, mode, **kwargs
                        )) for img in group_images
                    ]
                
                results.extend(group_results)
        
        # Generate report
        return self._generate_batch_report(results, output_dir)
    
    def _setup_plugins(self):
        """Setup plugin system"""
        # Stub implementation
        pass
    
    def _load_and_analyze(self, image_path: str) -> ImageData:
        """Load and analyze image"""
        # Load image using OpenCV
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Try to load as regular image first
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        if image is None:
            # Try to load as RAW if available
            try:
                import rawpy
                with rawpy.imread(image_path) as raw:
                    image = raw.postprocess(use_camera_wb=True, half_size=False)
            except:
                raise ValueError(f"Could not load image: {image_path}")
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract metadata if possible
        metadata = self._extract_metadata(image_path)
        
        return ImageData(image=image, metadata=metadata)
    
    def _process_hybrid(self, image_data: ImageData, request: ProcessingRequest) -> ProcessingResult:
        """Hybrid processing pipeline"""
        # Stub implementation
        return ProcessingResult(image_data.image, ProcessingMode.HYBRID)
    
    def _save_result(self, result: ProcessingResult, request: ProcessingRequest):
        """Save processing result"""
        if request.output_path and hasattr(result, 'image'):
            # Convert RGB to BGR for OpenCV
            if len(result.image.shape) == 3:
                save_image = cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR)
            else:
                save_image = result.image
            
            # Determine format and quality
            ext = os.path.splitext(request.output_path)[1].lower()
            
            if ext in ['.jpg', '.jpeg']:
                cv2.imwrite(request.output_path, save_image, 
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif ext in ['.tiff', '.tif']:
                cv2.imwrite(request.output_path, save_image, 
                           [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            else:
                cv2.imwrite(request.output_path, save_image)
    
    def _detect_applied_corrections(self, image_data: ImageData):
        """Detect corrections already applied"""
        return {}
    
    def _calculate_correction_params(self, lens_profile, applied_corrections, settings):
        """Calculate correction parameters"""
        return {}
    
    def _should_use_depth(self, image_data: ImageData) -> bool:
        """Determine if depth processing should be used"""
        return True
    
    def _load_lens_profile(self, lens_id: str):
        """Load lens profile"""
        return {'lens_id': lens_id}
    
    def _discover_images(self, input_dir: str):
        """Discover images in directory"""
        return []
    
    def _group_by_lens(self, images):
        """Group images by lens"""
        return {}
    
    def _cache_lens_profile(self, lens_group):
        """Cache lens profile"""
        pass
    
    def _parallel_process(self, images, mode, **kwargs):
        """Process images in parallel"""
        return []
    
    def _generate_batch_report(self, results, output_dir):
        """Generate batch processing report"""
        return BatchResult(results, {'processed': len(results)})
    
    def _extract_metadata(self, image_path: str) -> Dict:
        """Extract metadata from image file"""
        # Basic metadata extraction
        metadata = {'file_path': image_path}
        
        # Try to get EXIF data
        try:
            import PIL.Image
            import PIL.ExifTags
            
            with PIL.Image.open(image_path) as img:
                exif = img._getexif()
                if exif:
                    for tag_id, value in exif.items():
                        tag = PIL.ExifTags.TAGS.get(tag_id, tag_id)
                        metadata[tag] = value
        except Exception:
            pass  # EXIF extraction failed, continue without it
        
        return metadata
    
    def _calculate_correction_params(self, lens_profile, applied_corrections, settings):
        """Calculate correction parameters based on lens profile"""
        params = {}
        
        # Get typical corrections for this lens type
        if lens_profile.get('lens_id') == 'helios_44_58mm':
            params.update({
                'distortion_k1': 0.015,
                'distortion_k2': -0.002,
                'chromatic_red': 1.01,
                'chromatic_blue': 0.99
            })
        elif lens_profile.get('lens_id') == 'canon_50mm_f14':
            params.update({
                'distortion_k1': -0.02,
                'distortion_k2': 0.001,
                'vignetting_a1': 0.2,
                'vignetting_a2': -0.1
            })
        else:
            # Generic corrections
            params.update({
                'distortion_k1': 0.01,
                'chromatic_red': 1.005,
                'chromatic_blue': 0.995,
                'vignetting_a1': 0.1
            })
        
        # Apply user settings override
        if settings:
            params.update(settings)
        
        return params