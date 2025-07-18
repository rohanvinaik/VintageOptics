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
from .hybrid_pipeline import HybridPhysicsMLPipeline
from ..types.io import ImageData, ProcessingResult, BatchResult


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
        
        # Hybrid pipeline for advanced processing
        self.hybrid_pipeline = HybridPhysicsMLPipeline(self.config)
        
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
    
    def _process_hybrid(self, image_data: ImageData, request: ProcessingRequest) -> ProcessingResult:
        """Hybrid processing pipeline using physics-ML iteration"""
        
        # Prepare metadata for hybrid pipeline
        metadata = image_data.metadata.copy()
        if hasattr(image_data, 'exif'):
            metadata['exif'] = image_data.exif
            
        # Run hybrid pipeline
        result = self.hybrid_pipeline.process(
            image_data.image,
            metadata
        )
        
        # Enhance result with additional analysis
        result.quality_metrics = self.quality_analyzer.analyze(
            image_data.image, result.corrected_image
        )
        
        return result
    
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
    
    def train_vintage_ml(self, training_data_dir: str, annotations_file: Optional[str] = None):
        """Train the vintage ML components for hybrid processing"""
        
        # Load training images
        training_images = []
        image_files = self._discover_images(training_data_dir)
        
        for img_path in image_files[:100]:  # Limit for initial training
            try:
                img_data = self._load_and_analyze(img_path)
                training_images.append(img_data.image)
            except Exception as e:
                logger.warning(f"Failed to load training image {img_path}: {e}")
                
        # Load annotations if provided
        defect_annotations = None
        if annotations_file and os.path.exists(annotations_file):
            import json
            with open(annotations_file, 'r') as f:
                defect_annotations = json.load(f)
                
        # Train the hybrid pipeline's ML components
        self.hybrid_pipeline.train_vintage_ml(training_images, defect_annotations)
        
        logger.info(f"Trained vintage ML on {len(training_images)} images")
    
    def _setup_plugins(self):
        """Setup plugin system"""
        # Initialize plugin registry
        self.plugins = {
            'pre_processors': [],
            'post_processors': [],
            'detectors': [],
            'correctors': []
        }
        
        # Load built-in plugins
        from .plugin_system import PluginLoader
        if hasattr(self, 'config') and self.config.get('plugins', {}).get('enabled', False):
            plugin_loader = PluginLoader(self.config.get('plugins', {}).get('directory', 'plugins'))
            self.plugins = plugin_loader.load_all()
    
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
        
        # Estimate depth if enabled
        depth_map = None
        if self.config.get('depth', {}).get('enabled', False):
            try:
                depth_map = self.depth_processor.estimate_depth(image)
            except Exception as e:
                logger.warning(f"Depth estimation failed: {e}")
        
        return ImageData(
            image=image, 
            metadata=metadata,
            depth_map=depth_map
        )
    
    def _save_result(self, result: ProcessingResult, request: ProcessingRequest):
        """Save processing result"""
        # Handle the corrected_image attribute from hybrid pipeline
        if hasattr(result, 'corrected_image'):
            save_image = result.corrected_image
        elif hasattr(result, 'image'):
            save_image = result.image
        else:
            logger.warning("No image to save in result")
            return
            
        if request.output_path:
            # Convert RGB to BGR for OpenCV
            if len(save_image.shape) == 3:
                save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
            
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
                
            # Save metadata if requested
            if request.preserve_metadata and hasattr(result, 'metadata'):
                import json
                meta_path = request.output_path.rsplit('.', 1)[0] + '_metadata.json'
                with open(meta_path, 'w') as f:
                    json.dump(result.metadata, f, indent=2, default=str)
    
    def _detect_applied_corrections(self, image_data: ImageData) -> Dict:
        """Detect corrections already applied using ML and heuristics"""
        
        # Check metadata for correction flags
        corrections = {}
        metadata = image_data.metadata
        
        # Look for software tags indicating prior processing
        software = metadata.get('Software', '')
        if 'lightroom' in software.lower():
            corrections['prior_processing'] = 'lightroom'
        elif 'darktable' in software.lower():
            corrections['prior_processing'] = 'darktable'
            
        # Use vintage ML to detect characteristic patterns
        if self.hybrid_pipeline.ml_trained:
            # Analyze for telltale signs of correction
            patches = self.hybrid_pipeline._extract_patches(image_data.image)
            
            # Look for unnaturally uniform areas (over-correction)
            uniformity_scores = []
            for patch in patches[:20]:  # Sample patches
                uniformity = 1.0 - (patch.std() / (patch.mean() + 1e-8))
                uniformity_scores.append(uniformity)
                
            avg_uniformity = np.mean(uniformity_scores)
            if avg_uniformity > 0.8:
                corrections['over_corrected'] = True
                
        return corrections
    
    def _calculate_correction_params(self, lens_profile: Dict, 
                                   applied_corrections: Dict, 
                                   settings: Optional[Dict]) -> Dict:
        """Calculate correction parameters based on lens profile and prior corrections"""
        params = {}
        
        # Get base parameters for lens type
        lens_id = lens_profile.get('lens_id', 'unknown')
        
        # Load from lens database or use defaults
        if lens_id == 'helios_44_58mm':
            params.update({
                'distortion_k1': 0.015,
                'distortion_k2': -0.002,
                'chromatic_red': 1.01,
                'chromatic_blue': 0.99,
                'vignetting_a1': 0.3,
                'vignetting_a2': -0.15,
                'correct_distortion': True,
                'correct_chromatic': True,
                'correct_vignetting': True,
                'deconvolve': True
            })
        elif lens_id == 'canon_50mm_f14':
            params.update({
                'distortion_k1': -0.02,
                'distortion_k2': 0.001,
                'vignetting_a1': 0.2,
                'vignetting_a2': -0.1,
                'chromatic_red': 1.005,
                'chromatic_blue': 0.995,
                'correct_distortion': True,
                'correct_chromatic': True,
                'correct_vignetting': True,
                'deconvolve': False
            })
        else:
            # Generic vintage lens defaults
            params.update({
                'distortion_k1': 0.01,
                'chromatic_red': 1.005,
                'chromatic_blue': 0.995,
                'vignetting_a1': 0.1,
                'correct_distortion': True,
                'correct_chromatic': True,
                'correct_vignetting': True,
                'deconvolve': False
            })
        
        # Adjust based on prior corrections
        if applied_corrections.get('over_corrected'):
            # Reduce correction strength
            for key in ['distortion_k1', 'distortion_k2']:
                if key in params:
                    params[key] *= 0.5
                    
        # Apply user settings override
        if settings:
            params.update(settings)
        
        return params
    
    def _should_use_depth(self, image_data: ImageData) -> bool:
        """Determine if depth processing should be used"""
        # Use depth if available and image has sufficient detail
        if image_data.depth_map is None:
            return False
            
        # Check if image has depth variation
        if hasattr(image_data, 'depth_map'):
            depth_range = image_data.depth_map.max() - image_data.depth_map.min()
            return depth_range > 0.1  # Threshold for meaningful depth
            
        return False
    
    def _load_lens_profile(self, lens_id: str) -> Dict:
        """Load lens profile from database or library"""
        # This would query the lens database
        # For now, return mock profiles
        profiles = {
            'helios_44_58mm': {
                'lens_id': 'helios_44_58mm',
                'name': 'Helios 44-2 58mm f/2',
                'type': 'vintage',
                'mount': 'M42',
                'character': {
                    'swirly_bokeh': 0.8,
                    'sharpness_center': 0.9,
                    'sharpness_edge': 0.4,
                    'chromatic_aberration': 0.6,
                    'vignetting': 0.7
                }
            },
            'canon_50mm_f14': {
                'lens_id': 'canon_50mm_f14',
                'name': 'Canon FD 50mm f/1.4',
                'type': 'vintage',
                'mount': 'Canon FD',
                'character': {
                    'swirly_bokeh': 0.2,
                    'sharpness_center': 0.95,
                    'sharpness_edge': 0.7,
                    'chromatic_aberration': 0.3,
                    'vignetting': 0.5
                }
            }
        }
        
        return profiles.get(lens_id, {'lens_id': lens_id})
    
    def _discover_images(self, input_dir: str) -> List[str]:
        """Discover images in directory"""
        supported_formats = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', 
                           '.bmp', '.dng', '.cr2', '.nef', '.arw'}
        
        images = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in supported_formats:
                    images.append(os.path.join(root, file))
                    
        return sorted(images)
    
    def _group_by_lens(self, images: List[str]) -> Dict[str, List[str]]:
        """Group images by detected lens for batch efficiency"""
        groups = {}
        
        for img_path in images:
            try:
                # Quick lens detection from EXIF
                metadata = self._extract_metadata(img_path)
                lens_info = metadata.get('LensModel', 'unknown')
                
                if lens_info not in groups:
                    groups[lens_info] = []
                groups[lens_info].append(img_path)
                
            except Exception:
                # If detection fails, put in unknown group
                if 'unknown' not in groups:
                    groups['unknown'] = []
                groups['unknown'].append(img_path)
                
        return groups
    
    def _cache_lens_profile(self, lens_group: str):
        """Cache lens profile for batch processing"""
        # Check if already cached
        if not hasattr(self, '_lens_cache'):
            self._lens_cache = {}
            
        if lens_group not in self._lens_cache:
            # Load from database or generate
            profile = self._load_lens_profile(lens_group)
            self._lens_cache[lens_group] = profile
            
        return self._lens_cache[lens_group]
    
    def _parallel_process(self, images: List[str], mode: ProcessingMode, **kwargs) -> List:
        """Process images in parallel using multiprocessing"""
        from multiprocessing import Pool, cpu_count
        
        # Prepare process function
        def process_single(img_path):
            try:
                request = ProcessingRequest(
                    image_path=img_path,
                    mode=mode,
                    **kwargs
                )
                return self.process(request)
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")
                return None
                
        # Use process pool
        n_workers = min(cpu_count(), len(images))
        with Pool(n_workers) as pool:
            results = pool.map(process_single, images)
            
        # Filter out failures
        return [r for r in results if r is not None]
    
    def _generate_batch_report(self, results: List[ProcessingResult], 
                             output_dir: str) -> BatchResult:
        """Generate batch processing report"""
        report = {
            'total_processed': len(results),
            'successful': sum(1 for r in results if hasattr(r, 'corrected_image')),
            'processing_times': {},
            'quality_improvements': {},
            'detected_lenses': {}
        }
        
        # Aggregate statistics
        for result in results:
            if hasattr(result, 'lens_info'):
                lens = result.lens_info.get('name', 'unknown')
                report['detected_lenses'][lens] = report['detected_lenses'].get(lens, 0) + 1
                
            if hasattr(result, 'performance_metrics'):
                for metric, value in result.performance_metrics.items():
                    if metric not in report['processing_times']:
                        report['processing_times'][metric] = []
                    report['processing_times'][metric].append(value)
                    
        # Save report
        import json
        report_path = os.path.join(output_dir, 'batch_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        return BatchResult(results, report)
    
    def _extract_metadata(self, image_path: str) -> Dict:
        """Extract metadata from image file"""
        metadata = {'file_path': image_path}
        
        # Try EXIF extraction
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
            pass
            
        # Try ExifTool for more complete metadata
        try:
            from ..integrations import ExifToolIntegration
            exiftool = ExifToolIntegration()
            detailed_meta = exiftool.extract_metadata(image_path)
            metadata.update(detailed_meta)
        except Exception:
            pass
            
        return metadata


# Import logger
import logging
logger = logging.getLogger(__name__)
