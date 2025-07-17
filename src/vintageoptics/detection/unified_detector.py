# src/vintageoptics/detection/unified_detector.py
"""
Unified lens detection system combining EXIF, Lensfun, fingerprinting,
and manual lens detection for comprehensive lens identification.
"""

import numpy as np
import cv2
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
import logging
from pathlib import Path

from .metadata_extractor import MetadataExtractor
from .lens_fingerprinting import LensFingerprinting, OpticalFingerprint
from .base_detector import BaseLensDetector
from .electronic_detector import ElectronicLensDetector
from .vintage_detector import VintageLensDetector
from ..integrations.lensfun import LensfunIntegration
from ..database.repositories.lens_repo import LensRepository
from ..database.repositories.instance_repo import InstanceRepository

logger = logging.getLogger(__name__)


@dataclass
class LensDetectionResult:
    """Comprehensive lens detection result"""
    # Basic identification
    lens_id: str
    manufacturer: str
    model: str
    confidence: float
    
    # Detection method
    detection_method: str  # 'electronic', 'lensfun', 'fingerprint', 'manual'
    
    # Detailed info
    lens_type: str  # 'electronic', 'manual', 'adapted'
    mount_type: Optional[str] = None
    serial_number: Optional[str] = None
    
    # Optical specifications
    focal_length: Optional[float] = None
    focal_range: Optional[Tuple[float, float]] = None
    aperture: Optional[float] = None
    aperture_range: Optional[Tuple[float, float]] = None
    
    # Instance data
    instance_id: Optional[str] = None
    fingerprint: Optional[OpticalFingerprint] = None
    
    # Correction parameters
    correction_params: Optional[Dict[str, Any]] = None
    lensfun_profile: Optional[Any] = None
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class UnifiedLensDetector:
    """Unified lens detection combining all detection methods"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize components
        self.metadata_extractor = MetadataExtractor()
        self.fingerprinting = LensFingerprinting(config)
        self.lensfun = LensfunIntegration(config)
        
        # Specialized detectors
        self.electronic_detector = ElectronicLensDetector(config)
        self.vintage_detector = VintageLensDetector(config)
        
        # Database repositories
        self.lens_repo = LensRepository(config.get('database', {}))
        self.instance_repo = InstanceRepository(config.get('database', {}))
        
        # Known fingerprints cache
        self.known_fingerprints = []
        self._load_known_fingerprints()
    
    def detect_comprehensive(self, image_data: Dict) -> LensDetectionResult:
        """Perform comprehensive lens detection using all available methods"""
        
        logger.info("Starting comprehensive lens detection")
        
        # Extract image and metadata
        image = image_data.get('image')
        image_path = image_data.get('path')
        
        # 1. Extract all metadata
        metadata = {}
        if image_path:
            metadata = self.metadata_extractor.extract_all_metadata(image_path)
        
        # 2. Try electronic lens detection first
        electronic_result = self._detect_electronic_lens(metadata)
        if electronic_result and electronic_result.confidence > 0.8:
            logger.info(f"Electronic lens detected: {electronic_result.model}")
            
            # Enhance with Lensfun data if available
            self._enhance_with_lensfun(electronic_result)
            
            # Check for known instance
            self._check_instance(electronic_result, image)
            
            return electronic_result
        
        # 3. Try fingerprint matching
        fingerprint_result = self._detect_by_fingerprint(image)
        if fingerprint_result and fingerprint_result.confidence > 0.85:
            logger.info(f"Lens identified by fingerprint: {fingerprint_result.model}")
            return fingerprint_result
        
        # 4. Try Lensfun database
        lensfun_result = self._detect_from_lensfun(metadata)
        if lensfun_result and lensfun_result.confidence > 0.7:
            logger.info(f"Lens found in Lensfun: {lensfun_result.model}")
            return lensfun_result
        
        # 5. Try manual/vintage lens detection
        manual_result = self._detect_manual_lens(image, metadata)
        if manual_result:
            logger.info(f"Manual lens detected: {manual_result.model}")
            return manual_result
        
        # 6. Create unknown lens profile
        logger.warning("Could not identify lens, creating unknown profile")
        return self._create_unknown_profile(image, metadata)
    
    def _detect_electronic_lens(self, metadata: Dict) -> Optional[LensDetectionResult]:
        """Detect electronic lens from metadata"""
        
        lens_info = metadata.get('lens_info', {})
        
        if not lens_info.get('model'):
            return None
        
        # Create detection result
        result = LensDetectionResult(
            lens_id=self._generate_lens_id(lens_info),
            manufacturer=lens_info.get('make', 'Unknown'),
            model=lens_info.get('model', 'Unknown'),
            confidence=0.95,
            detection_method='electronic',
            lens_type='electronic',
            serial_number=lens_info.get('serial_number'),
            metadata=metadata
        )
        
        # Extract specifications
        if lens_info.get('focal_length'):
            result.focal_length = lens_info['focal_length']
        
        # Check if zoom lens
        if ' zoom' in result.model.lower() or '-' in result.model:
            # Try to extract focal range from model name
            import re
            range_match = re.search(r'(\d+)-(\d+)mm', result.model)
            if range_match:
                result.focal_range = (
                    float(range_match.group(1)),
                    float(range_match.group(2))
                )
        
        # Extract aperture info
        shooting_info = metadata.get('shooting_info', {})
        if shooting_info.get('aperture'):
            result.aperture = shooting_info['aperture']
        
        if lens_info.get('max_aperture'):
            result.aperture_range = (
                lens_info['max_aperture'],
                22.0  # Typical minimum
            )
        
        # Extract mount type
        result.mount_type = self._detect_mount_type(lens_info, metadata)
        
        return result
    
    def _detect_by_fingerprint(self, image: np.ndarray) -> Optional[LensDetectionResult]:
        """Detect lens by optical fingerprint matching"""
        
        if not self.known_fingerprints:
            return None
        
        # Try to match against known fingerprints
        match_result = self.fingerprinting.match_fingerprint(
            image, self.known_fingerprints, threshold=0.85
        )
        
        if match_result:
            fingerprint, confidence = match_result
            
            # Look up lens info from database
            lens_info = self.lens_repo.get_by_id(fingerprint.lens_model)
            
            if lens_info:
                result = LensDetectionResult(
                    lens_id=lens_info['lens_id'],
                    manufacturer=lens_info['manufacturer'],
                    model=lens_info['model'],
                    confidence=confidence,
                    detection_method='fingerprint',
                    lens_type=lens_info.get('lens_type', 'unknown'),
                    instance_id=fingerprint.instance_id,
                    fingerprint=fingerprint
                )
                
                # Add specifications
                result.focal_range = (
                    lens_info.get('focal_min_mm'),
                    lens_info.get('focal_max_mm')
                )
                result.aperture_range = (
                    lens_info.get('aperture_min'),
                    lens_info.get('aperture_max')
                )
                
                return result
        
        return None
    
    def _detect_from_lensfun(self, metadata: Dict) -> Optional[LensDetectionResult]:
        """Detect lens using Lensfun database"""
        
        # Extract camera and lens hints
        camera_info = metadata.get('camera_info', {})
        camera_make = camera_info.get('make', '')
        
        # Try various sources for lens info
        lens_hints = []
        
        # From EXIF
        exif = metadata.get('exif', {})
        for field in ['LensModel', 'Lens', 'LensType']:
            if field in exif:
                lens_hints.append(str(exif[field]))
        
        # From user comments
        for field in ['UserComment', 'ImageDescription']:
            if field in exif:
                comment = str(exif[field])
                if len(comment) < 200:  # Reasonable length
                    lens_hints.append(comment)
        
        # Try to find lens in Lensfun
        for hint in lens_hints:
            # Parse manufacturer and model
            parts = hint.split()
            if len(parts) >= 2:
                possible_maker = parts[0]
                possible_model = ' '.join(parts[1:])
                
                profile = self.lensfun.find_lens(
                    possible_maker, possible_model,
                    metadata.get('shooting_info', {}).get('focal_length')
                )
                
                if profile:
                    result = LensDetectionResult(
                        lens_id=f"lensfun_{profile.maker}_{profile.model}".replace(' ', '_'),
                        manufacturer=profile.maker,
                        model=profile.model,
                        confidence=0.8,
                        detection_method='lensfun',
                        lens_type='manual' if profile.mount in ['M42', 'M39'] else 'electronic',
                        mount_type=profile.mount,
                        lensfun_profile=profile
                    )
                    
                    # Add specifications
                    result.focal_range = (profile.focal_min, profile.focal_max)
                    result.aperture_range = (profile.aperture_min, profile.aperture_max)
                    
                    # Get correction parameters
                    shooting_info = metadata.get('shooting_info', {})
                    if shooting_info.get('focal_length') and shooting_info.get('aperture'):
                        result.correction_params = self.lensfun.get_correction_parameters(
                            profile,
                            shooting_info['focal_length'],
                            shooting_info['aperture']
                        )
                    
                    return result
        
        return None
    
    def _detect_manual_lens(self, image: np.ndarray, 
                          metadata: Dict) -> Optional[LensDetectionResult]:
        """Detect manual/vintage lens"""
        
        # Use vintage detector
        vintage_features = self.vintage_detector.detect_vintage_characteristics(image)
        
        # Check metadata for manual lens hints
        manual_hint = metadata.get('lens_info', {}).get('manual_lens_detected')
        
        if vintage_features or manual_hint:
            # Combine evidence
            confidence = 0.0
            lens_type = 'unknown'
            
            if vintage_features:
                confidence += vintage_features.get('confidence', 0) * 0.6
                lens_type = vintage_features.get('lens_type', 'vintage')
            
            if manual_hint:
                confidence += manual_hint.get('confidence', 0) * 0.4
                if manual_hint.get('type'):
                    lens_type = manual_hint['type']
            
            confidence = min(confidence, 0.95)
            
            # Create result
            result = LensDetectionResult(
                lens_id=f"manual_{lens_type}_{hash(str(vintage_features))}"[:20],
                manufacturer='Unknown',
                model=f"Manual {lens_type.title()} Lens",
                confidence=confidence,
                detection_method='manual',
                lens_type='manual'
            )
            
            # Add detected specifications
            if manual_hint:
                if manual_hint.get('focal_length'):
                    result.focal_length = manual_hint['focal_length']
                if manual_hint.get('max_aperture'):
                    result.aperture_range = (manual_hint['max_aperture'], 22.0)
            
            # Add vintage characteristics
            if vintage_features:
                result.metadata = {'vintage_features': vintage_features}
                
                # Try to identify specific vintage lens
                if vintage_features.get('swirly_bokeh', 0) > 0.7:
                    result.model = "Helios-type Lens"
                    result.warnings.append("Swirly bokeh detected - possibly Helios family")
                elif vintage_features.get('radioactive_tint', 0) > 0.6:
                    result.model = "Vintage Radioactive Lens"
                    result.warnings.append("Yellow tint detected - possibly radioactive glass")
            
            return result
        
        return None
    
    def _create_unknown_profile(self, image: np.ndarray, 
                              metadata: Dict) -> LensDetectionResult:
        """Create profile for unknown lens"""
        
        # Extract any available info
        shooting_info = metadata.get('shooting_info', {})
        
        result = LensDetectionResult(
            lens_id='unknown_' + str(hash(str(metadata)))[:10],
            manufacturer='Unknown',
            model='Unknown Lens',
            confidence=0.1,
            detection_method='unknown',
            lens_type='unknown',
            metadata=metadata
        )
        
        # Add any shooting info we have
        if shooting_info.get('focal_length'):
            result.focal_length = shooting_info['focal_length']
        if shooting_info.get('aperture'):
            result.aperture = shooting_info['aperture']
        
        result.warnings.append("Could not identify lens from available information")
        
        return result
    
    def _enhance_with_lensfun(self, result: LensDetectionResult):
        """Enhance detection result with Lensfun data"""
        
        profile = self.lensfun.find_lens(
            result.manufacturer,
            result.model,
            result.focal_length
        )
        
        if profile:
            result.lensfun_profile = profile
            
            # Update mount if not set
            if not result.mount_type:
                result.mount_type = profile.mount
            
            # Get correction parameters
            if result.focal_length and result.aperture:
                result.correction_params = self.lensfun.get_correction_parameters(
                    profile,
                    result.focal_length,
                    result.aperture
                )
    
    def _check_instance(self, result: LensDetectionResult, image: np.ndarray):
        """Check if this is a known lens instance"""
        
        # Try to match by serial number first
        if result.serial_number:
            instance = self.instance_repo.get_by_serial(
                result.lens_id,
                result.serial_number
            )
            
            if instance:
                result.instance_id = instance['instance_id']
                
                # Load fingerprint if available
                if instance.get('optical_fingerprint'):
                    result.fingerprint = OpticalFingerprint.from_dict(
                        instance['optical_fingerprint']
                    )
                
                logger.info(f"Known instance found: {result.instance_id}")
                return
        
        # Try fingerprint matching
        if self.known_fingerprints:
            match_result = self.fingerprinting.match_fingerprint(
                image,
                [fp for fp in self.known_fingerprints 
                 if fp.lens_model == result.model],
                threshold=0.9
            )
            
            if match_result:
                fingerprint, confidence = match_result
                result.instance_id = fingerprint.instance_id
                result.fingerprint = fingerprint
                logger.info(f"Instance matched by fingerprint: {result.instance_id}")
    
    def _generate_lens_id(self, lens_info: Dict) -> str:
        """Generate unique lens ID"""
        
        parts = []
        
        if lens_info.get('make'):
            parts.append(lens_info['make'].replace(' ', '_'))
        
        if lens_info.get('model'):
            parts.append(lens_info['model'].replace(' ', '_'))
        else:
            parts.append('unknown')
        
        return '_'.join(parts).lower()
    
    def _detect_mount_type(self, lens_info: Dict, metadata: Dict) -> Optional[str]:
        """Detect lens mount type"""
        
        # Check lens info
        if 'mount' in lens_info:
            return lens_info['mount']
        
        # Check camera info for mount hints
        camera_info = metadata.get('camera_info', {})
        camera_model = camera_info.get('model', '').lower()
        
        # Mount detection patterns
        mount_patterns = {
            'canon': ['ef', 'ef-s', 'ef-m', 'rf'],
            'nikon': ['f', 'z'],
            'sony': ['e', 'fe', 'a'],
            'fujifilm': ['x'],
            'micro four thirds': ['mft', 'm43'],
            'pentax': ['k'],
            'leica': ['m', 'l']
        }
        
        for brand, mounts in mount_patterns.items():
            if brand in camera_model:
                # Further detection based on lens model
                lens_model = lens_info.get('model', '').lower()
                for mount in mounts:
                    if mount in lens_model:
                        return mount.upper()
                
                # Default mount for brand
                return mounts[0].upper()
        
        return None
    
    def _load_known_fingerprints(self):
        """Load known lens fingerprints from database"""
        
        try:
            instances = self.instance_repo.get_all_with_fingerprints()
            
            for instance in instances:
                if instance.get('optical_fingerprint'):
                    fingerprint = OpticalFingerprint.from_dict(
                        instance['optical_fingerprint']
                    )
                    self.known_fingerprints.append(fingerprint)
            
            logger.info(f"Loaded {len(self.known_fingerprints)} known fingerprints")
            
        except Exception as e:
            logger.error(f"Failed to load fingerprints: {e}")
    
    def create_instance_profile(self, 
                              detection_result: LensDetectionResult,
                              calibration_images: List[np.ndarray]) -> OpticalFingerprint:
        """Create instance profile for a detected lens"""
        
        lens_info = {
            'model': detection_result.model,
            'manufacturer': detection_result.manufacturer,
            'lens_id': detection_result.lens_id
        }
        
        # Create fingerprint
        fingerprint = self.fingerprinting.create_fingerprint(
            calibration_images,
            lens_info,
            detection_result.correction_params
        )
        
        # Store in database
        instance_data = {
            'instance_id': fingerprint.instance_id,
            'lens_id': detection_result.lens_id,
            'serial_number': detection_result.serial_number,
            'optical_fingerprint': fingerprint.to_dict(),
            'first_seen': fingerprint.creation_date
        }
        
        self.instance_repo.create(instance_data)
        
        # Add to known fingerprints
        self.known_fingerprints.append(fingerprint)
        
        return fingerprint
