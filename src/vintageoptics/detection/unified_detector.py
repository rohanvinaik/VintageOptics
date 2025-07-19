"""
Unified detector combining vintage and electronic lens detection with HD analysis.

This module provides a unified interface for detecting lens types and
characteristics using both traditional and hyperdimensional methods.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import cv2
from pathlib import Path
import logging

from .vintage_detector import VintageDetector
from .electronic_detector import ElectronicDetector
from .lens_fingerprinting import LensFingerprinter
from ..hyperdimensional import HyperdimensionalEncoder, TopologicalDefectAnalyzer
from ..types.optics import LensType

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Unified detection result."""
    lens_type: LensType
    confidence: float
    vintage_probability: float
    electronic_probability: float
    
    # Specific detections
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    era: Optional[str] = None
    
    # Characteristics
    coating_type: Optional[str] = None
    optical_design: Optional[str] = None
    
    # HD analysis
    hypervector: Optional[np.ndarray] = None
    topological_features: Optional[List] = None
    
    # Detailed scores
    feature_scores: Dict[str, float] = None
    
    def __post_init__(self):
        if self.feature_scores is None:
            self.feature_scores = {}


class UnifiedDetector:
    """
    Unified lens detection system combining multiple detection methods
    with hyperdimensional analysis for robust lens identification.
    """
    
    def __init__(self, use_hd: bool = True):
        """
        Initialize unified detector.
        
        Args:
            use_hd: Whether to use hyperdimensional analysis
        """
        self.use_hd = use_hd
        
        # Component detectors
        self.vintage_detector = VintageDetector()
        self.electronic_detector = ElectronicDetector()
        self.fingerprinter = LensFingerprinter()
        
        # HD components
        if use_hd:
            self.hd_encoder = HyperdimensionalEncoder()
            self.topo_analyzer = TopologicalDefectAnalyzer(self.hd_encoder)
        else:
            self.hd_encoder = None
            self.topo_analyzer = None
        
        # Known lens database
        self.lens_database = self._load_lens_database()
        
    def detect(self, image: Union[str, Path, np.ndarray]) -> DetectionResult:
        """
        Perform unified lens detection.
        
        Args:
            image: Input image
            
        Returns:
            Comprehensive detection result
        """
        # Load image if needed
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Could not load image: {image}")
        
        logger.info("Performing unified lens detection...")
        
        # Initialize result
        result = DetectionResult(
            lens_type=LensType.UNKNOWN,
            confidence=0.0,
            vintage_probability=0.0,
            electronic_probability=0.0
        )
        
        # Run component detections
        self._run_vintage_detection(image, result)
        self._run_electronic_detection(image, result)
        
        # HD analysis if enabled
        if self.use_hd:
            self._run_hd_analysis(image, result)
        
        # Fingerprinting
        self._run_fingerprinting(image, result)
        
        # Determine final lens type
        self._determine_lens_type(result)
        
        # Match against known lenses
        self._match_known_lens(result)
        
        # Calculate overall confidence
        result.confidence = self._calculate_confidence(result)
        
        logger.info(f"Detection complete: {result.lens_type.value} "
                   f"(confidence: {result.confidence:.2%})")
        
        return result
    
    def _run_vintage_detection(self, image: np.ndarray, result: DetectionResult):
        """Run vintage lens detection."""
        vintage_score = self.vintage_detector.detect(image)
        result.vintage_probability = vintage_score
        
        # Get detailed vintage features
        vintage_features = self._extract_vintage_features(image)
        result.feature_scores.update({
            f"vintage_{k}": v for k, v in vintage_features.items()
        })
        
        # Detect specific vintage characteristics
        if vintage_score > 0.5:
            self._detect_vintage_specifics(image, result)
    
    def _extract_vintage_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract vintage-specific features."""
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Coating flare detection
        features['coating_flare'] = self._detect_coating_flare(image)
        
        # Aperture blade count (from bokeh shape)
        features['aperture_blades'] = self._estimate_aperture_blades(image)
        
        # Optical formula hints
        features['element_count'] = self._estimate_element_count(image)
        
        # Age indicators
        features['yellowing'] = self._detect_yellowing(image)
        features['contrast_fade'] = self._detect_contrast_fade(gray)
        
        return features
    
    def _detect_coating_flare(self, image: np.ndarray) -> float:
        """Detect characteristic coating flare patterns."""
        # Look for purple/green flare characteristic of older coatings
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Purple range
        purple_lower = np.array([120, 50, 50])
        purple_upper = np.array([150, 255, 255])
        purple_mask = cv2.inRange(hsv, purple_lower, purple_upper)
        
        # Green range  
        green_lower = np.array([40, 50, 50])
        green_upper = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Combined flare
        flare_mask = purple_mask | green_mask
        
        # Check for circular patterns (lens flare)
        circles = cv2.HoughCircles(
            flare_mask,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=100
        )
        
        if circles is not None:
            return min(1.0, len(circles[0]) / 5)  # Normalize
        
        return 0.0
    
    def _estimate_aperture_blades(self, image: np.ndarray) -> float:
        """Estimate aperture blade count from bokeh shape."""
        # Find bright out-of-focus areas
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blade_counts = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Significant bokeh ball
                # Approximate polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Number of vertices hints at blade count
                vertices = len(approx)
                if 5 <= vertices <= 12:  # Reasonable range
                    blade_counts.append(vertices)
        
        if blade_counts:
            # Most common blade count
            most_common = max(set(blade_counts), key=blade_counts.count)
            return most_common / 10.0  # Normalize
        
        return 0.5  # Default
    
    def _estimate_element_count(self, image: np.ndarray) -> float:
        """Estimate optical element count from ghost reflections."""
        # Look for internal reflections
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Find bright spots
        _, bright = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to separate spots
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
        
        # Count spots
        num_labels, _ = cv2.connectedComponents(opened)
        
        # Each surface can create a ghost, so elements ≈ ghosts / 2
        estimated_elements = (num_labels - 1) / 2
        
        # Normalize (typical vintage lenses have 4-8 elements)
        return min(1.0, estimated_elements / 8)
    
    def _detect_yellowing(self, image: np.ndarray) -> float:
        """Detect yellowing from aged optical elements."""
        if len(image.shape) != 3:
            return 0.0
        
        # Check color cast in LAB space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Yellowing shifts b channel positive
        b_mean = np.mean(b)
        
        # Expected neutral is 128
        yellowing = max(0, (b_mean - 128) / 128)
        
        return float(min(1.0, yellowing * 2))  # Scale up
    
    def _detect_contrast_fade(self, gray: np.ndarray) -> float:
        """Detect contrast fade from aging."""
        # Measure dynamic range
        p5, p95 = np.percentile(gray, [5, 95])
        dynamic_range = p95 - p5
        
        # Full range would be ~242 (255 * 0.95 - 255 * 0.05)
        contrast_fade = 1 - (dynamic_range / 242)
        
        return float(max(0, contrast_fade))
    
    def _detect_vintage_specifics(self, image: np.ndarray, result: DetectionResult):
        """Detect specific vintage lens characteristics."""
        # Era detection based on coating and design
        coating_flare = result.feature_scores.get('vintage_coating_flare', 0)
        yellowing = result.feature_scores.get('vintage_yellowing', 0)
        
        if coating_flare < 0.2 and yellowing < 0.1:
            result.era = "1980s-1990s"  # Multi-coated era
            result.coating_type = "Multi-coated"
        elif coating_flare > 0.5:
            result.era = "1950s-1960s"  # Single-coated era
            result.coating_type = "Single-coated"
        else:
            result.era = "1960s-1970s"  # Early multi-coating
            result.coating_type = "Early multi-coated"
        
        # Optical design hints
        blades = result.feature_scores.get('vintage_aperture_blades', 0.5) * 10
        if blades >= 8:
            result.optical_design = "Premium design"
        elif blades <= 5:
            result.optical_design = "Budget design"
        else:
            result.optical_design = "Standard design"
    
    def _run_electronic_detection(self, image: np.ndarray, result: DetectionResult):
        """Run electronic lens detection."""
        electronic_score = self.electronic_detector.detect(image)
        result.electronic_probability = electronic_score
        
        # Get detailed electronic features
        electronic_features = self._extract_electronic_features(image)
        result.feature_scores.update({
            f"electronic_{k}": v for k, v in electronic_features.items()
        })
    
    def _extract_electronic_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract electronic lens features."""
        features = {}
        
        # Digital sharpness
        features['digital_sharpness'] = self._measure_digital_sharpness(image)
        
        # Chromatic aberration correction
        features['ca_correction'] = self._detect_ca_correction(image)
        
        # Electronic vignetting correction
        features['vignette_correction'] = self._detect_vignette_correction(image)
        
        # In-camera processing artifacts
        features['processing_artifacts'] = self._detect_processing_artifacts(image)
        
        return features
    
    def _measure_digital_sharpness(self, image: np.ndarray) -> float:
        """Measure digital over-sharpening artifacts."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Look for halos around edges (oversharpening)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Check for bright halos
        edge_regions = dilated > 0
        non_edge = ~edge_regions
        
        if np.any(edge_regions) and np.any(non_edge):
            edge_mean = np.mean(gray[edge_regions])
            non_edge_mean = np.mean(gray[non_edge])
            
            # Oversharpening creates bright halos
            halo_strength = (edge_mean - non_edge_mean) / 255
            
            return float(max(0, min(1, halo_strength * 2)))
        
        return 0.0
    
    def _detect_ca_correction(self, image: np.ndarray) -> float:
        """Detect in-lens chromatic aberration correction."""
        if len(image.shape) != 3:
            return 0.0
        
        # Check channel alignment at edges
        edges_b = cv2.Canny(image[:, :, 0], 50, 150)
        edges_g = cv2.Canny(image[:, :, 1], 50, 150)
        edges_r = cv2.Canny(image[:, :, 2], 50, 150)
        
        # Perfect correction = perfect alignment
        alignment = np.mean(edges_b == edges_g) * np.mean(edges_g == edges_r)
        
        return float(alignment)
    
    def _detect_vignette_correction(self, image: np.ndarray) -> float:
        """Detect electronic vignetting correction."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        h, w = gray.shape
        center = (w // 2, h // 2)
        
        # Sample center and corners
        center_region = gray[h//2-50:h//2+50, w//2-50:w//2+50]
        corner_regions = [
            gray[:100, :100],  # Top-left
            gray[:100, -100:],  # Top-right
            gray[-100:, :100],  # Bottom-left
            gray[-100:, -100:]  # Bottom-right
        ]
        
        center_mean = np.mean(center_region)
        corner_mean = np.mean([np.mean(corner) for corner in corner_regions])
        
        # No vignetting = similar brightness
        vignette_ratio = corner_mean / (center_mean + 1e-8)
        
        # Strong correction can overcorrect (corners brighter than center)
        if vignette_ratio > 1.0:
            return 1.0  # Overcorrected
        elif vignette_ratio > 0.9:
            return 0.8  # Well corrected
        else:
            return vignette_ratio  # Partial or no correction
    
    def _detect_processing_artifacts(self, image: np.ndarray) -> float:
        """Detect in-camera processing artifacts."""
        # Look for typical digital processing signatures
        artifacts = 0.0
        
        # Noise reduction artifacts (loss of fine detail)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Check for smoothed texture
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = np.var(laplacian)
        
        # Very low texture variance indicates heavy NR
        if texture_variance < 10:
            artifacts += 0.3
        
        # Check for posterization
        hist, _ = np.histogram(gray, bins=256)
        
        # Gaps in histogram indicate posterization
        zero_bins = np.sum(hist == 0)
        if zero_bins > 100:  # Many missing levels
            artifacts += 0.3
        
        # JPEG artifacts
        # Check 8x8 block boundaries
        diff_h = np.diff(gray[::8, :], axis=0)
        diff_v = np.diff(gray[:, ::8], axis=1)
        
        block_artifacts = (np.mean(np.abs(diff_h)) + np.mean(np.abs(diff_v))) / 2
        if block_artifacts > 5:
            artifacts += 0.4
        
        return float(min(1.0, artifacts))
    
    def _run_hd_analysis(self, image: np.ndarray, result: DetectionResult):
        """Run hyperdimensional analysis."""
        logger.info("Running HD analysis...")
        
        # Topological analysis
        topo_result = self.topo_analyzer.analyze_defects(image)
        result.topological_features = (
            topo_result['dust_features'] +
            topo_result['scratch_features'] +
            topo_result['region_features']
        )
        
        # Store hypervector
        result.hypervector = topo_result['hypervector']
        
        # Add HD-based scores
        result.feature_scores['hd_defect_score'] = 1.0 / (1.0 + topo_result['total_features'] * 0.1)
        
    def _run_fingerprinting(self, image: np.ndarray, result: DetectionResult):
        """Run lens fingerprinting."""
        fingerprint = self.fingerprinter.generate_fingerprint(image)
        
        # Add fingerprint features to scores
        if fingerprint:
            result.feature_scores.update({
                f"fingerprint_{k}": v 
                for k, v in fingerprint.items() 
                if isinstance(v, (int, float))
            })
    
    def _determine_lens_type(self, result: DetectionResult):
        """Determine final lens type from all detections."""
        # Simple threshold-based decision
        if result.vintage_probability > 0.7:
            result.lens_type = LensType.VINTAGE_MANUAL
        elif result.electronic_probability > 0.7:
            result.lens_type = LensType.MODERN_ELECTRONIC
        elif result.vintage_probability > result.electronic_probability:
            result.lens_type = LensType.ADAPTED_VINTAGE
        else:
            result.lens_type = LensType.MODERN_ELECTRONIC
    
    def _match_known_lens(self, result: DetectionResult):
        """Match against known lens database."""
        if not self.lens_database or result.hypervector is None:
            return
        
        best_match = None
        best_similarity = 0.0
        
        for lens_id, lens_data in self.lens_database.items():
            if 'hypervector' in lens_data and self.hd_encoder:
                similarity = self.hd_encoder.similarity(
                    result.hypervector,
                    lens_data['hypervector']
                )
                
                if similarity > best_similarity and similarity > 0.8:
                    best_similarity = similarity
                    best_match = lens_id
        
        if best_match:
            lens_info = self.lens_database[best_match]
            result.manufacturer = lens_info.get('manufacturer')
            result.model = lens_info.get('model')
            logger.info(f"Matched lens: {result.manufacturer} {result.model}")
    
    def _calculate_confidence(self, result: DetectionResult) -> float:
        """Calculate overall detection confidence."""
        # Base confidence on primary detection
        if result.lens_type == LensType.VINTAGE_MANUAL:
            base_confidence = result.vintage_probability
        elif result.lens_type == LensType.MODERN_ELECTRONIC:
            base_confidence = result.electronic_probability
        else:
            base_confidence = max(result.vintage_probability, result.electronic_probability)
        
        # Boost confidence if specific lens matched
        if result.manufacturer and result.model:
            base_confidence = base_confidence * 0.8 + 0.2
        
        # Consider feature consistency
        feature_consistency = self._calculate_feature_consistency(result)
        
        # Final confidence
        confidence = base_confidence * 0.7 + feature_consistency * 0.3
        
        return float(confidence)
    
    def _calculate_feature_consistency(self, result: DetectionResult) -> float:
        """Calculate how consistent the detected features are."""
        if not result.feature_scores:
            return 0.5
        
        # Group features by type
        vintage_features = [v for k, v in result.feature_scores.items() if k.startswith('vintage_')]
        electronic_features = [v for k, v in result.feature_scores.items() if k.startswith('electronic_')]
        
        # Consistency = low variance within groups
        vintage_var = np.var(vintage_features) if vintage_features else 0
        electronic_var = np.var(electronic_features) if electronic_features else 0
        
        # Lower variance = higher consistency
        consistency = 1.0 / (1.0 + vintage_var + electronic_var)
        
        return float(consistency)
    
    def _load_lens_database(self) -> Dict[str, Dict]:
        """Load known lens database."""
        # In production, would load from actual database
        # For now, return example data
        return {
            "canon_fd_50_14": {
                "manufacturer": "Canon",
                "model": "FD 50mm f/1.4",
                "type": "vintage_manual",
                "era": "1970s-1980s",
                "coating": "S.S.C."
            },
            "nikon_ai_50_14": {
                "manufacturer": "Nikon",
                "model": "AI 50mm f/1.4",
                "type": "vintage_manual",
                "era": "1970s-1980s",
                "coating": "NIC"
            },
            "helios_44_2": {
                "manufacturer": "KMZ",
                "model": "Helios 44-2 58mm f/2",
                "type": "vintage_manual",
                "era": "1960s-1980s",
                "coating": "Single-coated"
            }
        }
    
    def add_to_database(self, 
                       lens_id: str,
                       manufacturer: str,
                       model: str,
                       sample_images: List[np.ndarray]):
        """Add a new lens to the database."""
        if self.use_hd and self.hd_encoder:
            # Generate hypervector from samples
            hypervectors = []
            for img in sample_images:
                topo = self.topo_analyzer.analyze_defects(img)
                hypervectors.append(topo['hypervector'])
            
            # Average hypervector
            avg_hv = np.mean(hypervectors, axis=0)
            avg_hv /= np.linalg.norm(avg_hv)
            
            self.lens_database[lens_id] = {
                "manufacturer": manufacturer,
                "model": model,
                "hypervector": avg_hv
            }
            
            logger.info(f"Added {manufacturer} {model} to database")


# Convenience function
def detect_lens(image_path: str) -> DetectionResult:
    """
    Quick lens detection.
    
    Args:
        image_path: Path to image
        
    Returns:
        Detection result
    """
    detector = UnifiedDetector(use_hd=True)
    return detector.detect(image_path)
