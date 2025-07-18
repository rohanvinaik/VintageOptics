"""
Advanced lens characterization with hyperdimensional computing integration.

This module provides comprehensive lens analysis, defect detection, and
characteristic profiling using both traditional and HD computing methods.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import cv2
from pathlib import Path
import json
import logging

from ..hyperdimensional import (
    HyperdimensionalLensAnalyzer,
    analyze_lens_defects,
    TopologicalFeature
)
from ..detection import VintageDetector, ElectronicDetector
from ..physics.optics_engine import OpticsEngine
from ..types.optics import LensParameters, OpticalDefect

logger = logging.getLogger(__name__)


@dataclass
class LensCharacteristics:
    """Complete lens characteristic profile."""
    # Basic properties
    focal_length: float
    max_aperture: float
    min_aperture: float
    optical_formula: Optional[str] = None
    
    # Optical characteristics
    vignetting_profile: Optional[np.ndarray] = None
    distortion_coefficients: Optional[Dict[str, float]] = None
    chromatic_aberration: Optional[Dict[str, float]] = None
    bokeh_quality: Optional[Dict[str, float]] = None
    
    # Defects and artifacts
    defects: List[OpticalDefect] = field(default_factory=list)
    dust_spots: int = 0
    scratches: int = 0
    fungus_areas: int = 0
    haze_level: float = 0.0
    
    # HD signature
    hyperdimensional_signature: Optional[np.ndarray] = None
    topological_features: Optional[List[TopologicalFeature]] = None
    
    # Quality metrics
    sharpness_score: float = 0.0
    contrast_score: float = 0.0
    color_accuracy: float = 0.0
    overall_quality: float = 0.0
    
    # Metadata
    lens_model: Optional[str] = None
    serial_number: Optional[str] = None
    manufacturing_year: Optional[int] = None
    coating_type: Optional[str] = None


class LensCharacterizer:
    """
    Advanced lens characterization system combining traditional analysis
    with hyperdimensional computing for robust defect detection and
    lens signature matching.
    """
    
    def __init__(self, use_hd: bool = True, hd_dimension: int = 10000):
        """
        Initialize the lens characterizer.
        
        Args:
            use_hd: Whether to use hyperdimensional computing features
            hd_dimension: Dimension for hypervectors (if use_hd is True)
        """
        self.use_hd = use_hd
        self.hd_analyzer = HyperdimensionalLensAnalyzer(hd_dimension) if use_hd else None
        
        # Traditional detectors
        self.vintage_detector = VintageDetector()
        self.electronic_detector = ElectronicDetector()
        
        # Physics engine for optical analysis
        self.optics_engine = OpticsEngine()
        
        # Lens database
        self.known_lenses = {}
        self._load_lens_database()
        
    def analyze(self, 
                image: Union[str, Path, np.ndarray],
                reference_images: Optional[List[np.ndarray]] = None,
                full_analysis: bool = True) -> LensCharacteristics:
        """
        Perform comprehensive lens analysis.
        
        Args:
            image: Input image (path or array)
            reference_images: Optional reference images for comparison
            full_analysis: Whether to perform all analysis steps
            
        Returns:
            Complete lens characteristics profile
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Could not load image: {image}")
        
        logger.info("Starting lens characterization...")
        
        # Initialize characteristics
        characteristics = LensCharacteristics(
            focal_length=50.0,  # Default, will be updated if metadata available
            max_aperture=1.4,
            min_aperture=16.0
        )
        
        # Extract metadata if available
        self._extract_metadata(image, characteristics)
        
        # Detect lens type (vintage vs modern)
        lens_type = self._detect_lens_type(image)
        logger.info(f"Detected lens type: {lens_type}")
        
        if full_analysis:
            # Optical analysis
            self._analyze_optics(image, characteristics)
            
            # Defect detection
            self._detect_defects(image, characteristics)
            
            # Quality assessment
            self._assess_quality(image, characteristics)
            
            # HD analysis if enabled
            if self.use_hd:
                self._perform_hd_analysis(image, characteristics)
            
            # Match against known lenses
            if self.use_hd and characteristics.hyperdimensional_signature is not None:
                matched_lens = self._match_known_lens(characteristics)
                if matched_lens:
                    logger.info(f"Matched lens: {matched_lens}")
                    characteristics.lens_model = matched_lens
        
        # Calculate overall quality score
        characteristics.overall_quality = self._calculate_overall_quality(characteristics)
        
        return characteristics
    
    def _extract_metadata(self, image: np.ndarray, characteristics: LensCharacteristics):
        """Extract lens metadata from EXIF if available."""
        try:
            # This would use exiftool integration in production
            # For now, just set defaults
            pass
        except Exception as e:
            logger.debug(f"Could not extract metadata: {e}")
    
    def _detect_lens_type(self, image: np.ndarray) -> str:
        """Detect whether lens is vintage or modern."""
        vintage_score = self.vintage_detector.detect(image)
        electronic_score = self.electronic_detector.detect(image)
        
        if vintage_score > electronic_score:
            return "vintage"
        else:
            return "modern"
    
    def _analyze_optics(self, image: np.ndarray, characteristics: LensCharacteristics):
        """Analyze optical properties of the lens."""
        logger.info("Analyzing optical properties...")
        
        # Vignetting analysis
        characteristics.vignetting_profile = self._analyze_vignetting(image)
        
        # Distortion analysis
        characteristics.distortion_coefficients = self._analyze_distortion(image)
        
        # Chromatic aberration
        characteristics.chromatic_aberration = self._analyze_chromatic_aberration(image)
        
        # Bokeh quality
        characteristics.bokeh_quality = self._analyze_bokeh(image)
    
    def _analyze_vignetting(self, image: np.ndarray) -> np.ndarray:
        """Analyze vignetting pattern."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Create radial profile
        h, w = gray.shape
        center = (w // 2, h // 2)
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        
        # Normalize distances
        max_dist = np.sqrt(center[0]**2 + center[1]**2)
        dist_norm = dist / max_dist
        
        # Compute average intensity at different radii
        radii = np.linspace(0, 1, 20)
        profile = []
        
        for i in range(len(radii) - 1):
            mask = (dist_norm >= radii[i]) & (dist_norm < radii[i + 1])
            if np.any(mask):
                avg_intensity = np.mean(gray[mask])
                profile.append(avg_intensity)
            else:
                profile.append(0)
        
        profile = np.array(profile)
        # Normalize to center intensity
        if profile[0] > 0:
            profile = profile / profile[0]
        
        return profile
    
    def _analyze_distortion(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze lens distortion."""
        # Simplified distortion analysis
        # In production, would use checkerboard calibration
        h, w = image.shape[:2]
        
        # Detect straight lines
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150)
        
        # Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        # Analyze line curvature
        if lines is not None:
            # Simplified: check deviation from straight
            barrel_score = 0.0
            pincushion_score = 0.0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line bends outward (barrel) or inward (pincushion)
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                center_x, center_y = w / 2, h / 2
                
                # Distance from center
                dist = np.sqrt((mid_x - center_x)**2 + (mid_y - center_y)**2)
                if dist > min(w, h) / 4:  # Only consider lines away from center
                    # Simplified scoring
                    if dist > np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2):
                        barrel_score += 1
                    else:
                        pincushion_score += 1
            
            total_lines = max(len(lines), 1)
            return {
                'k1': (barrel_score - pincushion_score) / total_lines * 0.1,
                'k2': 0.0,  # Would need more sophisticated analysis
                'p1': 0.0,
                'p2': 0.0
            }
        
        return {'k1': 0.0, 'k2': 0.0, 'p1': 0.0, 'p2': 0.0}
    
    def _analyze_chromatic_aberration(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze chromatic aberration."""
        if len(image.shape) != 3:
            return {'lateral': 0.0, 'longitudinal': 0.0}
        
        # Split channels
        b, g, r = cv2.split(image)
        
        # Find edges in each channel
        edges_r = cv2.Canny(r, 50, 150)
        edges_g = cv2.Canny(g, 50, 150)
        edges_b = cv2.Canny(b, 50, 150)
        
        # Compare edge positions
        # Lateral CA: edges don't align
        lateral_ca = 0.0
        
        # Find contours in each channel
        contours_r, _ = cv2.findContours(edges_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_g, _ = cv2.findContours(edges_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours_r and contours_g:
            # Compare centroids of largest contours
            if len(contours_r) > 0 and len(contours_g) > 0:
                M_r = cv2.moments(max(contours_r, key=cv2.contourArea))
                M_g = cv2.moments(max(contours_g, key=cv2.contourArea))
                
                if M_r["m00"] != 0 and M_g["m00"] != 0:
                    cx_r = M_r["m10"] / M_r["m00"]
                    cy_r = M_r["m01"] / M_r["m00"]
                    cx_g = M_g["m10"] / M_g["m00"]
                    cy_g = M_g["m01"] / M_g["m00"]
                    
                    lateral_ca = np.sqrt((cx_r - cx_g)**2 + (cy_r - cy_g)**2)
        
        # Longitudinal CA: different channels focus at different distances
        # Simplified: check sharpness difference between channels
        lap_r = cv2.Laplacian(r, cv2.CV_64F).var()
        lap_g = cv2.Laplacian(g, cv2.CV_64F).var()
        lap_b = cv2.Laplacian(b, cv2.CV_64F).var()
        
        longitudinal_ca = np.std([lap_r, lap_g, lap_b]) / np.mean([lap_r, lap_g, lap_b])
        
        return {
            'lateral': float(lateral_ca / max(image.shape[:2]) * 100),  # Percentage
            'longitudinal': float(longitudinal_ca)
        }
    
    def _analyze_bokeh(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze bokeh quality characteristics."""
        # Simplified bokeh analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Find bright spots (potential bokeh balls)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze bokeh ball characteristics
        roundness_scores = []
        smoothness_scores = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum size for bokeh ball
                # Check roundness
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    roundness_scores.append(circularity)
                
                # Check edge smoothness
                # Simplified: use convexity defects
                hull = cv2.convexHull(contour, returnPoints=False)
                if len(hull) > 3 and len(contour) > 3:
                    defects = cv2.convexityDefects(contour, hull)
                    if defects is not None:
                        smoothness = 1.0 - (len(defects) / len(contour))
                        smoothness_scores.append(smoothness)
        
        return {
            'roundness': float(np.mean(roundness_scores)) if roundness_scores else 0.0,
            'smoothness': float(np.mean(smoothness_scores)) if smoothness_scores else 0.0,
            'creaminess': 0.7,  # Would need more sophisticated analysis
            'swirl': 0.0  # Would detect spiral patterns for lenses like Helios
        }
    
    def _detect_defects(self, image: np.ndarray, characteristics: LensCharacteristics):
        """Detect lens defects and artifacts."""
        logger.info("Detecting defects...")
        
        if self.use_hd:
            # Use HD analysis for defect detection
            defects = analyze_lens_defects(image)
            characteristics.dust_spots = defects['dust_spots']
            characteristics.scratches = defects['scratches']
            characteristics.fungus_areas = defects['regions']
            
            # Store topological features
            if 'details' in defects:
                characteristics.topological_features = (
                    defects['details']['dust_features'] +
                    defects['details']['scratch_features'] +
                    defects['details']['region_features']
                )
        else:
            # Traditional defect detection
            self._detect_dust(image, characteristics)
            self._detect_scratches(image, characteristics)
            self._detect_fungus(image, characteristics)
            self._detect_haze(image, characteristics)
    
    def _detect_dust(self, image: np.ndarray, characteristics: LensCharacteristics):
        """Detect dust spots."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Use blob detector
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 10
        params.maxArea = 1000
        params.filterByCircularity = True
        params.minCircularity = 0.7
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)
        
        characteristics.dust_spots = len(keypoints)
        
        # Create defect objects
        for kp in keypoints:
            defect = OpticalDefect(
                type="dust",
                severity=kp.size / max(image.shape[:2]),
                location=(kp.pt[0] / image.shape[1], kp.pt[1] / image.shape[0]),
                size=kp.size
            )
            characteristics.defects.append(defect)
    
    def _detect_scratches(self, image: np.ndarray, characteristics: LensCharacteristics):
        """Detect scratches."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Use line detection
        edges = cv2.Canny(gray, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=5)
        
        if lines is not None:
            # Filter for scratch-like lines
            scratch_count = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Check if line is thin and long (scratch-like)
                if length > 50:
                    # Sample perpendicular to line to check width
                    angle = np.arctan2(y2 - y1, x2 - x1)
                    perp_angle = angle + np.pi/2
                    
                    # Sample points
                    sample_x = int((x1 + x2) / 2)
                    sample_y = int((y1 + y2) / 2)
                    
                    width = 0
                    for d in range(-5, 6):
                        px = int(sample_x + d * np.cos(perp_angle))
                        py = int(sample_y + d * np.sin(perp_angle))
                        
                        if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
                            if edges[py, px] > 0:
                                width += 1
                    
                    if width < 3:  # Thin line
                        scratch_count += 1
                        
                        defect = OpticalDefect(
                            type="scratch",
                            severity=length / max(image.shape[:2]),
                            location=(sample_x / image.shape[1], sample_y / image.shape[0]),
                            size=length
                        )
                        characteristics.defects.append(defect)
            
            characteristics.scratches = scratch_count
    
    def _detect_fungus(self, image: np.ndarray, characteristics: LensCharacteristics):
        """Detect fungus patterns."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Fungus typically appears as branching patterns
        # Use morphological operations to detect
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Top-hat to find small bright structures
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        # Threshold
        _, thresh = cv2.threshold(tophat, 20, 255, cv2.THRESH_BINARY)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
        
        fungus_count = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Fungus tends to be larger branching structures
            if area > 100:
                # Check for branching pattern
                component_mask = (labels == i).astype(np.uint8) * 255
                skeleton = cv2.ximgproc.thinning(component_mask)
                
                # Count branch points (pixels with >2 neighbors)
                branch_points = 0
                for y in range(1, skeleton.shape[0] - 1):
                    for x in range(1, skeleton.shape[1] - 1):
                        if skeleton[y, x] > 0:
                            neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2]) - skeleton[y, x]
                            if neighbors > 2 * 255:  # More than 2 neighbors
                                branch_points += 1
                
                if branch_points > 3:  # Has branching structure
                    fungus_count += 1
                    
                    cx, cy = centroids[i]
                    defect = OpticalDefect(
                        type="fungus",
                        severity=area / (image.shape[0] * image.shape[1]),
                        location=(cx / image.shape[1], cy / image.shape[0]),
                        size=np.sqrt(area)
                    )
                    characteristics.defects.append(defect)
        
        characteristics.fungus_areas = fungus_count
    
    def _detect_haze(self, image: np.ndarray, characteristics: LensCharacteristics):
        """Detect haze level."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Haze reduces contrast
        # Measure local contrast
        kernel_size = 21
        local_mean = cv2.blur(gray, (kernel_size, kernel_size))
        local_var = cv2.blur(gray**2, (kernel_size, kernel_size)) - local_mean**2
        local_std = np.sqrt(np.maximum(local_var, 0))
        
        # Average local contrast
        avg_contrast = np.mean(local_std)
        
        # Normalize (lower contrast = more haze)
        # Assuming good contrast is around 50-60
        haze_level = max(0, 1 - (avg_contrast / 60))
        
        characteristics.haze_level = float(haze_level)
        
        if haze_level > 0.3:
            defect = OpticalDefect(
                type="haze",
                severity=haze_level,
                location=(0.5, 0.5),  # Affects whole image
                size=max(image.shape[:2])
            )
            characteristics.defects.append(defect)
    
    def _assess_quality(self, image: np.ndarray, characteristics: LensCharacteristics):
        """Assess overall image quality metrics."""
        logger.info("Assessing quality...")
        
        # Sharpness
        characteristics.sharpness_score = self._measure_sharpness(image)
        
        # Contrast
        characteristics.contrast_score = self._measure_contrast(image)
        
        # Color accuracy (simplified)
        characteristics.color_accuracy = self._measure_color_accuracy(image)
    
    def _measure_sharpness(self, image: np.ndarray) -> float:
        """Measure image sharpness."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Use Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Normalize to 0-1 scale
        # Assuming good sharpness is around 500-1000
        return float(min(1.0, sharpness / 1000))
    
    def _measure_contrast(self, image: np.ndarray) -> float:
        """Measure image contrast."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # RMS contrast
        mean = np.mean(gray)
        rms_contrast = np.sqrt(np.mean((gray - mean)**2)) / mean
        
        # Normalize to 0-1 scale
        return float(min(1.0, rms_contrast * 2))
    
    def _measure_color_accuracy(self, image: np.ndarray) -> float:
        """Measure color accuracy (simplified)."""
        if len(image.shape) != 3:
            return 1.0
        
        # Check color balance
        b, g, r = cv2.split(image)
        
        # Compare channel means
        mean_b, mean_g, mean_r = np.mean(b), np.mean(g), np.mean(r)
        
        # Perfect gray would have equal means
        # Measure deviation
        avg_mean = (mean_b + mean_g + mean_r) / 3
        deviation = (abs(mean_b - avg_mean) + abs(mean_g - avg_mean) + abs(mean_r - avg_mean)) / 3
        
        # Normalize
        color_accuracy = 1 - (deviation / avg_mean)
        
        return float(max(0, min(1, color_accuracy)))
    
    def _perform_hd_analysis(self, image: np.ndarray, characteristics: LensCharacteristics):
        """Perform hyperdimensional analysis."""
        logger.info("Performing HD analysis...")
        
        # Get comprehensive HD analysis
        hd_results = self.hd_analyzer.analyze_and_correct(
            image,
            mode='auto',
            strength=0.0  # Just analyze, don't correct
        )
        
        # Store HD signature
        characteristics.hyperdimensional_signature = hd_results['defect_hypervector']
        
        # Store topological features if not already done
        if characteristics.topological_features is None:
            topo = hd_results['topology']
            characteristics.topological_features = (
                topo['dust_features'] +
                topo['scratch_features'] +
                topo['region_features']
            )
    
    def _match_known_lens(self, characteristics: LensCharacteristics) -> Optional[str]:
        """Match characteristics against known lens database."""
        if not self.known_lenses or characteristics.hyperdimensional_signature is None:
            return None
        
        best_match = None
        best_similarity = 0.0
        threshold = 0.75
        
        for lens_name, lens_data in self.known_lenses.items():
            if 'hd_signature' in lens_data:
                similarity = self.hd_analyzer.encoder.similarity(
                    characteristics.hyperdimensional_signature,
                    lens_data['hd_signature']
                )
                
                if similarity > best_similarity and similarity > threshold:
                    best_similarity = similarity
                    best_match = lens_name
        
        return best_match
    
    def _calculate_overall_quality(self, characteristics: LensCharacteristics) -> float:
        """Calculate overall quality score."""
        # Weight different factors
        weights = {
            'sharpness': 0.3,
            'contrast': 0.2,
            'color': 0.1,
            'defects': 0.4
        }
        
        # Defect penalty
        defect_score = 1.0
        defect_score -= characteristics.dust_spots * 0.01
        defect_score -= characteristics.scratches * 0.02
        defect_score -= characteristics.fungus_areas * 0.05
        defect_score -= characteristics.haze_level * 0.3
        defect_score = max(0, defect_score)
        
        # Combined score
        overall = (
            weights['sharpness'] * characteristics.sharpness_score +
            weights['contrast'] * characteristics.contrast_score +
            weights['color'] * characteristics.color_accuracy +
            weights['defects'] * defect_score
        )
        
        return float(overall)
    
    def _load_lens_database(self):
        """Load known lens signatures from database."""
        # In production, would load from actual database
        # For now, create some example signatures
        self.known_lenses = {
            "Canon FD 50mm f/1.4": {
                "focal_length": 50,
                "max_aperture": 1.4,
                "characteristics": {
                    "sharpness": 0.85,
                    "bokeh_quality": 0.9,
                    "vignetting": "moderate"
                }
            },
            "Helios 44-2 58mm f/2": {
                "focal_length": 58,
                "max_aperture": 2.0,
                "characteristics": {
                    "sharpness": 0.75,
                    "bokeh_quality": 0.95,
                    "swirly_bokeh": True,
                    "vignetting": "strong"
                }
            },
            "Pentax SMC 50mm f/1.7": {
                "focal_length": 50,
                "max_aperture": 1.7,
                "characteristics": {
                    "sharpness": 0.88,
                    "bokeh_quality": 0.85,
                    "coating": "SMC",
                    "vignetting": "low"
                }
            }
        }
    
    def add_lens_to_database(self, 
                            lens_name: str,
                            sample_images: List[np.ndarray],
                            metadata: Optional[Dict] = None):
        """Add a new lens to the database with its signature."""
        if self.use_hd:
            # Create HD signature
            signature = self.hd_analyzer.create_lens_signature(lens_name, sample_images)
            
            # Store in database
            self.known_lenses[lens_name] = {
                'hd_signature': signature,
                'metadata': metadata or {}
            }
            
            logger.info(f"Added {lens_name} to lens database")
    
    def compare_lenses(self,
                      image1: np.ndarray,
                      image2: np.ndarray) -> Dict[str, float]:
        """Compare characteristics of two lens images."""
        # Analyze both images
        char1 = self.analyze(image1, full_analysis=True)
        char2 = self.analyze(image2, full_analysis=True)
        
        # Compare characteristics
        comparison = {
            'sharpness_diff': abs(char1.sharpness_score - char2.sharpness_score),
            'contrast_diff': abs(char1.contrast_score - char2.contrast_score),
            'color_diff': abs(char1.color_accuracy - char2.color_accuracy),
            'quality_diff': abs(char1.overall_quality - char2.overall_quality),
            'defect_diff': abs(
                (char1.dust_spots + char1.scratches + char1.fungus_areas) -
                (char2.dust_spots + char2.scratches + char2.fungus_areas)
            )
        }
        
        # HD similarity if available
        if self.use_hd and char1.hyperdimensional_signature is not None and char2.hyperdimensional_signature is not None:
            comparison['hd_similarity'] = self.hd_analyzer.encoder.similarity(
                char1.hyperdimensional_signature,
                char2.hyperdimensional_signature
            )
        
        return comparison
    
    def generate_report(self, characteristics: LensCharacteristics) -> str:
        """Generate a human-readable report of lens characteristics."""
        report = []
        report.append("=== Lens Characterization Report ===\n")
        
        # Basic info
        if characteristics.lens_model:
            report.append(f"Lens Model: {characteristics.lens_model}")
        report.append(f"Focal Length: {characteristics.focal_length}mm")
        report.append(f"Aperture Range: f/{characteristics.max_aperture} - f/{characteristics.min_aperture}")
        
        # Quality scores
        report.append("\nQuality Metrics:")
        report.append(f"  Overall Quality: {characteristics.overall_quality:.1%}")
        report.append(f"  Sharpness: {characteristics.sharpness_score:.1%}")
        report.append(f"  Contrast: {characteristics.contrast_score:.1%}")
        report.append(f"  Color Accuracy: {characteristics.color_accuracy:.1%}")
        
        # Optical characteristics
        report.append("\nOptical Characteristics:")
        
        if characteristics.vignetting_profile is not None:
            vignetting_strength = 1 - np.min(characteristics.vignetting_profile)
            report.append(f"  Vignetting: {vignetting_strength:.1%}")
        
        if characteristics.chromatic_aberration:
            report.append(f"  Chromatic Aberration:")
            report.append(f"    Lateral: {characteristics.chromatic_aberration['lateral']:.2f}%")
            report.append(f"    Longitudinal: {characteristics.chromatic_aberration['longitudinal']:.2f}")
        
        if characteristics.bokeh_quality:
            report.append(f"  Bokeh Quality:")
            report.append(f"    Roundness: {characteristics.bokeh_quality['roundness']:.1%}")
            report.append(f"    Smoothness: {characteristics.bokeh_quality['smoothness']:.1%}")
        
        # Defects
        report.append("\nDetected Defects:")
        report.append(f"  Dust Spots: {characteristics.dust_spots}")
        report.append(f"  Scratches: {characteristics.scratches}")
        report.append(f"  Fungus Areas: {characteristics.fungus_areas}")
        report.append(f"  Haze Level: {characteristics.haze_level:.1%}")
        
        # HD analysis
        if characteristics.hyperdimensional_signature is not None:
            report.append("\nHyperdimensional Analysis:")
            report.append(f"  HD Signature Computed: Yes")
            report.append(f"  Topological Features: {len(characteristics.topological_features or [])}")
        
        return "\n".join(report)


# Convenience function for quick analysis
def quick_lens_analysis(image_path: str) -> LensCharacteristics:
    """
    Perform quick lens analysis on an image.
    
    Args:
        image_path: Path to the image
        
    Returns:
        Lens characteristics
    """
    characterizer = LensCharacterizer(use_hd=True)
    return characterizer.analyze(image_path)
