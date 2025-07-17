# src/vintageoptics/statistical/adaptive_cleanup.py
"""
Enhanced statistical defect detection and cleanup with character preservation.
Implements comprehensive defect categorization and intelligent cleanup strategies.
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import ndimage, signal
from skimage import morphology, feature, restoration

logger = logging.getLogger(__name__)


class DefectType(Enum):
    """Categorization of vintage lens defects"""
    DUST_SPOT = "dust_spot"
    SCRATCH = "scratch"
    FUNGUS = "fungus"
    COATING_WEAR = "coating_wear"
    HAZE = "haze"
    SEPARATION = "separation"
    OIL_SPOT = "oil_spot"
    BUBBLE = "bubble"
    DEBRIS = "debris"


@dataclass
class DefectRegion:
    """Detected defect with metadata"""
    defect_type: DefectType
    mask: np.ndarray
    confidence: float
    size: int
    location: Tuple[int, int]  # Center point
    severity: float  # 0-1 scale
    preservable: bool  # Whether it adds character


class AdaptiveCleanup:
    """Advanced statistical cleanup with character preservation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.dust_sensitivity = config.get('cleanup', {}).get('dust_sensitivity', 0.8)
        self.scratch_sensitivity = config.get('cleanup', {}).get('scratch_sensitivity', 0.7)
        self.fungus_sensitivity = config.get('cleanup', {}).get('fungus_sensitivity', 0.75)
        self.preserve_character = config.get('cleanup', {}).get('preserve_character', True)
        self.preservation_threshold = config.get('cleanup', {}).get('preservation_threshold', 0.3)
        
        # Initialize defect detectors
        self._init_detectors()
    
    def _init_detectors(self):
        """Initialize specialized defect detectors"""
        # Morphological kernels for different defect types
        self.kernels = {
            'dust': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            'scratch': cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)),
            'fungus': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            'coating': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        }
    
    def clean_with_preservation(self, image: np.ndarray, 
                               lens_profile: Dict,
                               depth_map: Optional[np.ndarray] = None,
                               instance_defects: Optional[Dict] = None) -> np.ndarray:
        """Clean image while preserving lens character"""
        
        logger.info("Starting adaptive cleanup with character preservation")
        
        # Convert to working format
        working_image = image.astype(np.float32)
        if len(working_image.shape) == 2:
            working_image = cv2.cvtColor(working_image, cv2.COLOR_GRAY2RGB)
        
        # Detect all defects
        defects = self._comprehensive_defect_detection(working_image, lens_profile)
        
        # Apply instance-specific known defects if available
        if instance_defects:
            defects = self._merge_known_defects(defects, instance_defects)
        
        # Classify defects by character preservation
        defects_to_clean, defects_to_preserve = self._classify_defects(
            defects, lens_profile, depth_map
        )
        
        # Apply intelligent cleanup
        cleaned = self._apply_intelligent_cleanup(
            working_image, defects_to_clean, lens_profile
        )
        
        # Apply artistic preservation for character defects
        if self.preserve_character and defects_to_preserve:
            cleaned = self._apply_artistic_preservation(
                cleaned, defects_to_preserve, lens_profile
            )
        
        return np.clip(cleaned, 0, 255).astype(image.dtype)
    
    def _comprehensive_defect_detection(self, image: np.ndarray, 
                                      lens_profile: Dict) -> List[DefectRegion]:
        """Detect all types of defects comprehensively"""
        defects = []
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # 1. Dust spot detection
        dust_defects = self._detect_dust_spots_advanced(gray)
        defects.extend(dust_defects)
        
        # 2. Scratch detection
        scratch_defects = self._detect_scratches_advanced(gray)
        defects.extend(scratch_defects)
        
        # 3. Fungus detection
        if lens_profile.get('vintage', True):
            fungus_defects = self._detect_fungus_patterns(gray)
            defects.extend(fungus_defects)
        
        # 4. Coating wear detection
        coating_defects = self._detect_coating_wear(image)
        defects.extend(coating_defects)
        
        # 5. Haze detection
        haze_defects = self._detect_haze_regions(gray)
        defects.extend(haze_defects)
        
        # 6. Oil spot detection
        oil_defects = self._detect_oil_spots(image)
        defects.extend(oil_defects)
        
        logger.info(f"Detected {len(defects)} total defects")
        return defects
    
    def _detect_dust_spots_advanced(self, gray: np.ndarray) -> List[DefectRegion]:
        """Advanced dust spot detection using multiple techniques"""
        defects = []
        
        # 1. Morphological top-hat for bright spots
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, self.kernels['dust'])
        
        # 2. Morphological black-hat for dark spots
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.kernels['dust'])
        
        # 3. Adaptive thresholding
        bright_threshold = np.percentile(tophat, 99) * self.dust_sensitivity
        dark_threshold = np.percentile(blackhat, 99) * self.dust_sensitivity
        
        _, bright_mask = cv2.threshold(tophat, bright_threshold, 255, cv2.THRESH_BINARY)
        _, dark_mask = cv2.threshold(blackhat, dark_threshold, 255, cv2.THRESH_BINARY)
        
        # 4. Size and shape filtering
        for mask, defect_subtype in [(bright_mask, "bright"), (dark_mask, "dark")]:
            contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                         cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 5 <= area <= 500:  # Size constraints for dust
                    # Check circularity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.5:  # Reasonably circular
                            # Create defect region
                            mask_single = np.zeros_like(gray)
                            cv2.fillPoly(mask_single, [contour], 255)
                            
                            M = cv2.moments(contour)
                            cx = int(M['m10'] / (M['m00'] + 1e-6))
                            cy = int(M['m01'] / (M['m00'] + 1e-6))
                            
                            defects.append(DefectRegion(
                                defect_type=DefectType.DUST_SPOT,
                                mask=mask_single,
                                confidence=0.8 + 0.2 * circularity,
                                size=int(area),
                                location=(cx, cy),
                                severity=0.3,  # Dust is usually low severity
                                preservable=False
                            ))
        
        return defects
    
    def _detect_scratches_advanced(self, gray: np.ndarray) -> List[DefectRegion]:
        """Advanced scratch detection using line detection and morphology"""
        defects = []
        
        # 1. Edge detection with multiple scales
        edges_fine = cv2.Canny(gray, 30, 100)
        edges_coarse = cv2.Canny(gray, 50, 150)
        
        # 2. Line detection using Hough transform
        lines_fine = cv2.HoughLinesP(edges_fine, 1, np.pi/180, 
                                    threshold=40, minLineLength=20, maxLineGap=5)
        lines_coarse = cv2.HoughLinesP(edges_coarse, 1, np.pi/180,
                                      threshold=60, minLineLength=40, maxLineGap=10)
        
        # 3. Morphological operations to connect line segments
        scratch_kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        scratch_kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        scratch_kernel_d1 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        
        # Create scratch candidate mask
        scratch_mask = np.zeros_like(gray)
        
        for lines in [lines_fine, lines_coarse]:
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(scratch_mask, (x1, y1), (x2, y2), 255, 2)
        
        # 4. Filter by linearity and continuity
        if np.any(scratch_mask):
            # Apply morphological closing to connect segments
            scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, 
                                           scratch_kernel_h)
            scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, 
                                           scratch_kernel_v)
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                scratch_mask.astype(np.uint8), connectivity=8
            )
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > 50:  # Minimum scratch size
                    # Check aspect ratio for linearity
                    x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                                stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                    aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
                    
                    if aspect_ratio > 3:  # Linear structure
                        mask_single = (labels == i).astype(np.uint8) * 255
                        
                        defects.append(DefectRegion(
                            defect_type=DefectType.SCRATCH,
                            mask=mask_single,
                            confidence=min(0.6 + aspect_ratio / 20, 0.95),
                            size=area,
                            location=tuple(centroids[i].astype(int)),
                            severity=0.5,
                            preservable=False
                        ))
        
        return defects
    
    def _detect_fungus_patterns(self, gray: np.ndarray) -> List[DefectRegion]:
        """Detect fungus growth patterns typical in vintage lenses"""
        defects = []
        
        # Fungus typically appears as branching, web-like structures
        # 1. Enhance fine structures
        enhanced = cv2.equalizeHist(gray)
        
        # 2. Detect branching patterns using morphological operations
        # Fungus has organic, branching structure
        selem = morphology.disk(3)
        opened = morphology.opening(enhanced, selem)
        fungus_candidates = enhanced - opened
        
        # 3. Texture analysis for organic patterns
        # Use Local Binary Patterns for texture
        from skimage.feature import local_binary_pattern
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Fungus tends to have specific texture patterns
        # Look for non-uniform, branching textures
        texture_variance = ndimage.generic_filter(lbp, np.var, size=15)
        
        # 4. Threshold based on texture and morphology
        fungus_threshold = np.percentile(texture_variance, 95) * self.fungus_sensitivity
        _, texture_mask = cv2.threshold(texture_variance.astype(np.uint8), 
                                      fungus_threshold, 255, cv2.THRESH_BINARY)
        
        # 5. Combine with morphological candidates
        combined_mask = cv2.bitwise_and(texture_mask, 
                                       (fungus_candidates > 20).astype(np.uint8) * 255)
        
        # 6. Filter by shape characteristics
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum fungus patch size
                # Check for branching structure using convex hull
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / (hull_area + 1e-6)
                
                if solidity < 0.8:  # Branching structures have low solidity
                    mask_single = np.zeros_like(gray)
                    cv2.fillPoly(mask_single, [contour], 255)
                    
                    M = cv2.moments(contour)
                    cx = int(M['m10'] / (M['m00'] + 1e-6))
                    cy = int(M['m01'] / (M['m00'] + 1e-6))
                    
                    defects.append(DefectRegion(
                        defect_type=DefectType.FUNGUS,
                        mask=mask_single,
                        confidence=0.7 + 0.3 * (1 - solidity),
                        size=int(area),
                        location=(cx, cy),
                        severity=0.8,  # Fungus is serious
                        preservable=area < 500  # Small fungus might add character
                    ))
        
        return defects
    
    def _detect_coating_wear(self, image: np.ndarray) -> List[DefectRegion]:
        """Detect coating degradation and wear patterns"""
        defects = []
        
        # Coating wear often shows as color shifts or uneven patches
        # 1. Analyze color consistency
        lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # 2. Detect color irregularities
        # Calculate local color variance
        kernel_size = 21
        l_mean = cv2.blur(l_channel, (kernel_size, kernel_size))
        a_mean = cv2.blur(a_channel, (kernel_size, kernel_size))
        b_mean = cv2.blur(b_channel, (kernel_size, kernel_size))
        
        l_var = cv2.blur((l_channel - l_mean)**2, (kernel_size, kernel_size))
        a_var = cv2.blur((a_channel - a_mean)**2, (kernel_size, kernel_size))
        b_var = cv2.blur((b_channel - b_mean)**2, (kernel_size, kernel_size))
        
        # Combined variance indicates coating inconsistency
        total_var = np.sqrt(l_var + a_var + b_var)
        
        # 3. Threshold for coating wear
        wear_threshold = np.percentile(total_var, 90)
        _, wear_mask = cv2.threshold(total_var.astype(np.uint8), 
                                   wear_threshold, 255, cv2.THRESH_BINARY)
        
        # 4. Morphological processing to find coherent regions
        wear_mask = cv2.morphologyEx(wear_mask, cv2.MORPH_OPEN, self.kernels['coating'])
        wear_mask = cv2.morphologyEx(wear_mask, cv2.MORPH_CLOSE, self.kernels['coating'])
        
        # 5. Find coating wear regions
        contours, _ = cv2.findContours(wear_mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Significant wear area
                mask_single = np.zeros_like(wear_mask)
                cv2.fillPoly(mask_single, [contour], 255)
                
                M = cv2.moments(contour)
                cx = int(M['m10'] / (M['m00'] + 1e-6))
                cy = int(M['m01'] / (M['m00'] + 1e-6))
                
                defects.append(DefectRegion(
                    defect_type=DefectType.COATING_WEAR,
                    mask=mask_single,
                    confidence=0.75,
                    size=int(area),
                    location=(cx, cy),
                    severity=0.4,
                    preservable=True  # Coating wear can add vintage character
                ))
        
        return defects
    
    def _detect_haze_regions(self, gray: np.ndarray) -> List[DefectRegion]:
        """Detect hazy or foggy regions in the lens"""
        defects = []
        
        # Haze reduces local contrast
        # 1. Calculate local contrast map
        kernel_size = 31
        local_mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        local_var = cv2.blur((gray.astype(np.float32) - local_mean)**2, 
                           (kernel_size, kernel_size))
        local_std = np.sqrt(local_var)
        
        # 2. Normalize contrast map
        contrast_map = local_std / (local_mean + 1e-6)
        
        # 3. Find low contrast regions (haze)
        haze_threshold = np.percentile(contrast_map, 10)
        haze_mask = (contrast_map < haze_threshold).astype(np.uint8) * 255
        
        # 4. Remove small regions and edges
        haze_mask = cv2.morphologyEx(haze_mask, cv2.MORPH_OPEN, 
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
        
        # 5. Find haze regions
        contours, _ = cv2.findContours(haze_mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000:  # Significant haze area
                mask_single = np.zeros_like(gray)
                cv2.fillPoly(mask_single, [contour], 255)
                
                M = cv2.moments(contour)
                cx = int(M['m10'] / (M['m00'] + 1e-6))
                cy = int(M['m01'] / (M['m00'] + 1e-6))
                
                # Calculate average contrast reduction
                region_contrast = np.mean(contrast_map[mask_single > 0])
                severity = 1.0 - min(region_contrast / haze_threshold, 1.0)
                
                defects.append(DefectRegion(
                    defect_type=DefectType.HAZE,
                    mask=mask_single,
                    confidence=0.8,
                    size=int(area),
                    location=(cx, cy),
                    severity=severity,
                    preservable=severity < 0.5  # Mild haze can be atmospheric
                ))
        
        return defects
    
    def _detect_oil_spots(self, image: np.ndarray) -> List[DefectRegion]:
        """Detect oil spots on lens surface"""
        defects = []
        
        # Oil spots have characteristic rainbow patterns in color
        # 1. Analyze color channels for oil film interference patterns
        if len(image.shape) == 3:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            
            # Oil creates high saturation rainbow patterns
            # 2. Detect high saturation regions with color variance
            high_sat_mask = s > np.percentile(s, 85)
            
            # 3. Check for color variance in high saturation regions
            kernel_size = 11
            h_var = cv2.blur((h.astype(np.float32))**2, (kernel_size, kernel_size))
            
            # 4. Combined oil spot detection
            oil_candidates = np.logical_and(high_sat_mask, h_var > 1000).astype(np.uint8) * 255
            
            # 5. Morphological cleanup
            oil_mask = cv2.morphologyEx(oil_candidates, cv2.MORPH_OPEN,
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            
            # 6. Find oil spot regions
            contours, _ = cv2.findContours(oil_mask, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 5000:  # Oil spots size range
                    mask_single = np.zeros_like(oil_mask)
                    cv2.fillPoly(mask_single, [contour], 255)
                    
                    M = cv2.moments(contour)
                    cx = int(M['m10'] / (M['m00'] + 1e-6))
                    cy = int(M['m01'] / (M['m00'] + 1e-6))
                    
                    defects.append(DefectRegion(
                        defect_type=DefectType.OIL_SPOT,
                        mask=mask_single,
                        confidence=0.7,
                        size=int(area),
                        location=(cx, cy),
                        severity=0.3,
                        preservable=False  # Oil should be cleaned
                    ))
        
        return defects
    
    def _merge_known_defects(self, detected: List[DefectRegion], 
                           known: Dict) -> List[DefectRegion]:
        """Merge detected defects with known instance defects"""
        # Add known defects that weren't detected
        if 'permanent_defects' in known:
            for defect_info in known['permanent_defects']:
                # Convert known defect format to DefectRegion
                defect = DefectRegion(
                    defect_type=DefectType[defect_info['type'].upper()],
                    mask=defect_info.get('mask', np.zeros((10, 10))),
                    confidence=1.0,  # Known defects are certain
                    size=defect_info.get('size', 100),
                    location=tuple(defect_info.get('location', (0, 0))),
                    severity=defect_info.get('severity', 0.5),
                    preservable=defect_info.get('character', False)
                )
                detected.append(defect)
        
        return detected
    
    def _classify_defects(self, defects: List[DefectRegion], 
                         lens_profile: Dict,
                         depth_map: Optional[np.ndarray]) -> Tuple[List[DefectRegion], List[DefectRegion]]:
        """Classify defects into those to clean and those to preserve"""
        to_clean = []
        to_preserve = []
        
        # Get lens character preferences
        preserve_patina = lens_profile.get('preserve_patina', True)
        artistic_defects = lens_profile.get('artistic_defects', False)
        
        for defect in defects:
            # Decision factors
            preserve = False
            
            # 1. Check if defect type can add character
            if defect.preservable and self.preserve_character:
                # 2. Check severity
                if defect.severity < self.preservation_threshold:
                    preserve = True
                
                # 3. Check lens-specific preferences
                if defect.defect_type == DefectType.COATING_WEAR and preserve_patina:
                    preserve = True
                elif defect.defect_type == DefectType.FUNGUS and artistic_defects:
                    if defect.size < 200:  # Small fungus for character
                        preserve = True
                
                # 4. Check depth context if available
                if depth_map is not None and preserve:
                    # Preserve defects in out-of-focus areas more readily
                    cy, cx = defect.location
                    if 0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]:
                        depth_value = depth_map[cy, cx]
                        if depth_value > 0.7:  # Far/blurred area
                            preserve = True
            
            if preserve:
                to_preserve.append(defect)
            else:
                to_clean.append(defect)
        
        logger.info(f"Classified defects: {len(to_clean)} to clean, {len(to_preserve)} to preserve")
        return to_clean, to_preserve
    
    def _apply_intelligent_cleanup(self, image: np.ndarray, 
                                 defects: List[DefectRegion],
                                 lens_profile: Dict) -> np.ndarray:
        """Apply defect-specific intelligent cleanup"""
        cleaned = image.copy()
        
        # Group defects by type for efficient processing
        defects_by_type = {}
        for defect in defects:
            if defect.defect_type not in defects_by_type:
                defects_by_type[defect.defect_type] = []
            defects_by_type[defect.defect_type].append(defect)
        
        # Apply type-specific cleanup
        for defect_type, defect_list in defects_by_type.items():
            if defect_type in [DefectType.DUST_SPOT, DefectType.OIL_SPOT]:
                cleaned = self._clean_spot_defects(cleaned, defect_list)
            elif defect_type == DefectType.SCRATCH:
                cleaned = self._clean_scratch_defects(cleaned, defect_list)
            elif defect_type == DefectType.FUNGUS:
                cleaned = self._clean_fungus_defects(cleaned, defect_list)
            elif defect_type == DefectType.HAZE:
                cleaned = self._clean_haze_defects(cleaned, defect_list)
            elif defect_type == DefectType.COATING_WEAR:
                cleaned = self._clean_coating_defects(cleaned, defect_list)
        
        return cleaned
    
    def _clean_spot_defects(self, image: np.ndarray, 
                           defects: List[DefectRegion]) -> np.ndarray:
        """Clean spot-like defects (dust, oil)"""
        cleaned = image.copy()
        
        # Combine masks for batch processing
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for defect in defects:
            combined_mask = cv2.bitwise_or(combined_mask, defect.mask)
        
        # Use advanced inpainting
        if np.any(combined_mask):
            # Try exemplar-based inpainting first
            try:
                from skimage.restoration import inpaint_biharmonic
                for c in range(image.shape[2]):
                    cleaned[:, :, c] = inpaint_biharmonic(
                        cleaned[:, :, c], combined_mask > 0
                    )
            except:
                # Fallback to OpenCV inpainting
                cleaned = cv2.inpaint(cleaned.astype(np.uint8), combined_mask,
                                    inpaintRadius=5, flags=cv2.INPAINT_TELEA)
                cleaned = cleaned.astype(np.float32)
        
        return cleaned
    
    def _clean_scratch_defects(self, image: np.ndarray,
                              defects: List[DefectRegion]) -> np.ndarray:
        """Clean linear scratch defects"""
        cleaned = image.copy()
        
        for defect in defects:
            # For scratches, use directional inpainting
            # Dilate scratch mask slightly for better coverage
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated_mask = cv2.dilate(defect.mask, kernel, iterations=1)
            
            # Use Navier-Stokes inpainting for better edge preservation
            cleaned = cv2.inpaint(cleaned.astype(np.uint8), dilated_mask,
                                inpaintRadius=7, flags=cv2.INPAINT_NS)
            cleaned = cleaned.astype(np.float32)
        
        return cleaned
    
    def _clean_fungus_defects(self, image: np.ndarray,
                            defects: List[DefectRegion]) -> np.ndarray:
        """Clean fungus growth patterns"""
        cleaned = image.copy()
        
        for defect in defects:
            if defect.severity > 0.6:  # Only clean severe fungus
                # Fungus requires careful cleaning to avoid artifacts
                # Use texture synthesis approach
                try:
                    # Dilate mask to ensure complete coverage
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    dilated_mask = cv2.dilate(defect.mask, kernel, iterations=2)
                    
                    # Use content-aware fill
                    cleaned = cv2.inpaint(cleaned.astype(np.uint8), dilated_mask,
                                        inpaintRadius=11, flags=cv2.INPAINT_TELEA)
                    cleaned = cleaned.astype(np.float32)
                except Exception as e:
                    logger.warning(f"Fungus cleaning failed: {e}")
        
        return cleaned
    
    def _clean_haze_defects(self, image: np.ndarray,
                          defects: List[DefectRegion]) -> np.ndarray:
        """Clean hazy regions using dehazing algorithms"""
        cleaned = image.copy()
        
        for defect in defects:
            # Apply local contrast enhancement
            mask_3d = np.stack([defect.mask / 255.0] * 3, axis=2)
            
            # Extract hazy region
            hazy_region = cleaned * mask_3d
            
            # Apply CLAHE to enhance contrast
            lab = cv2.cvtColor(hazy_region.astype(np.uint8), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB).astype(np.float32)
            
            # Blend enhanced region back
            cleaned = cleaned * (1 - mask_3d) + enhanced_rgb * mask_3d
        
        return cleaned
    
    def _clean_coating_defects(self, image: np.ndarray,
                             defects: List[DefectRegion]) -> np.ndarray:
        """Clean coating wear while preserving character"""
        cleaned = image.copy()
        
        for defect in defects:
            if defect.severity > 0.6:  # Only clean severe coating issues
                # Color correction for coating wear
                mask_3d = np.stack([defect.mask / 255.0] * 3, axis=2)
                
                # Calculate local color statistics
                region = cleaned * mask_3d
                surrounding_mask = cv2.dilate(defect.mask, 
                                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
                surrounding_mask = surrounding_mask - defect.mask
                surrounding_region = cleaned * np.stack([surrounding_mask / 255.0] * 3, axis=2)
                
                # Match color statistics
                for c in range(3):
                    region_mean = np.mean(region[:, :, c][defect.mask > 0])
                    surround_mean = np.mean(surrounding_region[:, :, c][surrounding_mask > 0])
                    
                    if region_mean > 0:
                        correction_factor = surround_mean / region_mean
                        cleaned[:, :, c] = cleaned[:, :, c] * (1 - mask_3d[:, :, c]) + \
                                         cleaned[:, :, c] * mask_3d[:, :, c] * correction_factor
        
        return cleaned
    
    def _apply_artistic_preservation(self, image: np.ndarray,
                                   defects: List[DefectRegion],
                                   lens_profile: Dict) -> np.ndarray:
        """Artistically preserve character defects"""
        result = image.copy()
        
        # Apply subtle enhancement to character defects
        for defect in defects:
            if defect.defect_type == DefectType.COATING_WEAR:
                # Enhance color shift slightly for character
                mask_3d = np.stack([defect.mask / 255.0] * 3, axis=2) * 0.3
                
                # Add slight warm tone
                warm_adjustment = np.array([1.05, 1.02, 0.98]).reshape(1, 1, 3)
                result = result * (1 - mask_3d) + result * mask_3d * warm_adjustment
                
            elif defect.defect_type == DefectType.FUNGUS and defect.size < 200:
                # Keep small fungus but reduce opacity
                mask_3d = np.stack([defect.mask / 255.0] * 3, axis=2) * 0.5
                
                # Slight blur to make less distracting
                blurred = cv2.GaussianBlur(result, (5, 5), 1.0)
                result = result * (1 - mask_3d) + blurred * mask_3d
        
        return result
    
    def analyze_defects(self, image: np.ndarray, 
                       lens_profile: Dict) -> Dict[str, any]:
        """Analyze and report on detected defects"""
        defects = self._comprehensive_defect_detection(image, lens_profile)
        
        # Categorize and summarize
        summary = {
            'total_defects': len(defects),
            'defects_by_type': {},
            'total_affected_area': 0,
            'severity_score': 0.0,
            'preservation_candidates': 0
        }
        
        for defect in defects:
            defect_type_name = defect.defect_type.value
            if defect_type_name not in summary['defects_by_type']:
                summary['defects_by_type'][defect_type_name] = {
                    'count': 0,
                    'total_area': 0,
                    'avg_severity': 0.0
                }
            
            summary['defects_by_type'][defect_type_name]['count'] += 1
            summary['defects_by_type'][defect_type_name]['total_area'] += defect.size
            summary['defects_by_type'][defect_type_name]['avg_severity'] += defect.severity
            
            summary['total_affected_area'] += defect.size
            summary['severity_score'] += defect.severity * defect.size
            
            if defect.preservable:
                summary['preservation_candidates'] += 1
        
        # Calculate averages
        if len(defects) > 0:
            summary['severity_score'] /= summary['total_affected_area']
            
            for defect_type in summary['defects_by_type'].values():
                if defect_type['count'] > 0:
                    defect_type['avg_severity'] /= defect_type['count']
        
        # Calculate image quality impact
        image_area = image.shape[0] * image.shape[1]
        summary['defect_coverage_percent'] = (summary['total_affected_area'] / image_area) * 100
        
        return summary


# Also update the __init__.py to export the new class
__all__ = ['AdaptiveCleanup', 'DefectType', 'DefectRegion']
