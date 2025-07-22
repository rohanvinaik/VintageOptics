# src/vintageoptics/statistical/__init__.py

"""
Statistical defect detection and cleanup
"""

import numpy as np
import cv2
from typing import Dict, Optional

class AdaptiveCleanup:
    """Adaptive statistical cleanup with character preservation"""
    
    def __init__(self, config):
        self.config = config
        self.dust_sensitivity = config.get('cleanup', {}).get('dust_sensitivity', 0.8)
        self.scratch_sensitivity = config.get('cleanup', {}).get('scratch_sensitivity', 0.7)
    
    def clean_with_preservation(self, image: np.ndarray, lens_profile: Dict, 
                               depth_map: Optional[np.ndarray]) -> np.ndarray:
        """Clean image while preserving lens character"""
        
        # Convert to working format
        working_image = image.astype(np.float32)
        
        # Detect defects
        defect_mask = self._detect_defects(working_image)
        
        # Apply character-preserving cleanup
        cleaned = self._apply_cleanup(working_image, defect_mask, lens_profile)
        
        return np.clip(cleaned, 0, 255).astype(image.dtype)
    
    def _detect_defects(self, image: np.ndarray) -> np.ndarray:
        """Detect various types of defects"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)
        
        # Detect dust spots
        dust_mask = self._detect_dust_spots(gray)
        
        # Detect scratches
        scratch_mask = self._detect_scratches(gray)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(dust_mask, scratch_mask)
        
        return combined_mask
    
    def _detect_dust_spots(self, image: np.ndarray) -> np.ndarray:
        """Detect dust spots using morphological operations"""
        # Use top-hat transform to find bright spots
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        
        # Threshold to find significant spots
        threshold = np.percentile(tophat, 95) * self.dust_sensitivity
        _, dust_mask = cv2.threshold(tophat, threshold, 255, cv2.THRESH_BINARY)
        
        # Remove very large regions (probably not dust)
        contours, _ = cv2.findContours(dust_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(dust_mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 <= area <= 200:  # Dust spots are typically small
                cv2.fillPoly(filtered_mask, [contour], 255)
        
        return filtered_mask
    
    def _detect_scratches(self, image: np.ndarray) -> np.ndarray:
        """Detect linear scratches"""
        # Edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=30, maxLineGap=10)
        
        # Create scratch mask
        scratch_mask = np.zeros_like(image)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Draw line with thickness to create mask
                cv2.line(scratch_mask, (x1, y1), (x2, y2), 255, 3)
        
        return scratch_mask
    
    def _apply_cleanup(self, image: np.ndarray, defect_mask: np.ndarray, 
                      lens_profile: Dict) -> np.ndarray:
        """Apply cleanup while preserving lens character"""
        
        if np.sum(defect_mask) == 0:
            return image  # No defects to clean
        
        # Check if lens has special bokeh characteristics to preserve
        preserve_bokeh = lens_profile.get('swirly_bokeh', False)
        
        # Apply different inpainting based on lens type
        if preserve_bokeh:
            # Use more conservative inpainting for special lenses
            cleaned = cv2.inpaint(
                image.astype(np.uint8), defect_mask, 
                inpaintRadius=2, flags=cv2.INPAINT_TELEA
            )
        else:
            # Use more aggressive inpainting for standard lenses
            cleaned = cv2.inpaint(
                image.astype(np.uint8), defect_mask,
                inpaintRadius=3, flags=cv2.INPAINT_NS
            )
        
        return cleaned.astype(np.float32)

__all__ = ['AdaptiveCleanup']
