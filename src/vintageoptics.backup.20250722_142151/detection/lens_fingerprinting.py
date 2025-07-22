# src/vintageoptics/detection/lens_fingerprinting.py

import numpy as np
from typing import Dict, List, Optional, Tuple
import cv2
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OpticalFingerprint:
    """Optical fingerprint of a lens"""
    lens_id: str
    distortion_signature: np.ndarray
    chromatic_signature: np.ndarray
    vignetting_signature: np.ndarray
    bokeh_signature: Dict[str, float]
    defect_pattern: np.ndarray
    color_response: np.ndarray
    contrast_curve: np.ndarray
    flare_pattern: Optional[np.ndarray]
    confidence: float


class LensFingerprinter:
    """Creates optical fingerprints for lens identification"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
    def extract_signature(self, image: np.ndarray) -> Dict[str, float]:
        """Extract optical signature from image"""
        signature = {}
        
        # Check for characteristic patterns
        signature['bokeh_swirl'] = self._detect_swirly_bokeh(image)
        signature['yellow_cast'] = self._detect_yellowing(image)
        signature['vignetting'] = self._measure_vignetting(image)
        signature['sharpness'] = self._measure_sharpness(image)
        
        return signature
    
    def generate_fingerprint(self, image: np.ndarray) -> Dict[str, float]:
        """Generate fingerprint from single image"""
        return self.extract_signature(image)
    
    def _detect_swirly_bokeh(self, image: np.ndarray) -> float:
        """Detect characteristic swirly bokeh pattern"""
        # Simplified detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Look for circular patterns in out-of-focus areas
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        swirl_score = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                # Check for elliptical shape (characteristic of swirl)
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    axes = ellipse[1]
                    if min(axes) > 0:
                        eccentricity = 1 - (min(axes) / max(axes))
                        swirl_score += eccentricity
        
        return min(1.0, swirl_score / 10)
    
    def _detect_yellowing(self, image: np.ndarray) -> float:
        """Detect yellowing from radioactive glass"""
        if len(image.shape) != 3:
            return 0.0
        
        # Check color balance
        b_mean = np.mean(image[:, :, 0])
        g_mean = np.mean(image[:, :, 1])
        r_mean = np.mean(image[:, :, 2])
        
        # Yellow cast shows as high red/green, low blue
        yellow_ratio = (r_mean + g_mean) / (2 * b_mean + 1)
        
        return min(1.0, max(0.0, (yellow_ratio - 1.0) / 2))
    
    def _measure_vignetting(self, image: np.ndarray) -> float:
        """Measure vignetting strength"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        h, w = gray.shape
        center = gray[h//2-50:h//2+50, w//2-50:w//2+50]
        corners = [
            gray[:100, :100],
            gray[:100, -100:],
            gray[-100:, :100],
            gray[-100:, -100:]
        ]
        
        center_mean = np.mean(center)
        corner_mean = np.mean([np.mean(c) for c in corners])
        
        if center_mean > 0:
            vignetting = 1 - (corner_mean / center_mean)
            return max(0.0, min(1.0, vignetting))
        
        return 0.0
    
    def _measure_sharpness(self, image: np.ndarray) -> float:
        """Measure image sharpness"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Use Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Normalize
        return min(1.0, sharpness / 1000)


# Additional supporting class
class LensFingerprinting:
    """Legacy class name for compatibility"""
    def __init__(self):
        self.fingerprinter = LensFingerprinter()
    
    def generate_fingerprint(self, image: np.ndarray) -> Dict[str, float]:
        return self.fingerprinter.generate_fingerprint(image)
