"""
Simple Enhanced Pipeline using actually available components
"""

import numpy as np
import cv2
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class SimpleEnhancedPipeline:
    """Enhanced pipeline using components that actually exist"""
    
    def __init__(self):
        self.components_loaded = []
        
        # Try to load existing physics models
        try:
            from vintageoptics.physics.models.vignetting import VignettingModel
            self.vignetting_model = VignettingModel()
            self.components_loaded.append("vignetting")
        except:
            self.vignetting_model = None
            
        try:
            from vintageoptics.physics.models.chromatic import ChromaticAberration
            self.chromatic_model = ChromaticAberration()
            self.components_loaded.append("chromatic")
        except:
            self.chromatic_model = None
            
        try:
            from vintageoptics.physics.models.aberrations import AberrationModel
            self.aberration_model = AberrationModel()
            self.components_loaded.append("aberrations")
        except:
            self.aberration_model = None
        
        logger.info(f"SimpleEnhancedPipeline initialized with: {self.components_loaded}")
    
    def process(self, image: np.ndarray, lens_profile: Any, 
                correction_mode: str = "hybrid") -> tuple:
        """Process image with available components"""
        
        start_time = time.time()
        result = image.copy()
        
        # Track what processing was applied
        applied_effects = []
        
        # Apply vignetting
        if self.vignetting_model and hasattr(lens_profile, 'vignetting_amount'):
            try:
                result = self._apply_advanced_vignetting(
                    result, 
                    lens_profile.vignetting_amount,
                    lens_profile.vignetting_falloff
                )
                applied_effects.append("advanced_vignetting")
            except:
                result = self._apply_simple_vignetting(result, lens_profile.vignetting_amount)
                applied_effects.append("simple_vignetting")
        else:
            result = self._apply_simple_vignetting(result, 0.3)
            applied_effects.append("simple_vignetting")
        
        # Apply chromatic aberration
        if correction_mode in ["synthesis", "hybrid"]:
            if self.chromatic_model and hasattr(lens_profile, 'chromatic_aberration'):
                try:
                    result = self._apply_chromatic_aberration(
                        result,
                        lens_profile.chromatic_aberration
                    )
                    applied_effects.append("chromatic_aberration")
                except:
                    pass
        
        # Apply vintage color grading
        result = self._apply_vintage_color_grading(result, lens_profile)
        applied_effects.append("color_grading")
        
        # Apply lens distortion
        if hasattr(lens_profile, 'k1') and hasattr(lens_profile, 'k2'):
            result = self._apply_lens_distortion(
                result,
                lens_profile.k1,
                lens_profile.k2
            )
            applied_effects.append("lens_distortion")
        
        # Calculate actual quality metrics
        processing_time = time.time() - start_time
        quality_metrics = self._calculate_real_metrics(result, image)
        
        # Add processing info to metrics
        quality_metrics['processing_time'] = processing_time
        quality_metrics['effects_applied'] = len(applied_effects)
        quality_metrics['effects_list'] = applied_effects
        
        logger.info(f"Enhanced processing completed in {processing_time:.3f}s")
        logger.info(f"Applied effects: {', '.join(applied_effects)}")
        
        return result, quality_metrics
    
    def _apply_simple_vignetting(self, image: np.ndarray, amount: float) -> np.ndarray:
        """Apply radial vignetting effect"""
        h, w = image.shape[:2]
        
        # Create radial gradient
        Y, X = np.ogrid[:h, :w]
        center_x, center_y = w/2, h/2
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # Create vignette mask with smooth falloff
        vignette = 1 - (dist_from_center / max_dist) ** 2.5 * amount
        vignette = np.clip(vignette, 0, 1)
        
        # Apply to each channel
        result = image.astype(np.float32)
        for i in range(3):
            result[:, :, i] *= vignette
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _apply_advanced_vignetting(self, image: np.ndarray, amount: float, falloff: float) -> np.ndarray:
        """Apply advanced vignetting with custom falloff"""
        h, w = image.shape[:2]
        
        # More sophisticated vignetting
        Y, X = np.ogrid[:h, :w]
        center_x, center_y = w/2, h/2
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # Use cosine falloff for smoother transition
        normalized_dist = dist_from_center / max_dist
        vignette = np.cos(normalized_dist * np.pi / 2) ** falloff
        vignette = 1 - (1 - vignette) * amount
        vignette = np.clip(vignette, 0, 1)
        
        # Apply to each channel
        result = image.astype(np.float32)
        for i in range(3):
            result[:, :, i] *= vignette
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _apply_chromatic_aberration(self, image: np.ndarray, amount: float) -> np.ndarray:
        """Apply lateral chromatic aberration"""
        h, w = image.shape[:2]
        center_x, center_y = w/2, h/2
        
        # Split channels
        b, g, r = cv2.split(image)
        
        # Create displacement maps
        Y, X = np.mgrid[:h, :w]
        dist_x = (X - center_x) / center_x
        dist_y = (Y - center_y) / center_y
        
        # Scale red and blue channels differently
        scale_r = 1 + amount * 0.01
        scale_b = 1 - amount * 0.01
        
        # Create maps for remapping
        map_r_x = center_x + (X - center_x) * scale_r
        map_r_y = center_y + (Y - center_y) * scale_r
        map_b_x = center_x + (X - center_x) * scale_b
        map_b_y = center_y + (Y - center_y) * scale_b
        
        # Remap channels
        r_shifted = cv2.remap(r, map_r_x.astype(np.float32), map_r_y.astype(np.float32), 
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        b_shifted = cv2.remap(b, map_b_x.astype(np.float32), map_b_y.astype(np.float32), 
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        # Merge back
        return cv2.merge([b_shifted, g, r_shifted])
    
    def _apply_vintage_color_grading(self, image: np.ndarray, lens_profile: Any) -> np.ndarray:
        """Apply era-appropriate color grading"""
        result = image.astype(np.float32)
        
        # Determine era-based color shift
        era = getattr(lens_profile, 'era', 'modern')
        
        if '1970' in str(era) or '1960' in str(era):
            # Warm, golden tones for vintage era
            result[:, :, 2] *= 1.12  # Boost red
            result[:, :, 1] *= 1.08  # Slight green boost
            result[:, :, 0] *= 0.92  # Reduce blue
            
            # Lift shadows for film-like look
            shadows_mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) < 50
            result[shadows_mask] = result[shadows_mask] * 1.15 + 5
            
        elif '1980' in str(era):
            # Slightly cooler, more saturated
            result[:, :, 2] *= 1.05
            result[:, :, 1] *= 1.03
            result[:, :, 0] *= 0.98
        else:
            # Modern - subtle warming
            result[:, :, 2] *= 1.03
            result[:, :, 1] *= 1.01
            result[:, :, 0] *= 0.99
        
        # Add slight S-curve for contrast
        result = self._apply_tone_curve(result)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _apply_tone_curve(self, image: np.ndarray) -> np.ndarray:
        """Apply S-curve for film-like contrast"""
        # Normalize to 0-1
        normalized = image / 255.0
        
        # Apply S-curve
        # Lift shadows, slightly compress midtones, preserve highlights
        curve = np.power(normalized, 0.85) * 1.1
        
        return curve * 255
    
    def _apply_lens_distortion(self, image: np.ndarray, k1: float, k2: float) -> np.ndarray:
        """Apply barrel/pincushion distortion"""
        h, w = image.shape[:2]
        
        # Camera matrix (simplified)
        cx, cy = w/2, h/2
        fx = fy = w
        camera_matrix = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]], dtype=np.float32)
        
        # Distortion coefficients
        dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float32)
        
        # Generate new camera matrix
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
        
        # Apply distortion
        result = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
        
        return result
    
    def _calculate_real_metrics(self, processed: np.ndarray, original: np.ndarray) -> Dict[str, float]:
        """Calculate actual quality metrics"""
        
        # Convert to grayscale for analysis
        gray_processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        
        # Sharpness using Laplacian variance
        laplacian = cv2.Laplacian(gray_processed, cv2.CV_64F)
        sharpness = laplacian.var()
        sharpness_normalized = min(1.0, sharpness / 1000.0)
        
        # Contrast using standard deviation
        contrast = gray_processed.std() / 128.0
        contrast_normalized = min(1.0, contrast)
        
        # Color shift measurement
        mean_diff = np.mean(np.abs(processed.astype(float) - original.astype(float)), axis=(0, 1))
        color_shift = np.sum(mean_diff) / (255 * 3)
        
        # Edge preservation (structural similarity)
        edges_original = cv2.Canny(gray_original, 50, 150)
        edges_processed = cv2.Canny(gray_processed, 50, 150)
        edge_preservation = np.sum(edges_original & edges_processed) / max(np.sum(edges_original), 1)
        
        # Noise estimation
        noise = cv2.Laplacian(gray_processed, cv2.CV_64F).std()
        noise_normalized = 1.0 - min(1.0, noise / 50.0)
        
        # Overall quality (weighted combination)
        overall_quality = (
            sharpness_normalized * 0.25 +
            contrast_normalized * 0.25 +
            edge_preservation * 0.25 +
            noise_normalized * 0.15 +
            (1.0 - color_shift) * 0.10
        )
        
        # Correction strength based on actual changes
        correction_strength = color_shift + (1.0 - edge_preservation) * 0.5
        correction_strength = min(1.0, correction_strength * 2)
        
        return {
            'overall_quality': overall_quality,
            'sharpness': sharpness_normalized,
            'contrast': contrast_normalized,
            'edge_preservation': edge_preservation,
            'noise_level': 1.0 - noise_normalized,
            'color_shift': color_shift,
            'correction_applied': correction_strength
        }


# Singleton
_simple_pipeline = None

def get_simple_enhanced_pipeline():
    global _simple_pipeline
    if _simple_pipeline is None:
        _simple_pipeline = SimpleEnhancedPipeline()
    return _simple_pipeline
