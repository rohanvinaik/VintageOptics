"""
Enhanced lens synthesis with hyperdimensional computing integration.

This module provides advanced lens characteristic synthesis combining
physical optics simulation with HD-based pattern matching.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import cv2
from scipy import signal, ndimage
import logging

from ..hyperdimensional import HyperdimensionalEncoder, HyperdimensionalLensAnalyzer
from ..physics.optics_engine import OpticsEngine
from ..types.optics import LensProfile, BokehShape
from .bokeh_synthesis import BokehSynthesizer
from .characteristic_library import CharacteristicLibrary

logger = logging.getLogger(__name__)


@dataclass
class SynthesisConfig:
    """Configuration for lens synthesis."""
    # Synthesis mode
    mode: str = "physical"  # "physical", "learned", "hybrid"
    
    # Effect strengths
    vignetting_strength: float = 1.0
    distortion_strength: float = 1.0
    chromatic_strength: float = 1.0
    bokeh_strength: float = 1.0
    flare_strength: float = 1.0
    
    # Quality settings
    preserve_sharpness: bool = True
    adaptive_strength: bool = True
    
    # HD settings
    use_hd_matching: bool = True
    hd_blend_factor: float = 0.3


class LensSynthesizer:
    """
    Advanced lens characteristic synthesizer combining physical simulation
    with learned patterns from hyperdimensional analysis.
    """
    
    def __init__(self, config: Optional[SynthesisConfig] = None):
        """
        Initialize the synthesizer.
        
        Args:
            config: Synthesis configuration
        """
        self.config = config or SynthesisConfig()
        
        # Components
        self.optics_engine = OpticsEngine()
        self.bokeh_synthesizer = BokehSynthesizer()
        self.char_library = CharacteristicLibrary()
        
        # HD components
        if self.config.use_hd_matching:
            self.hd_analyzer = HyperdimensionalLensAnalyzer()
            self.hd_encoder = HyperdimensionalEncoder()
        else:
            self.hd_analyzer = None
            self.hd_encoder = None
        
        # Precomputed kernels for efficiency
        self._init_kernels()
        
    def _init_kernels(self):
        """Initialize commonly used kernels."""
        # Gaussian kernels for various effects
        self.gaussian_kernels = {}
        for size in [3, 5, 7, 9, 11, 15, 21]:
            self.gaussian_kernels[size] = cv2.getGaussianKernel(size, size/3)
            
        # Directional kernels for motion blur
        self.motion_kernels = {}
        for angle in [0, 45, 90, 135]:
            kernel = self._create_motion_kernel(15, angle)
            self.motion_kernels[angle] = kernel
    
    def apply(self,
             image: np.ndarray,
             lens_profile: LensProfile,
             depth_map: Optional[np.ndarray] = None,
             strength: float = 1.0) -> np.ndarray:
        """
        Apply lens characteristics to an image.
        
        Args:
            image: Input image
            lens_profile: Lens profile to apply
            depth_map: Optional depth map for depth-aware effects
            strength: Overall effect strength (0-1)
            
        Returns:
            Image with synthesized lens characteristics
        """
        logger.info(f"Applying lens synthesis: {lens_profile.name}")
        
        result = image.copy().astype(np.float32) / 255.0
        
        # Adjust strengths based on config
        v_strength = strength * self.config.vignetting_strength * lens_profile.vignetting_amount
        d_strength = strength * self.config.distortion_strength * lens_profile.distortion_amount
        c_strength = strength * self.config.chromatic_strength * lens_profile.chromatic_aberration
        b_strength = strength * self.config.bokeh_strength
        f_strength = strength * self.config.flare_strength * lens_profile.flare_intensity
        
        # Apply effects in physically correct order
        
        # 1. Distortion (happens in lens)
        if d_strength > 0:
            result = self._apply_distortion(result, lens_profile, d_strength)
        
        # 2. Chromatic aberration
        if c_strength > 0:
            result = self._apply_chromatic_aberration(result, lens_profile, c_strength)
        
        # 3. Vignetting
        if v_strength > 0:
            result = self._apply_vignetting(result, lens_profile, v_strength)
        
        # 4. Bokeh (if depth map available)
        if b_strength > 0 and depth_map is not None:
            result = self._apply_bokeh(result, depth_map, lens_profile, b_strength)
        
        # 5. Flare and ghosts
        if f_strength > 0:
            result = self._apply_flare(result, lens_profile, f_strength)
        
        # 6. Coating characteristics
        result = self._apply_coating_effects(result, lens_profile, strength)
        
        # 7. HD-based refinement
        if self.config.use_hd_matching and self.hd_analyzer:
            result = self._apply_hd_refinement(result, lens_profile, strength)
        
        # Convert back to uint8
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        return result
    
    def _apply_distortion(self, 
                         image: np.ndarray,
                         profile: LensProfile,
                         strength: float) -> np.ndarray:
        """Apply lens distortion."""
        h, w = image.shape[:2]
        
        # Get distortion parameters
        k1 = profile.distortion_k1 * strength
        k2 = profile.distortion_k2 * strength
        p1 = profile.distortion_p1 * strength
        p2 = profile.distortion_p2 * strength
        
        # Camera matrix (simplified)
        cx, cy = w / 2, h / 2
        fx = fy = max(w, h)
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Distortion coefficients
        dist_coeffs = np.array([k1, k2, p1, p2, 0], dtype=np.float32)
        
        # Apply distortion
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix, 
            -dist_coeffs,  # Negative to apply distortion
            None,
            camera_matrix,
            (w, h),
            cv2.CV_32FC1
        )
        
        result = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
        
        return result
    
    def _apply_chromatic_aberration(self,
                                   image: np.ndarray,
                                   profile: LensProfile,
                                   strength: float) -> np.ndarray:
        """Apply chromatic aberration."""
        if len(image.shape) != 3:
            return image
        
        h, w = image.shape[:2]
        cx, cy = w / 2, h / 2
        
        # Split channels
        b, g, r = cv2.split(image)
        
        # Different magnification for each channel
        ca_amount = profile.chromatic_aberration * strength / 100
        
        # Red channel - slight magnification
        scale_r = 1 + ca_amount
        M_r = np.array([[scale_r, 0, cx * (1 - scale_r)],
                       [0, scale_r, cy * (1 - scale_r)]], dtype=np.float32)
        r_scaled = cv2.warpAffine(r, M_r, (w, h))
        
        # Blue channel - slight reduction
        scale_b = 1 - ca_amount
        M_b = np.array([[scale_b, 0, cx * (1 - scale_b)],
                       [0, scale_b, cy * (1 - scale_b)]], dtype=np.float32)
        b_scaled = cv2.warpAffine(b, M_b, (w, h))
        
        # Merge back
        result = cv2.merge([b_scaled, g, r_scaled])
        
        return result
    
    def _apply_vignetting(self,
                         image: np.ndarray,
                         profile: LensProfile,
                         strength: float) -> np.ndarray:
        """Apply vignetting effect."""
        h, w = image.shape[:2]
        
        # Create vignetting mask
        Y, X = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        
        # Distance from center normalized
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        dist_norm = dist / max_dist
        
        # Vignetting function
        falloff = profile.vignetting_falloff
        vignette = 1 - (dist_norm**falloff) * strength
        
        # Apply to all channels
        if len(image.shape) == 3:
            vignette = vignette[:, :, np.newaxis]
        
        result = image * vignette
        
        return result
    
    def _apply_bokeh(self,
                    image: np.ndarray,
                    depth_map: np.ndarray,
                    profile: LensProfile,
                    strength: float) -> np.ndarray:
        """Apply depth-aware bokeh effect."""
        return self.bokeh_synthesizer.synthesize(
            image,
            depth_map,
            bokeh_shape=profile.bokeh_shape,
            quality=profile.bokeh_quality * strength,
            aperture_blades=profile.aperture_blades
        )
    
    def _apply_flare(self,
                    image: np.ndarray,
                    profile: LensProfile,
                    strength: float) -> np.ndarray:
        """Apply lens flare and ghost effects."""
        # Find bright sources
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Threshold for bright sources
        _, bright = cv2.threshold(gray, 0.9, 1.0, cv2.THRESH_BINARY)
        
        if np.any(bright):
            # Get bright source locations
            bright_y, bright_x = np.where(bright > 0)
            
            if len(bright_y) > 0:
                # Use centroid of bright region
                cx = np.mean(bright_x)
                cy = np.mean(bright_y)
                
                # Create flare
                flare = self._create_flare(
                    image.shape[:2],
                    (cx, cy),
                    profile,
                    strength
                )
                
                # Blend with image
                if len(image.shape) == 3:
                    flare = flare[:, :, np.newaxis]
                
                result = image + flare * strength
                result = np.clip(result, 0, 1)
                
                return result
        
        return image
    
    def _create_flare(self,
                     shape: Tuple[int, int],
                     source: Tuple[float, float],
                     profile: LensProfile,
                     strength: float) -> np.ndarray:
        """Create lens flare pattern."""
        h, w = shape
        flare = np.zeros((h, w), dtype=np.float32)
        
        # Main flare
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - source[0])**2 + (Y - source[1])**2)
        
        # Airy disk pattern
        wavelength = 550e-9  # Green light
        aperture = profile.max_aperture
        
        # Simplified Airy pattern
        theta = dist / max(h, w) * 0.1
        airy = (2 * np.pi * aperture * theta / wavelength) + 1e-8
        pattern = (2 * np.sin(airy) / airy)**2
        
        flare += pattern * strength * 0.3
        
        # Ghost reflections
        cx, cy = w / 2, h / 2
        
        # Mirror position through center
        ghost_x = 2 * cx - source[0]
        ghost_y = 2 * cy - source[1]
        
        # Create ghosts at various positions
        for i in range(profile.element_count // 2):
            alpha = (i + 1) / (profile.element_count // 2)
            gx = source[0] + alpha * (ghost_x - source[0])
            gy = source[1] + alpha * (ghost_y - source[1])
            
            # Ghost size and intensity
            size = 20 + i * 10
            intensity = 0.1 * (1 - alpha) * strength
            
            # Add ghost
            ghost_dist = np.sqrt((X - gx)**2 + (Y - gy)**2)
            ghost = np.exp(-(ghost_dist**2) / (2 * size**2))
            flare += ghost * intensity
        
        return flare
    
    def _apply_coating_effects(self,
                              image: np.ndarray,
                              profile: LensProfile,
                              strength: float) -> np.ndarray:
        """Apply coating-specific effects."""
        if profile.coating_type == "uncoated":
            # Add veiling glare
            glare = cv2.GaussianBlur(image, (51, 51), 20)
            result = image * (1 - 0.1 * strength) + glare * 0.1 * strength
            
        elif profile.coating_type == "single-coated":
            # Add slight color cast (purple/amber)
            if len(image.shape) == 3:
                color_cast = np.array([1.02, 1.0, 0.98])  # Slight amber
                result = image * color_cast
            else:
                result = image
                
        else:  # Multi-coated
            # Modern coatings - minimal effect
            result = image
        
        return np.clip(result, 0, 1)
    
    def _apply_hd_refinement(self,
                            image: np.ndarray,
                            profile: LensProfile,
                            strength: float) -> np.ndarray:
        """Apply HD-based refinement to match lens signature."""
        # Get target lens signature
        target_signature = self._get_lens_signature(profile)
        
        if target_signature is None:
            return image
        
        # Analyze current image
        current_analysis = self.hd_analyzer.analyze_and_correct(
            (image * 255).astype(np.uint8),
            mode='auto',
            strength=0.0  # Just analyze
        )
        
        current_signature = current_analysis['defect_hypervector']
        
        # Calculate difference vector
        diff_vector = target_signature - current_signature
        
        # Apply corrections to move toward target
        corrections = self._decode_corrections(diff_vector)
        
        # Blend corrections
        result = image
        blend_factor = self.config.hd_blend_factor * strength
        
        for correction_type, correction_map in corrections.items():
            if correction_type == 'sharpness':
                # Adjust sharpness
                kernel = np.array([[-1, -1, -1],
                                  [-1, 9 + correction_map, -1],
                                  [-1, -1, -1]]) / 9
                sharpened = cv2.filter2D(result, -1, kernel)
                result = result * (1 - blend_factor) + sharpened * blend_factor
                
            elif correction_type == 'contrast':
                # Adjust contrast
                mean = np.mean(result)
                contrasted = (result - mean) * (1 + correction_map) + mean
                result = result * (1 - blend_factor) + contrasted * blend_factor
        
        return np.clip(result, 0, 1)
    
    def _get_lens_signature(self, profile: LensProfile) -> Optional[np.ndarray]:
        """Get HD signature for lens profile."""
        # In production, would look up from database
        # For now, create synthetic signature
        if self.hd_encoder:
            # Create feature vector based on profile
            features = np.array([
                profile.vignetting_amount,
                profile.distortion_amount,
                profile.chromatic_aberration / 10,
                profile.bokeh_quality,
                profile.flare_intensity,
                float(profile.aperture_blades) / 12
            ])
            
            # Expand to HD dimension
            signature = np.zeros(self.hd_encoder.dim)
            for i, feat in enumerate(features):
                # Use different frequency for each feature
                freq = (i + 1) * 0.1
                for j in range(self.hd_encoder.dim):
                    signature[j] += feat * np.sin(freq * j / self.hd_encoder.dim)
            
            # Normalize
            signature /= np.linalg.norm(signature)
            
            return signature
        
        return None
    
    def _decode_corrections(self, diff_vector: np.ndarray) -> Dict[str, float]:
        """Decode correction parameters from difference vector."""
        # Simple decoding - in practice would use learned mapping
        corrections = {}
        
        # Extract features from different parts of vector
        dim = len(diff_vector)
        
        # Sharpness adjustment
        sharpness_component = np.mean(diff_vector[:dim//3])
        corrections['sharpness'] = np.tanh(sharpness_component * 10)
        
        # Contrast adjustment  
        contrast_component = np.mean(diff_vector[dim//3:2*dim//3])
        corrections['contrast'] = np.tanh(contrast_component * 10)
        
        return corrections
    
    def _create_motion_kernel(self, size: int, angle: float) -> np.ndarray:
        """Create motion blur kernel."""
        kernel = np.zeros((size, size))
        center = size // 2
        
        # Convert angle to radians
        angle_rad = np.deg2rad(angle)
        
        # Create line
        for i in range(size):
            x = int(center + (i - center) * np.cos(angle_rad))
            y = int(center + (i - center) * np.sin(angle_rad))
            
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1
        
        # Normalize
        kernel /= np.sum(kernel)
        
        return kernel
    
    def create_custom_profile(self,
                            name: str,
                            reference_images: List[np.ndarray]) -> LensProfile:
        """
        Create a custom lens profile from reference images.
        
        Args:
            name: Name for the profile
            reference_images: Example images from the lens
            
        Returns:
            Generated lens profile
        """
        logger.info(f"Creating custom profile: {name}")
        
        # Analyze reference images
        characteristics = []
        
        for img in reference_images:
            # Use optics engine to estimate parameters
            params = self.optics_engine.estimate_parameters(img)
            characteristics.append(params)
        
        # Average characteristics
        avg_params = {}
        for key in characteristics[0].keys():
            values = [c[key] for c in characteristics]
            avg_params[key] = np.mean(values)
        
        # Create profile
        profile = LensProfile(
            name=name,
            focal_length=50.0,  # Default
            max_aperture=2.0,   # Default
            vignetting_amount=avg_params.get('vignetting', 0.3),
            distortion_amount=abs(avg_params.get('k1', 0.0)),
            chromatic_aberration=avg_params.get('chromatic', 1.0),
            bokeh_quality=0.8,  # Default
            custom_parameters=avg_params
        )
        
        # Generate HD signature if available
        if self.hd_analyzer:
            profile.hd_signature = self.hd_analyzer.create_lens_signature(
                name,
                reference_images
            )
        
        return profile
    
    def blend_profiles(self,
                      profile1: LensProfile,
                      profile2: LensProfile,
                      blend_factor: float = 0.5) -> LensProfile:
        """
        Blend two lens profiles.
        
        Args:
            profile1: First profile
            profile2: Second profile  
            blend_factor: Blend amount (0 = profile1, 1 = profile2)
            
        Returns:
            Blended profile
        """
        # Linear interpolation of parameters
        blended = LensProfile(
            name=f"{profile1.name}_x_{profile2.name}",
            focal_length=profile1.focal_length * (1 - blend_factor) + 
                        profile2.focal_length * blend_factor,
            max_aperture=profile1.max_aperture * (1 - blend_factor) +
                        profile2.max_aperture * blend_factor,
            vignetting_amount=profile1.vignetting_amount * (1 - blend_factor) +
                             profile2.vignetting_amount * blend_factor,
            distortion_amount=profile1.distortion_amount * (1 - blend_factor) +
                             profile2.distortion_amount * blend_factor,
            chromatic_aberration=profile1.chromatic_aberration * (1 - blend_factor) +
                                profile2.chromatic_aberration * blend_factor,
            bokeh_quality=profile1.bokeh_quality * (1 - blend_factor) +
                         profile2.bokeh_quality * blend_factor
        )
        
        # Blend HD signatures if available
        if (hasattr(profile1, 'hd_signature') and 
            hasattr(profile2, 'hd_signature') and
            profile1.hd_signature is not None and
            profile2.hd_signature is not None):
            
            blended.hd_signature = (
                profile1.hd_signature * (1 - blend_factor) +
                profile2.hd_signature * blend_factor
            )
            blended.hd_signature /= np.linalg.norm(blended.hd_signature)
        
        return blended


# Convenience functions
def synthesize_lens_effect(image: Union[str, np.ndarray],
                          lens_name: str,
                          strength: float = 0.8) -> np.ndarray:
    """
    Apply a named lens effect to an image.
    
    Args:
        image: Input image
        lens_name: Name of lens profile
        strength: Effect strength
        
    Returns:
        Processed image
    """
    if isinstance(image, str):
        image = cv2.imread(image)
    
    # Get profile from library
    synthesizer = LensSynthesizer()
    library = CharacteristicLibrary()
    
    profile = library.get_profile(lens_name)
    if profile is None:
        raise ValueError(f"Unknown lens profile: {lens_name}")
    
    return synthesizer.apply(image, profile, strength=strength)
