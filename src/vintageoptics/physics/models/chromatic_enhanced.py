# src/vintageoptics/physics/models/chromatic_enhanced.py
"""
Enhanced chromatic aberration correction with wavelength-specific processing.
Implements advanced CA correction from the enhancement analysis.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass
import cv2
from scipy import interpolate, signal
import numba
from numba import cuda
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChromaticAberrationParams:
    """Parameters for chromatic aberration correction"""
    # Lateral (transverse) CA parameters
    red_scale: float = 1.0
    green_scale: float = 1.0  # Reference channel
    blue_scale: float = 1.0
    
    # Longitudinal (axial) CA parameters
    red_focus_shift: float = 0.0
    blue_focus_shift: float = 0.0
    
    # Advanced parameters
    wavelength_coefficients: Dict[int, float] = None  # nm -> scale factor
    radial_dependency: List[float] = None  # Polynomial coefficients
    
    # Depth-aware parameters (for longitudinal CA)
    depth_influence: float = 0.0
    bokeh_preservation: bool = True
    
    def __post_init__(self):
        if self.wavelength_coefficients is None:
            # Default wavelength-specific corrections
            self.wavelength_coefficients = {
                440: self.blue_scale,    # Blue
                550: self.green_scale,   # Green (reference)
                640: self.red_scale      # Red
            }
        
        if self.radial_dependency is None:
            # Default radial dependency (constant)
            self.radial_dependency = [1.0]


class WavelengthModel:
    """Model for wavelength-dependent optical properties"""
    
    def __init__(self):
        # Standard RGB to wavelength approximations
        self.rgb_peak_wavelengths = {
            'R': 640,  # nm
            'G': 550,  # nm
            'B': 440   # nm
        }
        
        # CIE 1931 color matching functions (simplified)
        self.cie_wavelengths = np.array([380, 440, 490, 510, 550, 570, 590, 610, 640, 700, 780])
        self.cie_x = np.array([0.0014, 0.3483, 0.0956, 0.5047, 0.9950, 0.9520, 0.7570, 0.5030, 0.2650, 0.0107, 0.0])
        self.cie_y = np.array([0.0000, 0.0348, 0.3230, 0.8620, 0.9950, 0.8700, 0.6310, 0.3810, 0.1750, 0.0041, 0.0])
        self.cie_z = np.array([0.0065, 1.7721, 0.2720, 0.0782, 0.0203, 0.0087, 0.0040, 0.0020, 0.0000, 0.0000, 0.0])
        
    def rgb_to_wavelength_weights(self, rgb: np.ndarray) -> Dict[int, float]:
        """Convert RGB values to wavelength weights"""
        # Normalize RGB
        rgb_norm = rgb / (np.sum(rgb) + 1e-6)
        
        # Simple mapping (can be made more sophisticated)
        weights = {}
        
        # Primary wavelengths
        weights[self.rgb_peak_wavelengths['R']] = rgb_norm[0]
        weights[self.rgb_peak_wavelengths['G']] = rgb_norm[1]
        weights[self.rgb_peak_wavelengths['B']] = rgb_norm[2]
        
        # Add intermediate wavelengths
        weights[590] = (rgb_norm[0] + rgb_norm[1]) * 0.3  # Yellow
        weights[490] = (rgb_norm[1] + rgb_norm[2]) * 0.3  # Cyan
        
        return weights
    
    def dispersion_model(self, wavelength: float, material: str = 'standard') -> float:
        """Model chromatic dispersion for different lens materials"""
        # Simplified Sellmeier equation
        if material == 'standard':
            # Typical optical glass
            n_ref = 1.5168  # Reference index at 550nm
            A = 0.00175
            B = 0.00420
            
        elif material == 'low_dispersion':
            # ED/LD glass
            n_ref = 1.4970
            A = 0.00095
            B = 0.00280
            
        elif material == 'high_dispersion':
            # High dispersion glass (vintage)
            n_ref = 1.5350
            A = 0.00250
            B = 0.00580
        
        else:
            # Default
            n_ref = 1.5168
            A = 0.00175
            B = 0.00420
        
        # Calculate refractive index
        lambda_um = wavelength / 1000.0  # Convert to micrometers
        n = n_ref + A / (lambda_um ** 2) - B * (lambda_um ** 2)
        
        return n


class ChromaticAberrationCorrector:
    """Advanced chromatic aberration correction system"""
    
    def __init__(self, params: ChromaticAberrationParams = None):
        self.params = params or ChromaticAberrationParams()
        self.wavelength_model = WavelengthModel()
        self.use_gpu = cuda.is_available()
        
    def correct_chromatic_aberration(self, 
                                   image: np.ndarray,
                                   depth_map: Optional[np.ndarray] = None,
                                   preserve_bokeh: bool = True) -> np.ndarray:
        """Apply comprehensive chromatic aberration correction"""
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image  # No correction for non-color images
        
        # Separate corrections for lateral and longitudinal CA
        corrected = self.correct_lateral_ca(image)
        
        if depth_map is not None and self.params.depth_influence > 0:
            corrected = self.correct_longitudinal_ca(
                corrected, depth_map, preserve_bokeh
            )
        
        return corrected
    
    def correct_lateral_ca(self, image: np.ndarray) -> np.ndarray:
        """Correct lateral (transverse) chromatic aberration"""
        
        height, width = image.shape[:2]
        
        # Method selection based on parameters
        if len(self.params.wavelength_coefficients) > 3:
            # Use wavelength-specific correction
            return self._correct_lateral_ca_wavelength(image)
        else:
            # Use standard RGB channel scaling
            return self._correct_lateral_ca_rgb(image)
    
    def _correct_lateral_ca_rgb(self, image: np.ndarray) -> np.ndarray:
        """Standard RGB channel-based lateral CA correction"""
        
        height, width = image.shape[:2]
        center_x, center_y = width / 2, height / 2
        
        # Create output image
        corrected = np.zeros_like(image)
        
        # Process each channel
        for channel, scale in enumerate([self.params.red_scale, 
                                        self.params.green_scale, 
                                        self.params.blue_scale]):
            if abs(scale - 1.0) < 0.0001:
                # No correction needed
                corrected[:, :, channel] = image[:, :, channel]
            else:
                # Apply radial scaling
                if self.use_gpu and height * width > 1000000:
                    # Use GPU for large images
                    corrected[:, :, channel] = self._gpu_radial_scale(
                        image[:, :, channel], scale, center_x, center_y
                    )
                else:
                    # CPU implementation
                    corrected[:, :, channel] = self._radial_scale_channel(
                        image[:, :, channel], scale, center_x, center_y
                    )
        
        return corrected
    
    def _correct_lateral_ca_wavelength(self, image: np.ndarray) -> np.ndarray:
        """Wavelength-specific lateral CA correction"""
        
        height, width = image.shape[:2]
        center_x, center_y = width / 2, height / 2
        
        # Convert to spectral representation (simplified)
        spectral_bands = self._rgb_to_spectral_bands(image)
        
        # Correct each spectral band
        corrected_bands = {}
        for wavelength, band in spectral_bands.items():
            scale = self.params.wavelength_coefficients.get(wavelength, 1.0)
            
            if abs(scale - 1.0) > 0.0001:
                corrected_bands[wavelength] = self._radial_scale_channel(
                    band, scale, center_x, center_y
                )
            else:
                corrected_bands[wavelength] = band
        
        # Convert back to RGB
        corrected = self._spectral_bands_to_rgb(corrected_bands)
        
        return corrected
    
    def _radial_scale_channel(self, channel: np.ndarray, scale: float,
                            center_x: float, center_y: float) -> np.ndarray:
        """Apply radial scaling to a single channel"""
        
        height, width = channel.shape
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:height, 0:width].astype(np.float32)
        
        # Calculate scaled coordinates
        x_centered = x_coords - center_x
        y_centered = y_coords - center_y
        
        # Apply radial-dependent scaling if specified
        if len(self.params.radial_dependency) > 1:
            # Calculate radial distance (normalized)
            max_radius = np.sqrt(center_x**2 + center_y**2)
            r_norm = np.sqrt(x_centered**2 + y_centered**2) / max_radius
            
            # Evaluate polynomial
            radial_scale = np.polyval(self.params.radial_dependency[::-1], r_norm)
            effective_scale = 1.0 + (scale - 1.0) * radial_scale
        else:
            effective_scale = scale
        
        # Scale coordinates
        x_scaled = x_centered * effective_scale + center_x
        y_scaled = y_centered * effective_scale + center_y
        
        # Interpolate
        # Using RectBivariateSpline for better quality
        y_range = np.arange(height)
        x_range = np.arange(width)
        
        interpolator = interpolate.RectBivariateSpline(
            y_range, x_range, channel, kx=3, ky=3
        )
        
        # Clip coordinates to valid range
        x_scaled = np.clip(x_scaled, 0, width - 1)
        y_scaled = np.clip(y_scaled, 0, height - 1)
        
        # Interpolate values
        corrected = interpolator.ev(y_scaled, x_scaled)
        
        return np.clip(corrected, 0, 255).astype(channel.dtype)
    
    @staticmethod
    @numba.jit(nopython=True, parallel=True)
    def _radial_scale_numba(channel: np.ndarray, scale: float,
                           center_x: float, center_y: float,
                           radial_coeffs: np.ndarray) -> np.ndarray:
        """Numba-accelerated radial scaling"""
        
        height, width = channel.shape
        corrected = np.zeros_like(channel)
        max_radius = np.sqrt(center_x**2 + center_y**2)
        
        for y in numba.prange(height):
            for x in numba.prange(width):
                # Calculate source coordinates
                x_centered = x - center_x
                y_centered = y - center_y
                
                # Radial distance
                r = np.sqrt(x_centered**2 + y_centered**2)
                r_norm = r / max_radius
                
                # Radial-dependent scale
                radial_scale = 0.0
                for i, coeff in enumerate(radial_coeffs):
                    radial_scale += coeff * (r_norm ** i)
                
                effective_scale = 1.0 + (scale - 1.0) * radial_scale
                
                # Scaled coordinates
                x_src = x_centered * effective_scale + center_x
                y_src = y_centered * effective_scale + center_y
                
                # Bilinear interpolation
                if 0 <= x_src < width - 1 and 0 <= y_src < height - 1:
                    x_int = int(x_src)
                    y_int = int(y_src)
                    dx = x_src - x_int
                    dy = y_src - y_int
                    
                    # Sample four neighbors
                    p00 = channel[y_int, x_int]
                    p01 = channel[y_int, x_int + 1]
                    p10 = channel[y_int + 1, x_int]
                    p11 = channel[y_int + 1, x_int + 1]
                    
                    # Bilinear interpolation
                    value = (p00 * (1 - dx) * (1 - dy) +
                            p01 * dx * (1 - dy) +
                            p10 * (1 - dx) * dy +
                            p11 * dx * dy)
                    
                    corrected[y, x] = value
        
        return corrected
    
    def _gpu_radial_scale(self, channel: np.ndarray, scale: float,
                         center_x: float, center_y: float) -> np.ndarray:
        """GPU-accelerated radial scaling using CUDA"""
        
        if not self.use_gpu:
            return self._radial_scale_channel(channel, scale, center_x, center_y)
        
        # Transfer to GPU
        d_channel = cuda.to_device(channel)
        d_corrected = cuda.device_array_like(channel)
        
        # Configure kernel
        threads_per_block = (16, 16)
        blocks_per_grid = (
            (channel.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
            (channel.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
        )
        
        # Launch kernel
        self._radial_scale_kernel[blocks_per_grid, threads_per_block](
            d_channel, d_corrected, scale, center_x, center_y,
            self.params.radial_dependency
        )
        
        # Copy back
        return d_corrected.copy_to_host()
    
    @staticmethod
    @cuda.jit
    def _radial_scale_kernel(channel, corrected, scale, center_x, center_y, radial_coeffs):
        """CUDA kernel for radial scaling"""
        
        y, x = cuda.grid(2)
        
        if y < channel.shape[0] and x < channel.shape[1]:
            # Calculate source coordinates
            x_centered = x - center_x
            y_centered = y - center_y
            
            # Radial distance
            r = cuda.libdevice.sqrtf(x_centered**2 + y_centered**2)
            max_radius = cuda.libdevice.sqrtf(center_x**2 + center_y**2)
            r_norm = r / max_radius
            
            # Radial-dependent scale
            radial_scale = 0.0
            for i in range(len(radial_coeffs)):
                radial_scale += radial_coeffs[i] * cuda.libdevice.powf(r_norm, i)
            
            effective_scale = 1.0 + (scale - 1.0) * radial_scale
            
            # Scaled coordinates
            x_src = x_centered * effective_scale + center_x
            y_src = y_centered * effective_scale + center_y
            
            # Bilinear interpolation
            if 0 <= x_src < channel.shape[1] - 1 and 0 <= y_src < channel.shape[0] - 1:
                x_int = int(x_src)
                y_int = int(y_src)
                dx = x_src - x_int
                dy = y_src - y_int
                
                # Sample four neighbors
                p00 = channel[y_int, x_int]
                p01 = channel[y_int, x_int + 1]
                p10 = channel[y_int + 1, x_int]
                p11 = channel[y_int + 1, x_int + 1]
                
                # Bilinear interpolation
                value = (p00 * (1 - dx) * (1 - dy) +
                        p01 * dx * (1 - dy) +
                        p10 * (1 - dx) * dy +
                        p11 * dx * dy)
                
                corrected[y, x] = value
            else:
                corrected[y, x] = 0
    
    def correct_longitudinal_ca(self, image: np.ndarray, 
                              depth_map: np.ndarray,
                              preserve_bokeh: bool = True) -> np.ndarray:
        """Correct longitudinal (axial) chromatic aberration"""
        
        if depth_map.shape[:2] != image.shape[:2]:
            # Resize depth map if needed
            depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]))
        
        # Normalize depth map
        depth_norm = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map) + 1e-6)
        
        # Separate image into in-focus and out-of-focus regions
        focus_mask = self._create_focus_mask(depth_norm, threshold=0.3)
        
        corrected = image.copy()
        
        # Apply different corrections based on depth
        if preserve_bokeh:
            # Only correct in-focus areas strongly
            corrected = self._selective_longitudinal_correction(
                image, depth_norm, focus_mask
            )
        else:
            # Full correction
            corrected = self._full_longitudinal_correction(
                image, depth_norm
            )
        
        return corrected
    
    def _create_focus_mask(self, depth_map: np.ndarray, 
                         threshold: float = 0.3) -> np.ndarray:
        """Create mask for in-focus regions"""
        
        # Calculate depth gradient
        grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Low gradient = likely in focus
        focus_mask = gradient_mag < threshold
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        focus_mask = cv2.morphologyEx(focus_mask.astype(np.uint8), 
                                     cv2.MORPH_CLOSE, kernel)
        focus_mask = cv2.morphologyEx(focus_mask, cv2.MORPH_OPEN, kernel)
        
        return focus_mask.astype(float)
    
    def _selective_longitudinal_correction(self, image: np.ndarray,
                                         depth_map: np.ndarray,
                                         focus_mask: np.ndarray) -> np.ndarray:
        """Selective longitudinal CA correction preserving bokeh"""
        
        corrected = image.copy()
        
        # Create smooth transition weights
        weights = cv2.GaussianBlur(focus_mask, (21, 21), 5)
        
        # Apply depth-dependent shift to each channel
        for channel, shift in enumerate([self.params.red_focus_shift, 0, 
                                        self.params.blue_focus_shift]):
            if abs(shift) > 0.001:
                # Calculate effective shift based on depth
                depth_shift = shift * (1 - depth_map) * self.params.depth_influence
                
                # Apply weighted shift
                shifted = self._apply_depth_shift(image[:, :, channel], depth_shift)
                
                # Blend based on focus mask
                corrected[:, :, channel] = (
                    weights * shifted + 
                    (1 - weights) * image[:, :, channel]
                ).astype(image.dtype)
        
        return corrected
    
    def _full_longitudinal_correction(self, image: np.ndarray,
                                    depth_map: np.ndarray) -> np.ndarray:
        """Full longitudinal CA correction without bokeh preservation"""
        
        corrected = image.copy()
        
        for channel, shift in enumerate([self.params.red_focus_shift, 0, 
                                        self.params.blue_focus_shift]):
            if abs(shift) > 0.001:
                # Calculate depth-dependent shift
                depth_shift = shift * (1 - depth_map) * self.params.depth_influence
                
                # Apply shift
                corrected[:, :, channel] = self._apply_depth_shift(
                    image[:, :, channel], depth_shift
                )
        
        return corrected
    
    def _apply_depth_shift(self, channel: np.ndarray, 
                         depth_shift: np.ndarray) -> np.ndarray:
        """Apply depth-dependent focus shift to a channel"""
        
        # Convert shift to blur/sharpen operation
        corrected = channel.copy().astype(float)
        
        # Create multiple blur levels
        blur_levels = []
        max_blur = 5
        for i in range(max_blur):
            if i == 0:
                blur_levels.append(channel)
            else:
                kernel_size = 2 * i + 1
                blurred = cv2.GaussianBlur(channel, (kernel_size, kernel_size), i)
                blur_levels.append(blurred)
        
        # Interpolate between blur levels based on depth shift
        for y in range(channel.shape[0]):
            for x in range(channel.shape[1]):
                shift = depth_shift[y, x]
                
                if shift > 0:
                    # Positive shift = blur
                    level = min(shift * max_blur, max_blur - 1)
                    level_int = int(level)
                    level_frac = level - level_int
                    
                    if level_int < max_blur - 1:
                        corrected[y, x] = (
                            blur_levels[level_int][y, x] * (1 - level_frac) +
                            blur_levels[level_int + 1][y, x] * level_frac
                        )
                    else:
                        corrected[y, x] = blur_levels[level_int][y, x]
                
                elif shift < 0:
                    # Negative shift = sharpen
                    sharpened = 2 * channel[y, x] - blur_levels[1][y, x]
                    corrected[y, x] = channel[y, x] * (1 + shift) + sharpened * (-shift)
        
        return np.clip(corrected, 0, 255).astype(channel.dtype)
    
    def _rgb_to_spectral_bands(self, image: np.ndarray) -> Dict[int, np.ndarray]:
        """Convert RGB image to spectral bands (simplified)"""
        
        # Simple linear combination approach
        spectral_bands = {}
        
        # Primary wavelengths
        spectral_bands[640] = image[:, :, 0].astype(float)  # Red
        spectral_bands[550] = image[:, :, 1].astype(float)  # Green
        spectral_bands[440] = image[:, :, 2].astype(float)  # Blue
        
        # Interpolate intermediate wavelengths
        spectral_bands[590] = 0.7 * image[:, :, 0] + 0.3 * image[:, :, 1]  # Yellow
        spectral_bands[490] = 0.3 * image[:, :, 1] + 0.7 * image[:, :, 2]  # Cyan
        
        return spectral_bands
    
    def _spectral_bands_to_rgb(self, spectral_bands: Dict[int, np.ndarray]) -> np.ndarray:
        """Convert spectral bands back to RGB"""
        
        height, width = next(iter(spectral_bands.values())).shape
        rgb = np.zeros((height, width, 3), dtype=np.float32)
        
        # Reconstruction matrix (simplified)
        # This should ideally use proper color matching functions
        wavelength_weights = {
            440: np.array([0.0, 0.0, 1.0]),    # Blue
            490: np.array([0.0, 0.5, 0.5]),    # Cyan
            550: np.array([0.0, 1.0, 0.0]),    # Green
            590: np.array([0.5, 0.5, 0.0]),    # Yellow
            640: np.array([1.0, 0.0, 0.0])     # Red
        }
        
        # Accumulate contributions
        total_weight = np.zeros(3)
        
        for wavelength, band in spectral_bands.items():
            if wavelength in wavelength_weights:
                weight = wavelength_weights[wavelength]
                for c in range(3):
                    rgb[:, :, c] += band * weight[c]
                total_weight += weight
        
        # Normalize
        for c in range(3):
            if total_weight[c] > 0:
                rgb[:, :, c] /= total_weight[c]
        
        return np.clip(rgb, 0, 255).astype(np.uint8)
    
    def analyze_chromatic_aberration(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze chromatic aberration in the image"""
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            return {'ca_severity': 0.0}
        
        height, width = image.shape[:2]
        center_x, center_y = width / 2, height / 2
        
        # Extract channels
        red = image[:, :, 0].astype(float)
        green = image[:, :, 1].astype(float)
        blue = image[:, :, 2].astype(float)
        
        # Analyze edge regions
        edge_mask = self._create_edge_mask(image)
        
        # Measure channel misalignment at edges
        red_edges = cv2.Canny(red.astype(np.uint8), 50, 150)
        green_edges = cv2.Canny(green.astype(np.uint8), 50, 150)
        blue_edges = cv2.Canny(blue.astype(np.uint8), 50, 150)
        
        # Calculate misalignment
        rg_diff = np.sum(np.abs(red_edges - green_edges) * edge_mask)
        gb_diff = np.sum(np.abs(green_edges - blue_edges) * edge_mask)
        rb_diff = np.sum(np.abs(red_edges - blue_edges) * edge_mask)
        
        edge_pixels = np.sum(edge_mask)
        
        if edge_pixels > 0:
            lateral_ca = (rg_diff + gb_diff + rb_diff) / (3 * edge_pixels * 255)
        else:
            lateral_ca = 0.0
        
        # Analyze radial CA pattern
        radial_ca = self._analyze_radial_ca(red, green, blue, center_x, center_y)
        
        # Estimate required correction parameters
        estimated_params = self._estimate_correction_params(
            red, green, blue, lateral_ca, radial_ca
        )
        
        return {
            'lateral_ca_severity': lateral_ca,
            'radial_ca_pattern': radial_ca,
            'estimated_red_scale': estimated_params['red_scale'],
            'estimated_blue_scale': estimated_params['blue_scale'],
            'ca_type': 'lateral' if lateral_ca > 0.01 else 'minimal',
            'overall_severity': (lateral_ca + abs(radial_ca)) / 2
        }
    
    def _create_edge_mask(self, image: np.ndarray) -> np.ndarray:
        """Create mask for edge regions where CA is most visible"""
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Find edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to include nearby pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edge_mask = cv2.dilate(edges, kernel, iterations=2)
        
        return edge_mask > 0
    
    def _analyze_radial_ca(self, red: np.ndarray, green: np.ndarray, 
                          blue: np.ndarray, center_x: float, 
                          center_y: float) -> float:
        """Analyze radial chromatic aberration pattern"""
        
        height, width = red.shape
        
        # Sample at different radii
        radial_samples = []
        num_radii = 10
        num_angles = 16
        
        for r_idx in range(1, num_radii):
            radius = (r_idx / num_radii) * min(width, height) / 2
            
            rg_diffs = []
            gb_diffs = []
            
            for angle in np.linspace(0, 2 * np.pi, num_angles, endpoint=False):
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                
                if 0 <= x < width and 0 <= y < height:
                    # Window around point
                    window = 3
                    x1, x2 = max(0, x-window), min(width, x+window+1)
                    y1, y2 = max(0, y-window), min(height, y+window+1)
                    
                    # Compare channel alignments
                    r_patch = red[y1:y2, x1:x2]
                    g_patch = green[y1:y2, x1:x2]
                    b_patch = blue[y1:y2, x1:x2]
                    
                    rg_diff = np.mean(np.abs(r_patch - g_patch))
                    gb_diff = np.mean(np.abs(g_patch - b_patch))
                    
                    rg_diffs.append(rg_diff)
                    gb_diffs.append(gb_diff)
            
            if rg_diffs and gb_diffs:
                radial_samples.append({
                    'radius': radius,
                    'rg_diff': np.mean(rg_diffs),
                    'gb_diff': np.mean(gb_diffs)
                })
        
        # Analyze radial trend
        if len(radial_samples) > 3:
            radii = [s['radius'] for s in radial_samples]
            rg_diffs = [s['rg_diff'] for s in radial_samples]
            
            # Fit linear trend
            coeffs = np.polyfit(radii, rg_diffs, 1)
            radial_ca = coeffs[0]  # Slope indicates radial CA
        else:
            radial_ca = 0.0
        
        return radial_ca
    
    def _estimate_correction_params(self, red: np.ndarray, green: np.ndarray,
                                  blue: np.ndarray, lateral_ca: float,
                                  radial_ca: float) -> Dict[str, float]:
        """Estimate optimal correction parameters"""
        
        # Simple estimation based on CA analysis
        # In practice, this would use more sophisticated optimization
        
        params = {
            'red_scale': 1.0,
            'blue_scale': 1.0
        }
        
        # Lateral CA suggests channel scaling
        if lateral_ca > 0.01:
            # Estimate based on severity
            scale_adjustment = min(lateral_ca * 0.5, 0.02)
            
            if radial_ca > 0:
                # Red channel needs expansion
                params['red_scale'] = 1.0 + scale_adjustment
                params['blue_scale'] = 1.0 - scale_adjustment * 0.7
            else:
                # Blue channel needs expansion
                params['blue_scale'] = 1.0 + scale_adjustment
                params['red_scale'] = 1.0 - scale_adjustment * 0.7
        
        return params
    
    def create_ca_visualization(self, image: np.ndarray, 
                              corrected: np.ndarray) -> np.ndarray:
        """Create visualization showing CA correction effect"""
        
        height, width = image.shape[:2]
        
        # Create side-by-side comparison
        vis = np.zeros((height, width * 2, 3), dtype=np.uint8)
        
        # Original on left
        vis[:, :width] = image
        
        # Corrected on right
        vis[:, width:] = corrected
        
        # Add dividing line
        vis[:, width-2:width+2] = 128
        
        # Add channel difference visualization at bottom
        diff_height = 100
        diff_vis = np.zeros((diff_height, width * 2, 3), dtype=np.uint8)
        
        # Calculate channel differences
        for c in range(3):
            orig_diff = np.abs(image[:, :, c].astype(float) - 
                             image[:, :, 1].astype(float))
            corr_diff = np.abs(corrected[:, :, c].astype(float) - 
                             corrected[:, :, 1].astype(float))
            
            # Create heatmaps
            orig_heat = self._create_difference_heatmap(orig_diff, diff_height)
            corr_heat = self._create_difference_heatmap(corr_diff, diff_height)
            
            # Place in visualization
            y_start = c * (diff_height // 3)
            y_end = (c + 1) * (diff_height // 3)
            
            diff_vis[y_start:y_end, :width] = orig_heat[y_start:y_end]
            diff_vis[y_start:y_end, width:] = corr_heat[y_start:y_end]
        
        # Combine
        final_vis = np.vstack([vis, diff_vis])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(final_vis, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(final_vis, 'Corrected', (width + 10, 30), font, 1, (255, 255, 255), 2)
        
        return final_vis
    
    def _create_difference_heatmap(self, diff: np.ndarray, 
                                  target_height: int) -> np.ndarray:
        """Create heatmap visualization of channel differences"""
        
        # Resize to target height
        aspect = diff.shape[1] / diff.shape[0]
        target_width = int(target_height * aspect)
        
        diff_resized = cv2.resize(diff, (target_width, target_height))
        
        # Normalize
        diff_norm = np.clip(diff_resized / np.max(diff_resized + 1e-6), 0, 1)
        
        # Apply colormap
        heatmap = cv2.applyColorMap((diff_norm * 255).astype(np.uint8), 
                                   cv2.COLORMAP_JET)
        
        return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
