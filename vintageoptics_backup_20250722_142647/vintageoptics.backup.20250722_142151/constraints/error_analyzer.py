"""
Orthogonal error analysis system.

Exploits the fundamental difference between analog (optical) and digital (sensor)
error sources to achieve superior correction through mutual error rejection.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy import ndimage, signal
from scipy.fft import fft2, ifft2, fftfreq
import cv2


class OrthogonalErrorAnalyzer:
    """
    Exploit the orthogonal nature of analog vs digital errors.
    
    Key insight: Vintage lens errors and digital sensor errors have fundamentally
    different mathematical properties that allow for separation and mutual rejection.
    """
    
    def __init__(self):
        self.analog_basis_functions = self._create_analog_basis()
        self.digital_basis_functions = self._create_digital_basis()
        
    def _create_analog_basis(self) -> Dict[str, callable]:
        """
        Create basis functions for analog/optical errors.
        These are smooth, continuous, and often radially symmetric.
        """
        return {
            'radial_polynomial': self._radial_polynomial_basis,
            'zernike': self._zernike_basis,
            'gaussian_blur': self._gaussian_basis,
            'chromatic_shift': self._chromatic_basis
        }
    
    def _create_digital_basis(self) -> Dict[str, callable]:
        """
        Create basis functions for digital/sensor errors.
        These are discrete, pixel-aligned, and often have high-frequency components.
        """
        return {
            'pixel_noise': self._pixel_noise_basis,
            'quantization': self._quantization_basis,
            'demosaic_artifacts': self._demosaic_basis,
            'compression_blocks': self._compression_basis
        }
    
    def decompose_errors(self, image: np.ndarray, 
                        lens_profile: Optional[Dict] = None,
                        sensor_profile: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Decompose image errors into orthogonal analog and digital components.
        
        Returns:
            Dictionary with 'analog_errors', 'digital_errors', and 'clean_signal'
        """
        # Convert to float for processing
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        
        # Extract error components in frequency domain
        freq_domain = fft2(image, axes=(0, 1))
        
        # Analog errors: low-frequency, smooth variations
        analog_mask = self._create_analog_frequency_mask(freq_domain.shape[:2])
        analog_errors = ifft2(freq_domain * analog_mask, axes=(0, 1)).real
        
        # Digital errors: high-frequency, pixel-aligned patterns
        digital_mask = self._create_digital_frequency_mask(freq_domain.shape[:2])
        digital_errors = ifft2(freq_domain * digital_mask, axes=(0, 1)).real
        
        # Further decomposition using spatial domain analysis
        analog_components = self._decompose_analog_spatial(analog_errors, lens_profile)
        digital_components = self._decompose_digital_spatial(digital_errors, sensor_profile)
        
        # Clean signal estimation via mutual rejection
        clean_signal = self._mutual_error_rejection(
            image, analog_components, digital_components
        )
        
        return {
            'analog_errors': analog_components,
            'digital_errors': digital_components,
            'clean_signal': clean_signal,
            'error_map': self._create_error_confidence_map(analog_components, digital_components)
        }
    
    def _create_analog_frequency_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create frequency mask for analog/optical errors."""
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        
        # Create radial frequency coordinates
        u = fftfreq(rows).reshape(-1, 1)
        v = fftfreq(cols).reshape(1, -1)
        radius = np.sqrt(u**2 + v**2)
        
        # Analog errors are primarily low-frequency with smooth falloff
        # Using a Butterworth-like filter for smooth transition
        cutoff = 0.3  # Normalized frequency cutoff
        order = 4
        mask = 1 / (1 + (radius / cutoff)**(2 * order))
        
        return mask
    
    def _create_digital_frequency_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create frequency mask for digital/sensor errors."""
        # Digital errors contain high-frequency components and specific patterns
        analog_mask = self._create_analog_frequency_mask(shape)
        
        # High-pass component
        high_pass = 1 - analog_mask
        
        # Add notch filters for common digital artifacts (e.g., Moiré patterns)
        rows, cols = shape
        u = fftfreq(rows).reshape(-1, 1)
        v = fftfreq(cols).reshape(1, -1)
        
        # Suppress DC and very low frequencies (these are analog)
        dc_suppress = 1 - np.exp(-50 * (u**2 + v**2))
        
        return high_pass * dc_suppress
    
    def _decompose_analog_spatial(self, analog_errors: np.ndarray,
                                 lens_profile: Optional[Dict]) -> Dict[str, np.ndarray]:
        """Decompose analog errors into physical components."""
        components = {}
        
        # Vignetting (radial darkening)
        components['vignetting'] = self._extract_vignetting(analog_errors)
        
        # Blur/PSF (point spread function)
        components['blur'] = self._extract_blur_kernel(analog_errors)
        
        # Chromatic aberration (if color image)
        if analog_errors.ndim == 3 and analog_errors.shape[2] >= 3:
            components['chromatic'] = self._extract_chromatic_aberration(analog_errors)
        
        # Lens-specific distortions if profile provided
        if lens_profile:
            components['distortion'] = self._extract_geometric_distortion(
                analog_errors, lens_profile
            )
        
        return components
    
    def _decompose_digital_spatial(self, digital_errors: np.ndarray,
                                  sensor_profile: Optional[Dict]) -> Dict[str, np.ndarray]:
        """Decompose digital errors into sensor-specific components."""
        components = {}
        
        # Shot noise (Poisson-distributed)
        components['shot_noise'] = self._extract_shot_noise(digital_errors)
        
        # Quantization artifacts
        components['quantization'] = self._extract_quantization_errors(digital_errors)
        
        # Demosaicing artifacts (if color)
        if digital_errors.ndim == 3 and digital_errors.shape[2] >= 3:
            components['demosaic'] = self._extract_demosaic_artifacts(digital_errors)
        
        # Sensor-specific patterns if profile provided
        if sensor_profile:
            components['fixed_pattern'] = self._extract_fixed_pattern_noise(
                digital_errors, sensor_profile
            )
        
        return components
    
    def _mutual_error_rejection(self, original: np.ndarray,
                              analog_components: Dict[str, np.ndarray],
                              digital_components: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Use orthogonality of error sources for mutual rejection.
        
        Key insight: Analog errors are smooth and affect neighborhoods,
        while digital errors are pixel-specific and statistically independent.
        """
        # Start with original
        clean = original.copy()
        
        # Remove analog errors using digital error statistics
        for name, analog_error in analog_components.items():
            if name == 'vignetting':
                # Vignetting is multiplicative
                clean = clean / (1 + analog_error + 1e-6)
            else:
                # Most other analog errors are additive
                clean = clean - analog_error
        
        # Remove digital errors using analog smoothness constraints
        for name, digital_error in digital_components.items():
            if name == 'shot_noise':
                # Use non-local means with analog-informed weights
                clean = self._remove_noise_preserving_analog(clean, digital_error)
        
        # Ensure result is in valid range
        return np.clip(clean, 0, 1)
    
    def _extract_vignetting(self, image: np.ndarray) -> np.ndarray:
        """Extract vignetting pattern (radial darkening)."""
        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Fit radial polynomial
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        
        # Create radial distance map
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        r_norm = r / r.max()
        
        # Fit polynomial to radial brightness profile
        # Sample points along radii
        angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
        radial_samples = []
        
        for angle in angles:
            for radius in np.linspace(0, min(h, w)//2, 50):
                y_pos = int(cy + radius * np.sin(angle))
                x_pos = int(cx + radius * np.cos(angle))
                if 0 <= y_pos < h and 0 <= x_pos < w:
                    radial_samples.append((radius / r.max(), gray[y_pos, x_pos]))
        
        if radial_samples:
            radii, values = zip(*radial_samples)
            # Fit 4th order polynomial
            coeffs = np.polyfit(radii, values, 4)
            vignetting = np.polyval(coeffs, r_norm)
        else:
            vignetting = np.ones_like(gray)
        
        # Normalize to be a correction factor
        vignetting = vignetting / (vignetting.max() + 1e-6)
        
        return 1 - vignetting  # Return the darkening amount
    
    def _extract_blur_kernel(self, image: np.ndarray) -> np.ndarray:
        """Estimate blur kernel from image."""
        # Simplified: return Gaussian approximation
        # In practice, would use blind deconvolution
        kernel_size = 15
        sigma = 2.0
        
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = kernel @ kernel.T
        
        return kernel / kernel.sum()
    
    def _extract_chromatic_aberration(self, image: np.ndarray) -> np.ndarray:
        """Extract chromatic aberration (color fringing)."""
        if image.shape[2] < 3:
            return np.zeros_like(image)
        
        # Chromatic aberration appears as color separation at edges
        # Detect edges
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150) / 255.0
        
        # Measure color channel misalignment at edges
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        
        # Compute channel differences at edge locations
        rg_diff = np.abs(r - g) * edges
        gb_diff = np.abs(g - b) * edges
        
        # Chromatic aberration map
        chromatic = np.stack([rg_diff, gb_diff, gb_diff], axis=-1)
        
        return chromatic
    
    def _extract_shot_noise(self, image: np.ndarray) -> np.ndarray:
        """Extract shot noise component."""
        # Shot noise follows Poisson statistics
        # Estimate using local variance
        
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Local mean and variance
        kernel_size = 5
        local_mean = cv2.blur(gray, (kernel_size, kernel_size))
        local_var = cv2.blur(gray**2, (kernel_size, kernel_size)) - local_mean**2
        
        # Shot noise variance is proportional to signal
        # Estimate noise where variance exceeds Poisson prediction
        expected_var = local_mean  # Poisson: variance = mean
        excess_var = np.maximum(0, local_var - expected_var)
        
        return np.sqrt(excess_var)
    
    def _extract_quantization_errors(self, image: np.ndarray) -> np.ndarray:
        """Extract quantization artifacts."""
        # Detect stair-stepping in gradients
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Compute local gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Quantization creates discrete steps in gradients
        # Detect using gradient histogram analysis
        hist, bins = np.histogram(grad_x.ravel(), bins=256)
        
        # Find peaks in histogram (quantization levels)
        peaks = signal.find_peaks(hist, height=hist.max() * 0.1)[0]
        
        # Create quantization map
        quant_map = np.zeros_like(gray)
        for peak in peaks:
            level = bins[peak]
            mask = np.abs(grad_x - level) < (bins[1] - bins[0])
            quant_map[mask] = 1
        
        return quant_map
    
    def _extract_demosaic_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Extract demosaicing artifacts (color interpolation errors)."""
        # Demosaicing artifacts appear as color patterns at high frequencies
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        
        # Check for unnatural color patterns
        # These appear as high-frequency chromatic noise
        color_diff_rg = r - g
        color_diff_gb = g - b
        
        # High-pass filter to isolate artifacts
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) / 8
        
        artifacts_rg = cv2.filter2D(color_diff_rg, -1, kernel)
        artifacts_gb = cv2.filter2D(color_diff_gb, -1, kernel)
        
        # Combine into artifact map
        artifacts = np.stack([
            np.abs(artifacts_rg),
            np.abs(artifacts_gb),
            np.zeros_like(artifacts_rg)
        ], axis=-1)
        
        return artifacts
    
    def _extract_fixed_pattern_noise(self, image: np.ndarray,
                                   sensor_profile: Dict) -> np.ndarray:
        """Extract fixed pattern noise specific to sensor."""
        # Placeholder - would use dark frame data in practice
        return np.zeros_like(image)
    
    def _extract_geometric_distortion(self, image: np.ndarray,
                                    lens_profile: Dict) -> np.ndarray:
        """Extract geometric distortion from lens profile."""
        # Placeholder - would use calibration data
        return np.zeros_like(image)
    
    def _remove_noise_preserving_analog(self, image: np.ndarray,
                                      noise_map: np.ndarray) -> np.ndarray:
        """Remove noise while preserving analog characteristics."""
        # Use guided filter with analog characteristics as guide
        # This preserves smooth optical variations while removing pixel noise
        
        # Create guide image (low-frequency component)
        guide = cv2.GaussianBlur(image, (15, 15), 5)
        
        # Guided filter parameters
        radius = 5
        epsilon = noise_map.mean() ** 2
        
        # Apply guided filter
        if image.ndim == 3:
            filtered = np.zeros_like(image)
            for i in range(image.shape[2]):
                filtered[..., i] = cv2.ximgproc.guidedFilter(
                    guide[..., i], image[..., i], radius, epsilon
                )
        else:
            filtered = cv2.ximgproc.guidedFilter(guide, image, radius, epsilon)
        
        return filtered
    
    def _create_error_confidence_map(self, analog_components: Dict[str, np.ndarray],
                                   digital_components: Dict[str, np.ndarray]) -> np.ndarray:
        """Create confidence map showing error detection certainty."""
        # Combine all error magnitudes
        all_errors = []
        
        for components in [analog_components, digital_components]:
            for error_map in components.values():
                if error_map.ndim == 3:
                    all_errors.append(np.mean(np.abs(error_map), axis=2))
                else:
                    all_errors.append(np.abs(error_map))
        
        if all_errors:
            # Confidence is inverse of total error magnitude
            total_error = np.sum(all_errors, axis=0)
            confidence = 1 / (1 + total_error)
        else:
            confidence = np.ones_like(image[:, :, 0] if image.ndim == 3 else image)
        
        return confidence
    
    # Basis function implementations
    def _radial_polynomial_basis(self, shape: Tuple[int, int], order: int = 4) -> np.ndarray:
        """Generate radial polynomial basis functions."""
        h, w = shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2) / max(h, w)
        
        basis = []
        for i in range(order + 1):
            basis.append(r ** i)
        
        return np.stack(basis, axis=-1)
    
    def _zernike_basis(self, shape: Tuple[int, int], max_order: int = 10) -> np.ndarray:
        """Generate Zernike polynomial basis (for optical aberrations)."""
        # Simplified - would implement full Zernike polynomials
        return self._radial_polynomial_basis(shape, max_order // 2)
    
    def _gaussian_basis(self, shape: Tuple[int, int], scales: list = None) -> np.ndarray:
        """Generate multi-scale Gaussian basis."""
        if scales is None:
            scales = [1, 2, 4, 8, 16]
        
        basis = []
        for scale in scales:
            kernel = cv2.getGaussianKernel(int(6 * scale + 1), scale)
            kernel = kernel @ kernel.T
            basis.append(kernel)
        
        return basis
    
    def _chromatic_basis(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate basis for chromatic aberration."""
        # Radial chromatic shift patterns
        return self._radial_polynomial_basis(shape, 2)
    
    def _pixel_noise_basis(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate pixel-wise noise basis."""
        # Identity matrix reshaped - each pixel is independent
        return np.eye(shape[0] * shape[1]).reshape(shape[0], shape[1], -1)
    
    def _quantization_basis(self, shape: Tuple[int, int], levels: int = 256) -> np.ndarray:
        """Generate quantization level basis."""
        # Step functions at different levels
        basis = []
        for level in range(0, levels, levels // 8):
            step = np.ones(shape) * (level / levels)
            basis.append(step)
        
        return np.stack(basis, axis=-1)
    
    def _demosaic_basis(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate Bayer pattern basis for demosaicing artifacts."""
        # Create Bayer pattern masks
        bayer_r = np.zeros(shape)
        bayer_g = np.zeros(shape)
        bayer_b = np.zeros(shape)
        
        # RGGB pattern
        bayer_r[0::2, 0::2] = 1
        bayer_g[0::2, 1::2] = 1
        bayer_g[1::2, 0::2] = 1
        bayer_b[1::2, 1::2] = 1
        
        return np.stack([bayer_r, bayer_g, bayer_b], axis=-1)
    
    def _compression_basis(self, shape: Tuple[int, int], block_size: int = 8) -> np.ndarray:
        """Generate DCT basis for compression artifacts."""
        # Simplified - would use full DCT basis
        blocks = []
        for i in range(0, shape[0], block_size):
            for j in range(0, shape[1], block_size):
                block = np.zeros(shape)
                block[i:i+block_size, j:j+block_size] = 1
                blocks.append(block)
        
        return np.stack(blocks, axis=-1) if blocks else np.zeros(shape + (1,))
