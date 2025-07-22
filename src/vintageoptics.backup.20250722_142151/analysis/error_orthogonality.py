"""
Error Orthogonality Analysis Engine

Leverages the complementary nature of analog (physics-based) and digital (binary/noisy) 
error sources for mutual error rejection and signal enhancement.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from scipy import signal, stats
from sklearn.decomposition import FastICA
import cv2

@dataclass
class ErrorProfile:
    """Characterizes error patterns from different sources"""
    spatial_correlation: np.ndarray  # How errors correlate spatially
    frequency_signature: np.ndarray  # Fourier domain characteristics
    statistical_moments: Dict[str, float]  # Mean, variance, skew, kurtosis
    temporal_stability: float  # How consistent errors are over time
    source_type: str  # 'analog' or 'digital'


class OrthogonalErrorAnalyzer:
    """
    Analyzes and separates orthogonal error sources using signal processing
    and statistical independence principles.
    """
    
    def __init__(self):
        self.ica = FastICA(n_components=4, random_state=42)
        self.error_profiles: Dict[str, ErrorProfile] = {}
        
    def characterize_analog_errors(self, 
                                 vintage_output: np.ndarray,
                                 lens_params: Dict) -> ErrorProfile:
        """
        Characterize continuous, physics-based errors from vintage lenses.
        
        These errors typically show:
        - Smooth spatial gradients (vignetting)
        - Wavelength-dependent patterns (chromatic aberration)
        - Radial symmetry (barrel/pincushion distortion)
        - Continuous blur functions (spherical aberration)
        """
        h, w = vintage_output.shape[:2]
        center = (w//2, h//2)
        
        # Analyze radial patterns
        y, x = np.ogrid[:h, :w]
        radius_map = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        radius_norm = radius_map / radius_map.max()
        
        # Extract radial profile
        radial_bins = np.linspace(0, 1, 50)
        radial_profile = []
        for i in range(len(radial_bins)-1):
            mask = (radius_norm >= radial_bins[i]) & (radius_norm < radial_bins[i+1])
            if mask.any():
                radial_profile.append(vintage_output[mask].mean())
        
        # Frequency analysis for smooth gradients
        fft = np.fft.fft2(cv2.cvtColor(vintage_output, cv2.COLOR_BGR2GRAY))
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Low frequency dominance indicates smooth errors
        low_freq_ratio = np.sum(magnitude[:h//4, :w//4]) / np.sum(magnitude)
        
        # Spatial correlation analysis
        gray = cv2.cvtColor(vintage_output, cv2.COLOR_BGR2GRAY).astype(float)
        autocorr = signal.correlate2d(gray, gray, mode='same')
        autocorr_norm = autocorr / autocorr.max()
        
        return ErrorProfile(
            spatial_correlation=autocorr_norm,
            frequency_signature=magnitude,
            statistical_moments={
                'mean': np.mean(vintage_output),
                'variance': np.var(vintage_output),
                'skewness': stats.skew(vintage_output.flatten()),
                'kurtosis': stats.kurtosis(vintage_output.flatten())
            },
            temporal_stability=0.95,  # Analog errors are highly stable
            source_type='analog'
        )
    
    def characterize_digital_errors(self,
                                  digital_output: np.ndarray,
                                  sensor_params: Dict) -> ErrorProfile:
        """
        Characterize discrete, sensor-based errors from digital sources.
        
        These errors typically show:
        - Random noise patterns (shot noise, read noise)
        - Fixed pattern noise (hot pixels, column defects)
        - Quantization artifacts
        - Compression artifacts (if applicable)
        """
        h, w = digital_output.shape[:2]
        
        # Noise analysis using local variance
        gray = cv2.cvtColor(digital_output, cv2.COLOR_BGR2GRAY).astype(float)
        kernel_size = 5
        mean = cv2.blur(gray, (kernel_size, kernel_size))
        sqr_mean = cv2.blur(gray**2, (kernel_size, kernel_size))
        variance = sqr_mean - mean**2
        
        # Detect fixed pattern noise
        # Average multiple frames if available to isolate FPN
        fpn_estimate = self._estimate_fixed_pattern_noise(digital_output)
        
        # Frequency analysis for high-frequency noise
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # High frequency ratio indicates digital noise
        high_freq_ratio = 1 - np.sum(magnitude[:h//4, :w//4]) / np.sum(magnitude)
        
        # Detect quantization levels
        unique_values = len(np.unique(digital_output))
        quantization_score = unique_values / (256**3)  # Normalized by max possible
        
        return ErrorProfile(
            spatial_correlation=variance,
            frequency_signature=magnitude,
            statistical_moments={
                'mean': np.mean(digital_output),
                'variance': np.var(digital_output),
                'skewness': stats.skew(digital_output.flatten()),
                'kurtosis': stats.kurtosis(digital_output.flatten())
            },
            temporal_stability=0.3,  # Digital noise varies frame-to-frame
            source_type='digital'
        )
    
    def separate_orthogonal_components(self,
                                     mixed_signal: np.ndarray,
                                     analog_profile: ErrorProfile,
                                     digital_profile: ErrorProfile) -> Dict[str, np.ndarray]:
        """
        Use Independent Component Analysis to separate orthogonal error sources.
        
        The key insight: analog and digital errors are statistically independent
        due to their different physical origins.
        """
        # Reshape for ICA
        h, w, c = mixed_signal.shape
        mixed_flat = mixed_signal.reshape(-1, c)
        
        # Perform ICA
        sources = self.ica.fit_transform(mixed_flat)
        mixing_matrix = self.ica.mixing_
        
        # Classify components based on their characteristics
        components = {}
        for i in range(sources.shape[1]):
            component = sources[:, i].reshape(h, w)
            
            # Analyze component characteristics
            comp_fft = np.fft.fft2(component)
            comp_magnitude = np.abs(np.fft.fftshift(comp_fft))
            low_freq_ratio = np.sum(comp_magnitude[:h//4, :w//4]) / np.sum(comp_magnitude)
            
            # Classify based on frequency content
            if low_freq_ratio > 0.8:
                components['analog'] = component
            else:
                components['digital'] = component
        
        return components
    
    def mutual_error_rejection(self,
                             vintage_input: np.ndarray,
                             digital_input: np.ndarray,
                             confidence_threshold: float = 0.7) -> np.ndarray:
        """
        Core algorithm: Use orthogonal error characteristics for mutual rejection.
        
        Logic puzzle approach:
        - Where vintage shows smooth gradients AND digital shows noise → trust vintage
        - Where digital shows clean edges AND vintage shows aberration → trust digital
        - Where both agree → high confidence in signal
        - Where both disagree similarly → likely true signal variation
        """
        # Characterize both sources
        analog_profile = self.characterize_analog_errors(vintage_input, {})
        digital_profile = self.characterize_digital_errors(digital_input, {})
        
        # Compute local signal confidence maps
        vintage_confidence = self._compute_confidence_map(vintage_input, analog_profile)
        digital_confidence = self._compute_confidence_map(digital_input, digital_profile)
        
        # Decision logic for each pixel
        output = np.zeros_like(vintage_input)
        
        # Where vintage has high confidence (low physics-based distortion)
        vintage_mask = vintage_confidence > confidence_threshold
        output[vintage_mask] = vintage_input[vintage_mask]
        
        # Where digital has high confidence (low noise)
        digital_mask = digital_confidence > confidence_threshold
        output[digital_mask] = digital_input[digital_mask]
        
        # Where both have high confidence, blend based on relative confidence
        both_confident = vintage_mask & digital_mask
        if both_confident.any():
            vintage_weight = vintage_confidence[both_confident]
            digital_weight = digital_confidence[both_confident]
            total_weight = vintage_weight + digital_weight
            
            output[both_confident] = (
                vintage_input[both_confident] * (vintage_weight / total_weight)[:, np.newaxis] +
                digital_input[both_confident] * (digital_weight / total_weight)[:, np.newaxis]
            )
        
        # Where neither is confident, use advanced fusion
        neither_confident = ~(vintage_mask | digital_mask)
        if neither_confident.any():
            # Apply orthogonal component separation
            mixed = (vintage_input + digital_input) / 2
            components = self.separate_orthogonal_components(
                mixed, analog_profile, digital_profile
            )
            
            # Reconstruct from separated components
            if 'analog' in components and 'digital' in components:
                # Remove estimated errors from each domain
                output[neither_confident] = mixed[neither_confident] - \
                                          0.5 * components['analog'][neither_confident, np.newaxis] - \
                                          0.5 * components['digital'][neither_confident, np.newaxis]
        
        return output
    
    def _compute_confidence_map(self, 
                              image: np.ndarray, 
                              error_profile: ErrorProfile) -> np.ndarray:
        """
        Generate pixel-wise confidence based on local error characteristics.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
        
        if error_profile.source_type == 'analog':
            # For analog: confidence inversely related to optical aberrations
            # Use gradient magnitude as proxy for aberration strength
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Normalize and invert
            confidence = 1 - (grad_mag / grad_mag.max())
            
        else:  # digital
            # For digital: confidence inversely related to local noise
            # Use local variance as noise proxy
            kernel_size = 5
            mean = cv2.blur(gray, (kernel_size, kernel_size))
            sqr_mean = cv2.blur(gray**2, (kernel_size, kernel_size))
            variance = sqr_mean - mean**2
            
            # Normalize and invert
            confidence = 1 - (variance / variance.max())
        
        return confidence
    
    def _estimate_fixed_pattern_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate fixed pattern noise from single image using median filtering.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
        
        # Large median filter to estimate image without FPN
        filtered = cv2.medianBlur(gray.astype(np.uint8), 21).astype(float)
        
        # FPN estimate is the difference
        fpn = gray - filtered
        
        return fpn


class HybridErrorCorrector:
    """
    Orchestrates the complete error correction pipeline using orthogonal analysis.
    """
    
    def __init__(self):
        self.analyzer = OrthogonalErrorAnalyzer()
        self.correction_history: List[Dict] = []
    
    def process(self,
                vintage_image: np.ndarray,
                digital_image: np.ndarray,
                metadata: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Main processing pipeline for orthogonal error correction.
        
        Returns:
            Corrected image and detailed analysis report
        """
        # Ensure images are aligned
        if vintage_image.shape != digital_image.shape:
            digital_image = cv2.resize(digital_image, 
                                     (vintage_image.shape[1], vintage_image.shape[0]))
        
        # Perform mutual error rejection
        corrected = self.analyzer.mutual_error_rejection(
            vintage_image, 
            digital_image,
            confidence_threshold=0.7
        )
        
        # Generate analysis report
        report = {
            'vintage_characteristics': self.analyzer.characterize_analog_errors(
                vintage_image, metadata or {}
            ),
            'digital_characteristics': self.analyzer.characterize_digital_errors(
                digital_image, metadata or {}
            ),
            'correction_confidence': self._compute_overall_confidence(
                vintage_image, digital_image, corrected
            ),
            'processing_metadata': {
                'timestamp': np.datetime64('now'),
                'version': '1.0.0',
                'method': 'orthogonal_error_rejection'
            }
        }
        
        self.correction_history.append(report)
        
        return corrected, report
    
    def _compute_overall_confidence(self,
                                  vintage: np.ndarray,
                                  digital: np.ndarray,
                                  corrected: np.ndarray) -> float:
        """
        Compute overall confidence in the correction quality.
        """
        # MSE between inputs and output
        vintage_mse = np.mean((vintage - corrected)**2)
        digital_mse = np.mean((digital - corrected)**2)
        
        # Lower MSE to both inputs indicates good correction
        max_possible_mse = np.mean((vintage - digital)**2)
        confidence = 1 - (min(vintage_mse, digital_mse) / max_possible_mse)
        
        return float(np.clip(confidence, 0, 1))
