"""
Enhanced quality metrics with hyperdimensional computing integration.

This module provides comprehensive quality assessment combining traditional
metrics with HD-based analysis for more robust quality evaluation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import cv2
from scipy import signal, ndimage
from skimage import metrics as skimage_metrics
import logging

from ..hyperdimensional import HyperdimensionalEncoder, analyze_lens_defects
from ..types.optics import ImageMetadata

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics."""
    # Traditional metrics
    sharpness: float = 0.0
    contrast: float = 0.0
    noise_level: float = 0.0
    dynamic_range: float = 0.0
    color_accuracy: float = 0.0
    
    # Advanced metrics
    local_contrast: float = 0.0
    edge_sharpness: float = 0.0
    texture_detail: float = 0.0
    
    # HD-based metrics
    hd_quality_score: float = 0.0
    defect_impact: float = 0.0
    
    # Perceptual metrics
    perceptual_quality: float = 0.0
    aesthetic_score: float = 0.0
    
    # Overall score
    overall_quality: float = 0.0
    
    # Detailed analysis
    quality_map: Optional[np.ndarray] = None
    sharpness_map: Optional[np.ndarray] = None
    contrast_map: Optional[np.ndarray] = None


class QualityAnalyzer:
    """
    Advanced quality analysis system combining traditional and HD methods.
    """
    
    def __init__(self, use_hd: bool = True):
        self.use_hd = use_hd
        self.hd_encoder = HyperdimensionalEncoder() if use_hd else None
        
        # Calibration parameters
        self.sharpness_threshold = 100.0
        self.contrast_threshold = 0.5
        self.noise_threshold = 10.0
        
    def analyze(self, 
                image: np.ndarray,
                reference: Optional[np.ndarray] = None,
                compute_maps: bool = False) -> QualityMetrics:
        """
        Perform comprehensive quality analysis.
        
        Args:
            image: Input image
            reference: Optional reference image for comparison
            compute_maps: Whether to compute detailed quality maps
            
        Returns:
            Complete quality metrics
        """
        metrics = QualityMetrics()
        
        # Traditional metrics
        self._compute_traditional_metrics(image, metrics)
        
        # Advanced metrics
        self._compute_advanced_metrics(image, metrics)
        
        # HD-based metrics if enabled
        if self.use_hd:
            self._compute_hd_metrics(image, metrics)
        
        # Perceptual metrics
        self._compute_perceptual_metrics(image, metrics)
        
        # Reference-based metrics if available
        if reference is not None:
            self._compute_reference_metrics(image, reference, metrics)
        
        # Compute quality maps if requested
        if compute_maps:
            self._compute_quality_maps(image, metrics)
        
        # Calculate overall quality
        metrics.overall_quality = self._calculate_overall_quality(metrics)
        
        return metrics
    
    def _compute_traditional_metrics(self, image: np.ndarray, metrics: QualityMetrics):
        """Compute traditional image quality metrics."""
        # Convert to grayscale for some metrics
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Sharpness (Laplacian variance)
        metrics.sharpness = self._measure_sharpness(gray)
        
        # Contrast (RMS contrast)
        metrics.contrast = self._measure_contrast(gray)
        
        # Noise level
        metrics.noise_level = self._estimate_noise(gray)
        
        # Dynamic range
        metrics.dynamic_range = self._measure_dynamic_range(gray)
        
        # Color accuracy (if color image)
        if len(image.shape) == 3:
            metrics.color_accuracy = self._measure_color_accuracy(image)
    
    def _measure_sharpness(self, gray: np.ndarray) -> float:
        """Measure image sharpness using multiple methods."""
        # Method 1: Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = laplacian.var()
        
        # Method 2: Gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mean = np.mean(grad_mag)
        
        # Method 3: Frequency domain analysis
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # High frequency content
        h, w = gray.shape
        center_h, center_w = h // 2, w // 2
        high_freq_region = magnitude_spectrum.copy()
        # Zero out low frequencies
        radius = min(h, w) // 4
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - center_w)**2 + (Y - center_h)**2)
        high_freq_region[dist < radius] = 0
        high_freq_content = np.sum(high_freq_region) / np.sum(magnitude_spectrum)
        
        # Combine methods
        normalized_lap = min(1.0, lap_var / self.sharpness_threshold)
        normalized_grad = min(1.0, grad_mean / 50.0)
        normalized_freq = high_freq_content
        
        # Weighted combination
        sharpness = 0.4 * normalized_lap + 0.3 * normalized_grad + 0.3 * normalized_freq
        
        return float(sharpness)
    
    def _measure_contrast(self, gray: np.ndarray) -> float:
        """Measure image contrast using multiple methods."""
        # Method 1: RMS contrast
        mean = np.mean(gray)
        rms_contrast = np.sqrt(np.mean((gray - mean)**2)) / (mean + 1e-8)
        
        # Method 2: Michelson contrast
        min_val, max_val = np.min(gray), np.max(gray)
        michelson = (max_val - min_val) / (max_val + min_val + 1e-8)
        
        # Method 3: Weber contrast (local)
        kernel_size = 21
        local_mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        local_diff = np.abs(gray - local_mean)
        weber_contrast = np.mean(local_diff / (local_mean + 1e-8))
        
        # Normalize and combine
        normalized_rms = min(1.0, rms_contrast / self.contrast_threshold)
        normalized_michelson = michelson
        normalized_weber = min(1.0, weber_contrast / 0.3)
        
        contrast = 0.4 * normalized_rms + 0.3 * normalized_michelson + 0.3 * normalized_weber
        
        return float(contrast)
    
    def _estimate_noise(self, gray: np.ndarray) -> float:
        """Estimate noise level in the image."""
        # Method 1: Median absolute deviation
        median = cv2.medianBlur(gray, 5)
        diff = gray.astype(np.float32) - median.astype(np.float32)
        mad = np.median(np.abs(diff))
        noise_mad = mad * 1.4826  # Scale factor for Gaussian noise
        
        # Method 2: Laplacian method
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        convolved = cv2.filter2D(gray, cv2.CV_64F, kernel)
        noise_lap = np.sqrt(0.5 * np.pi) * np.mean(np.abs(convolved))
        
        # Method 3: Local variance in smooth regions
        # Find smooth regions
        edges = cv2.Canny(gray, 50, 150)
        smooth_mask = edges == 0
        
        # Compute local variance
        local_var = ndimage.generic_filter(gray, np.var, size=5)
        smooth_var = local_var[smooth_mask]
        
        if len(smooth_var) > 0:
            noise_var = np.sqrt(np.median(smooth_var))
        else:
            noise_var = 0
        
        # Combine estimates
        noise_level = np.mean([noise_mad, noise_lap, noise_var])
        
        # Normalize (lower is better)
        normalized_noise = 1.0 - min(1.0, noise_level / self.noise_threshold)
        
        return float(normalized_noise)
    
    def _measure_dynamic_range(self, gray: np.ndarray) -> float:
        """Measure effective dynamic range."""
        # Compute histogram
        hist, _ = np.histogram(gray, bins=256, range=(0, 255))
        
        # Find effective range (ignore outliers)
        cumsum = np.cumsum(hist)
        total = cumsum[-1]
        
        # Find 0.1% and 99.9% percentiles
        low_idx = np.where(cumsum >= total * 0.001)[0][0]
        high_idx = np.where(cumsum >= total * 0.999)[0][0]
        
        # Effective range
        effective_range = high_idx - low_idx
        
        # Normalize to 0-1
        dynamic_range = effective_range / 255.0
        
        return float(dynamic_range)
    
    def _measure_color_accuracy(self, image: np.ndarray) -> float:
        """Measure color accuracy and balance."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Check color balance in AB channels
        a_mean, b_mean = np.mean(a), np.mean(b)
        
        # Perfect gray has a=128, b=128 (in 8-bit)
        a_deviation = abs(a_mean - 128) / 128
        b_deviation = abs(b_mean - 128) / 128
        
        # Color variance (should have some color variation)
        a_var, b_var = np.var(a), np.var(b)
        color_variety = min(1.0, (a_var + b_var) / 1000)
        
        # Saturation check
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        avg_saturation = np.mean(saturation) / 255
        
        # Combine metrics
        balance_score = 1.0 - (a_deviation + b_deviation) / 2
        color_score = 0.4 * balance_score + 0.3 * color_variety + 0.3 * avg_saturation
        
        return float(color_score)
    
    def _compute_advanced_metrics(self, image: np.ndarray, metrics: QualityMetrics):
        """Compute advanced quality metrics."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Local contrast
        metrics.local_contrast = self._measure_local_contrast(gray)
        
        # Edge sharpness
        metrics.edge_sharpness = self._measure_edge_sharpness(gray)
        
        # Texture detail
        metrics.texture_detail = self._measure_texture_detail(gray)
    
    def _measure_local_contrast(self, gray: np.ndarray) -> float:
        """Measure local contrast using adaptive methods."""
        # Compute local statistics
        kernel_size = 15
        local_mean = cv2.boxFilter(gray.astype(np.float32), -1, (kernel_size, kernel_size))
        local_mean_sq = cv2.boxFilter(gray.astype(np.float32)**2, -1, (kernel_size, kernel_size))
        local_var = local_mean_sq - local_mean**2
        local_std = np.sqrt(np.maximum(local_var, 0))
        
        # Local contrast = std / mean
        local_contrast = local_std / (local_mean + 1e-8)
        
        # Average local contrast
        avg_local_contrast = np.mean(local_contrast)
        
        # Normalize
        return float(min(1.0, avg_local_contrast / 0.5))
    
    def _measure_edge_sharpness(self, gray: np.ndarray) -> float:
        """Measure sharpness specifically at edges."""
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to get edge regions
        kernel = np.ones((5, 5), np.uint8)
        edge_regions = cv2.dilate(edges, kernel, iterations=1)
        
        # Compute gradient magnitude in edge regions
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Average gradient at edges
        edge_sharpness = np.mean(grad_mag[edge_regions > 0]) if np.any(edge_regions) else 0
        
        # Normalize
        return float(min(1.0, edge_sharpness / 100))
    
    def _measure_texture_detail(self, gray: np.ndarray) -> float:
        """Measure fine texture detail preservation."""
        # Use Gabor filters to detect texture
        texture_response = 0
        
        # Multiple orientations and frequencies
        for theta in np.linspace(0, np.pi, 4, endpoint=False):
            for frequency in [0.1, 0.2, 0.3]:
                # Create Gabor kernel
                kernel = cv2.getGaborKernel(
                    (21, 21), 
                    4.0,  # sigma
                    theta, 
                    10.0,  # wavelength
                    0.5,  # aspect ratio
                    0,  # phase
                    ktype=cv2.CV_32F
                )
                
                # Apply filter
                filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
                texture_response += np.mean(np.abs(filtered))
        
        # Normalize by number of filters
        texture_response /= 12
        
        # Normalize to 0-1
        return float(min(1.0, texture_response / 20))
    
    def _compute_hd_metrics(self, image: np.ndarray, metrics: QualityMetrics):
        """Compute HD-based quality metrics."""
        # Analyze defects using HD
        defect_analysis = analyze_lens_defects(image)
        
        # Calculate defect impact on quality
        total_defects = (
            defect_analysis['dust_spots'] +
            defect_analysis['scratches'] +
            defect_analysis['regions']
        )
        
        # Defect impact (lower is better, so invert)
        metrics.defect_impact = 1.0 / (1.0 + total_defects * 0.1)
        
        # HD quality score based on defect patterns
        if 'hypervector' in defect_analysis:
            # Compare with "perfect" hypervector (no defects)
            perfect_hv = np.zeros_like(defect_analysis['hypervector'])
            defect_magnitude = np.linalg.norm(defect_analysis['hypervector'])
            
            # Quality inversely proportional to defect magnitude
            metrics.hd_quality_score = 1.0 / (1.0 + defect_magnitude)
        else:
            metrics.hd_quality_score = metrics.defect_impact
    
    def _compute_perceptual_metrics(self, image: np.ndarray, metrics: QualityMetrics):
        """Compute perceptual quality metrics."""
        # Natural Image Quality Evaluator (NIQE)-inspired metrics
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Extract natural scene statistics
        # Mean subtracted contrast normalized (MSCN) coefficients
        mu = cv2.GaussianBlur(gray.astype(np.float64), (7, 7), 1.166)
        mu_sq = cv2.GaussianBlur(gray.astype(np.float64)**2, (7, 7), 1.166)
        sigma = np.sqrt(np.abs(mu_sq - mu**2))
        
        mscn = (gray - mu) / (sigma + 1)
        
        # Fit generalized Gaussian distribution to MSCN
        # Simplified: use kurtosis as quality indicator
        kurtosis = np.mean(mscn**4) / (np.mean(mscn**2)**2 + 1e-8) - 3
        
        # Natural images have kurtosis close to 0
        naturalness = np.exp(-abs(kurtosis) / 2)
        
        # Aesthetic score based on composition
        aesthetic = self._compute_aesthetic_score(image)
        
        # Combine
        metrics.perceptual_quality = 0.7 * naturalness + 0.3 * aesthetic
        metrics.aesthetic_score = aesthetic
    
    def _compute_aesthetic_score(self, image: np.ndarray) -> float:
        """Compute aesthetic quality score."""
        h, w = image.shape[:2]
        
        # Rule of thirds
        thirds_score = self._check_rule_of_thirds(image)
        
        # Color harmony
        color_score = self._check_color_harmony(image) if len(image.shape) == 3 else 0.5
        
        # Symmetry
        symmetry_score = self._check_symmetry(image)
        
        # Simplicity (not too cluttered)
        simplicity_score = self._check_simplicity(image)
        
        # Combine
        aesthetic = 0.3 * thirds_score + 0.3 * color_score + 0.2 * symmetry_score + 0.2 * simplicity_score
        
        return float(aesthetic)
    
    def _check_rule_of_thirds(self, image: np.ndarray) -> float:
        """Check if important features align with rule of thirds."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        
        # Simple saliency using gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        saliency_map = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize saliency
        saliency_map = saliency_map / (np.max(saliency_map) + 1e-8)
        
        # Check if salient regions align with thirds
        third_h, third_w = h // 3, w // 3
        
        # Define rule of thirds lines
        thirds_mask = np.zeros_like(saliency_map)
        # Vertical lines
        thirds_mask[:, third_w-10:third_w+10] = 1
        thirds_mask[:, 2*third_w-10:2*third_w+10] = 1
        # Horizontal lines
        thirds_mask[third_h-10:third_h+10, :] = 1
        thirds_mask[2*third_h-10:2*third_h+10, :] = 1
        
        # Score based on saliency alignment
        alignment = np.sum(saliency_map * thirds_mask) / (np.sum(saliency_map) + 1e-8)
        
        return float(min(1.0, alignment * 2))  # Scale up
    
    def _check_color_harmony(self, image: np.ndarray) -> float:
        """Check color harmony using color wheel theory."""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        
        # Compute hue histogram
        hist, _ = np.histogram(hue, bins=180, range=(0, 180))
        
        # Find dominant hues
        dominant_hues = []
        for i in range(len(hist)):
            if hist[i] > np.mean(hist):
                dominant_hues.append(i * 2)  # Convert to degrees
        
        if len(dominant_hues) < 2:
            return 0.5
        
        # Check for harmonic relationships
        harmony_score = 0
        for i in range(len(dominant_hues)):
            for j in range(i + 1, len(dominant_hues)):
                angle_diff = abs(dominant_hues[i] - dominant_hues[j])
                
                # Complementary (180°)
                if abs(angle_diff - 180) < 30:
                    harmony_score += 1
                # Triadic (120°)
                elif abs(angle_diff - 120) < 30:
                    harmony_score += 0.8
                # Analogous (30°)
                elif angle_diff < 60:
                    harmony_score += 0.6
        
        # Normalize
        max_possible = len(dominant_hues) * (len(dominant_hues) - 1) / 2
        return float(min(1.0, harmony_score / (max_possible + 1)))
    
    def _check_symmetry(self, image: np.ndarray) -> float:
        """Check image symmetry."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Horizontal symmetry
        flipped_h = cv2.flip(gray, 1)
        h_symmetry = 1 - np.mean(np.abs(gray - flipped_h)) / 255
        
        # Vertical symmetry
        flipped_v = cv2.flip(gray, 0)
        v_symmetry = 1 - np.mean(np.abs(gray - flipped_v)) / 255
        
        # Return max symmetry
        return float(max(h_symmetry, v_symmetry))
    
    def _check_simplicity(self, image: np.ndarray) -> float:
        """Check visual simplicity (not cluttered)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Edge density as clutter measure
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Texture complexity
        texture = self._measure_texture_detail(gray)
        
        # Simplicity is inverse of complexity
        simplicity = 1 - (0.6 * edge_density + 0.4 * texture)
        
        return float(simplicity)
    
    def _compute_reference_metrics(self, image: np.ndarray, reference: np.ndarray, metrics: QualityMetrics):
        """Compute metrics comparing to reference image."""
        # Ensure same size
        if image.shape != reference.shape:
            reference = cv2.resize(reference, (image.shape[1], image.shape[0]))
        
        # PSNR
        mse = np.mean((image - reference)**2)
        if mse > 0:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        else:
            psnr = 100
        
        # SSIM
        gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray2 = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if len(reference.shape) == 3 else reference
        
        ssim = skimage_metrics.structural_similarity(gray1, gray2)
        
        # Update metrics
        metrics.perceptual_quality = (metrics.perceptual_quality + ssim) / 2
    
    def _compute_quality_maps(self, image: np.ndarray, metrics: QualityMetrics):
        """Compute spatial quality maps."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Sharpness map
        metrics.sharpness_map = self._compute_sharpness_map(gray)
        
        # Contrast map  
        metrics.contrast_map = self._compute_contrast_map(gray)
        
        # Overall quality map
        metrics.quality_map = (metrics.sharpness_map + metrics.contrast_map) / 2
    
    def _compute_sharpness_map(self, gray: np.ndarray) -> np.ndarray:
        """Compute local sharpness map."""
        # Local Laplacian variance
        kernel_size = 15
        sharpness_map = np.zeros_like(gray, dtype=np.float32)
        
        for i in range(0, gray.shape[0] - kernel_size, kernel_size // 2):
            for j in range(0, gray.shape[1] - kernel_size, kernel_size // 2):
                patch = gray[i:i+kernel_size, j:j+kernel_size]
                laplacian = cv2.Laplacian(patch, cv2.CV_64F)
                sharpness = laplacian.var()
                
                # Normalize
                sharpness = min(1.0, sharpness / self.sharpness_threshold)
                
                # Fill region
                sharpness_map[i:i+kernel_size, j:j+kernel_size] = sharpness
        
        return sharpness_map
    
    def _compute_contrast_map(self, gray: np.ndarray) -> np.ndarray:
        """Compute local contrast map."""
        # Local standard deviation
        kernel_size = 15
        local_mean = cv2.boxFilter(gray.astype(np.float32), -1, (kernel_size, kernel_size))
        local_mean_sq = cv2.boxFilter(gray.astype(np.float32)**2, -1, (kernel_size, kernel_size))
        local_var = local_mean_sq - local_mean**2
        local_std = np.sqrt(np.maximum(local_var, 0))
        
        # Normalize
        contrast_map = local_std / (local_mean + 1e-8)
        contrast_map = np.minimum(contrast_map, 1.0)
        
        return contrast_map
    
    def _calculate_overall_quality(self, metrics: QualityMetrics) -> float:
        """Calculate overall quality score from individual metrics."""
        # Define weights for different metrics
        weights = {
            'sharpness': 0.15,
            'contrast': 0.10,
            'noise': 0.10,
            'dynamic_range': 0.05,
            'color': 0.05,
            'local_contrast': 0.10,
            'edge_sharpness': 0.10,
            'texture': 0.05,
            'hd_quality': 0.10,
            'defect_impact': 0.10,
            'perceptual': 0.10
        }
        
        # Weighted sum
        overall = (
            weights['sharpness'] * metrics.sharpness +
            weights['contrast'] * metrics.contrast +
            weights['noise'] * metrics.noise_level +
            weights['dynamic_range'] * metrics.dynamic_range +
            weights['color'] * metrics.color_accuracy +
            weights['local_contrast'] * metrics.local_contrast +
            weights['edge_sharpness'] * metrics.edge_sharpness +
            weights['texture'] * metrics.texture_detail +
            weights['hd_quality'] * metrics.hd_quality_score +
            weights['defect_impact'] * metrics.defect_impact +
            weights['perceptual'] * metrics.perceptual_quality
        )
        
        return float(overall)
    
    def compare_quality(self, image1: np.ndarray, image2: np.ndarray) -> Dict[str, float]:
        """Compare quality between two images."""
        # Analyze both images
        metrics1 = self.analyze(image1)
        metrics2 = self.analyze(image2)
        
        # Compute differences
        comparison = {
            'overall_diff': metrics2.overall_quality - metrics1.overall_quality,
            'sharpness_diff': metrics2.sharpness - metrics1.sharpness,
            'contrast_diff': metrics2.contrast - metrics1.contrast,
            'noise_diff': metrics2.noise_level - metrics1.noise_level,
            'perceptual_diff': metrics2.perceptual_quality - metrics1.perceptual_quality
        }
        
        # Determine which is better
        comparison['better_image'] = 2 if comparison['overall_diff'] > 0 else 1
        comparison['improvement_percent'] = abs(comparison['overall_diff']) * 100
        
        return comparison


# Convenience functions
def quick_quality_check(image: Union[str, np.ndarray]) -> float:
    """
    Quick quality check returning overall quality score.
    
    Args:
        image: Image path or array
        
    Returns:
        Overall quality score (0-1)
    """
    if isinstance(image, str):
        image = cv2.imread(image)
        
    analyzer = QualityAnalyzer()
    metrics = analyzer.analyze(image)
    return metrics.overall_quality


def detailed_quality_report(image: Union[str, np.ndarray]) -> str:
    """
    Generate detailed quality report.
    
    Args:
        image: Image path or array
        
    Returns:
        Formatted quality report string
    """
    if isinstance(image, str):
        image = cv2.imread(image)
        
    analyzer = QualityAnalyzer()
    metrics = analyzer.analyze(image, compute_maps=True)
    
    report = []
    report.append("=== Image Quality Report ===\n")
    report.append(f"Overall Quality: {metrics.overall_quality:.1%}\n")
    
    report.append("Technical Metrics:")
    report.append(f"  Sharpness: {metrics.sharpness:.1%}")
    report.append(f"  Contrast: {metrics.contrast:.1%}")
    report.append(f"  Noise Level: {metrics.noise_level:.1%}")
    report.append(f"  Dynamic Range: {metrics.dynamic_range:.1%}")
    report.append(f"  Color Accuracy: {metrics.color_accuracy:.1%}\n")
    
    report.append("Advanced Metrics:")
    report.append(f"  Local Contrast: {metrics.local_contrast:.1%}")
    report.append(f"  Edge Sharpness: {metrics.edge_sharpness:.1%}")
    report.append(f"  Texture Detail: {metrics.texture_detail:.1%}\n")
    
    report.append("Perceptual Metrics:")
    report.append(f"  Perceptual Quality: {metrics.perceptual_quality:.1%}")
    report.append(f"  Aesthetic Score: {metrics.aesthetic_score:.1%}\n")
    
    if metrics.hd_quality_score > 0:
        report.append("HD Analysis:")
        report.append(f"  HD Quality Score: {metrics.hd_quality_score:.1%}")
        report.append(f"  Defect Impact: {metrics.defect_impact:.1%}")
    
    return "\n".join(report)
