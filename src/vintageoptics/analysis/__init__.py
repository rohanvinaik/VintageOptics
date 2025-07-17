# src/vintageoptics/analysis/__init__.py

"""
Image quality analysis and metrics
"""

import numpy as np
import cv2
from typing import Dict

class QualityAnalyzer:
    """Analyze image quality and processing results"""
    
    def __init__(self):
        pass
    
    def analyze(self, original: np.ndarray, processed: np.ndarray) -> Dict:
        """Analyze quality difference between original and processed images"""
        metrics = {}
        
        # Convert to grayscale for analysis
        if len(original.shape) == 3:
            orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            proc_gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        else:
            orig_gray = original
            proc_gray = processed
        
        # Sharpness analysis
        metrics['sharpness'] = self._analyze_sharpness(orig_gray, proc_gray)
        
        # Distortion analysis
        metrics['distortion'] = self._analyze_distortion(orig_gray, proc_gray)
        
        # Noise analysis
        metrics['noise'] = self._analyze_noise(orig_gray, proc_gray)
        
        # Overall quality score (0-1, higher is better)
        metrics['quality_score'] = self._calculate_overall_score(metrics)
        
        return metrics
    
    def _analyze_sharpness(self, original: np.ndarray, processed: np.ndarray) -> Dict:
        """Analyze sharpness improvements"""
        # Laplacian variance method
        orig_sharpness = cv2.Laplacian(original, cv2.CV_64F).var()
        proc_sharpness = cv2.Laplacian(processed, cv2.CV_64F).var()
        
        # Gradient magnitude method
        orig_grad = self._gradient_magnitude(original)
        proc_grad = self._gradient_magnitude(processed)
        
        return {
            'original_laplacian': float(orig_sharpness),
            'processed_laplacian': float(proc_sharpness),
            'laplacian_improvement': float(proc_sharpness / (orig_sharpness + 1e-6)),
            'original_gradient': float(orig_grad),
            'processed_gradient': float(proc_grad),
            'gradient_improvement': float(proc_grad / (orig_grad + 1e-6))
        }
    
    def _analyze_distortion(self, original: np.ndarray, processed: np.ndarray) -> Dict:
        """Analyze distortion correction effectiveness"""
        # Find lines using Hough transform
        orig_lines = self._detect_lines(original)
        proc_lines = self._detect_lines(processed)
        
        # Calculate line straightness
        orig_straightness = self._calculate_line_straightness(orig_lines)
        proc_straightness = self._calculate_line_straightness(proc_lines)
        
        return {
            'original_lines': len(orig_lines),
            'processed_lines': len(proc_lines),
            'original_straightness': orig_straightness,
            'processed_straightness': proc_straightness,
            'straightness_improvement': proc_straightness / (orig_straightness + 1e-6)
        }
    
    def _analyze_noise(self, original: np.ndarray, processed: np.ndarray) -> Dict:
        """Analyze noise characteristics"""
        # Estimate noise using high-frequency content
        orig_noise = self._estimate_noise(original)
        proc_noise = self._estimate_noise(processed)
        
        # Calculate PSNR
        mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
        psnr = 20 * np.log10(255.0 / (np.sqrt(mse) + 1e-6))
        
        return {
            'original_noise': float(orig_noise),
            'processed_noise': float(proc_noise),
            'noise_reduction': float((orig_noise - proc_noise) / (orig_noise + 1e-6)),
            'psnr': float(psnr)
        }
    
    def _gradient_magnitude(self, image: np.ndarray) -> float:
        """Calculate average gradient magnitude"""
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return np.mean(magnitude)
    
    def _detect_lines(self, image: np.ndarray) -> list:
        """Detect lines using Hough transform"""
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        return lines if lines is not None else []
    
    def _calculate_line_straightness(self, lines: list) -> float:
        """Calculate average line straightness"""
        if not lines:
            return 0.0
        
        # For now, return line count as a proxy for straightness
        # In a full implementation, this would analyze line curvature
        return len(lines) / 100.0  # Normalize
    
    def _estimate_noise(self, image: np.ndarray) -> float:
        """Estimate noise level using high-frequency content"""
        # Use high-pass filter to isolate noise
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        high_freq = cv2.filter2D(image.astype(np.float32), -1, kernel)
        return np.std(high_freq)
    
    def _calculate_overall_score(self, metrics: Dict) -> float:
        """Calculate overall quality score from individual metrics"""
        sharpness_score = min(1.0, metrics['sharpness']['gradient_improvement'] / 2.0)
        distortion_score = min(1.0, metrics['distortion']['straightness_improvement'] / 2.0)
        noise_score = max(0.0, 1.0 - metrics['noise']['processed_noise'] / 50.0)
        
        # Weighted average
        overall = (sharpness_score * 0.4 + distortion_score * 0.3 + noise_score * 0.3)
        return max(0.0, min(1.0, overall))

__all__ = ['QualityAnalyzer']
