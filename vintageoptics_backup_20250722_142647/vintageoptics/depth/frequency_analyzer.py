# src/vintageoptics/depth/frequency_analyzer.py

import cv2
import numpy as np
from typing import Dict

class FrequencyDepthAnalyzer:
    """Analyze depth using frequency domain characteristics"""
    
    def __init__(self):
        self.window_size = 64
        self.overlap = 0.5
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth map using local frequency analysis"""
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        
        # Initialize depth map
        depth_map = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)
        
        # Sliding window analysis
        window = np.hanning(self.window_size)
        window_2d = window[:, None] * window[None, :]
        
        stride = int(self.window_size * (1 - self.overlap))
        
        for y in range(0, h - self.window_size, stride):
            for x in range(0, w - self.window_size, stride):
                # Extract window
                patch = gray[y:y+self.window_size, x:x+self.window_size].astype(np.float32)
                
                # Apply window function
                windowed_patch = patch * window_2d
                
                # FFT analysis
                fft = np.fft.fft2(windowed_patch)
                fft_mag = np.abs(fft)
                
                # Analyze frequency distribution
                freq_features = self._extract_frequency_features(fft_mag)
                
                # Convert to depth
                depth_value = self._frequency_to_depth(freq_features)
                
                # Update depth map
                depth_map[y:y+self.window_size, x:x+self.window_size] += depth_value * window_2d
                weight_map[y:y+self.window_size, x:x+self.window_size] += window_2d
        
        # Normalize by weights
        depth_map = np.divide(depth_map, weight_map, 
                             out=np.zeros_like(depth_map), 
                             where=weight_map > 0)
        
        return depth_map
    
    def _extract_frequency_features(self, fft_mag: np.ndarray) -> Dict:
        """Extract depth-relevant features from frequency spectrum"""
        
        # Shift zero frequency to center
        fft_shifted = np.fft.fftshift(fft_mag)
        h, w = fft_shifted.shape
        center = (h // 2, w // 2)
        
        # Create radial frequency bins
        y, x = np.ogrid[:h, :w]
        radius = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        # Calculate radial average
        max_radius = int(np.min(center))
        radial_profile = np.zeros(max_radius)
        
        for r in range(max_radius):
            mask = (radius >= r) & (radius < r + 1)
            if np.any(mask):
                radial_profile[r] = np.mean(fft_shifted[mask])
        
        # Extract features
        features = {
            'high_freq_energy': np.sum(radial_profile[max_radius//2:]) / (np.sum(radial_profile) + 1e-6),
            'cutoff_frequency': self._find_cutoff_frequency(radial_profile),
            'spectral_slope': self._calculate_spectral_slope(radial_profile),
            'total_energy': np.sum(radial_profile)
        }
        
        return features
    
    def _frequency_to_depth(self, features: Dict) -> float:
        """Convert frequency features to depth estimate"""
        
        # Higher frequency content = sharper = closer
        # Lower frequency content = blurrier = farther
        
        # Normalize features
        high_freq = features['high_freq_energy']
        cutoff = features['cutoff_frequency'] / 32.0  # Normalize to [0,1]
        slope = features['spectral_slope']
        
        # Combine features
        sharpness_score = (high_freq * 0.4 + cutoff * 0.4 + (1 / (abs(slope) + 1)) * 0.2)
        
        # Invert to get depth (sharp = near = low depth value)
        depth = 1.0 - sharpness_score
        
        return depth
    
    def _find_cutoff_frequency(self, radial_profile: np.ndarray) -> float:
        """Find cutoff frequency in radial profile"""
        # Stub implementation
        return len(radial_profile) * 0.5
    
    def _calculate_spectral_slope(self, radial_profile: np.ndarray) -> float:
        """Calculate spectral slope"""
        # Stub implementation - simple linear fit
        if len(radial_profile) < 2:
            return 0.0
        x = np.arange(len(radial_profile))
        slope = np.polyfit(x, radial_profile, 1)[0]
        return slope
