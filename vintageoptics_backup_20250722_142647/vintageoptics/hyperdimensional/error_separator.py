"""
Orthogonal error separation using hyperdimensional computing principles.

This module leverages the orthogonality of analog and digital error patterns
to perform mutual error rejection and signal enhancement.
"""

import numpy as np
from typing import Tuple, Optional, Dict
import cv2
from scipy import signal
from scipy.sparse.linalg import svds
import logging

from .hd_encoder import HyperdimensionalEncoder

logger = logging.getLogger(__name__)


class OrthogonalErrorSeparator:
    """
    Separates analog (vintage lens) and digital (sensor) errors using their
    orthogonal characteristics in high-dimensional space.
    
    Key insight: Analog errors are continuous/smooth while digital errors are
    discrete/quantized, making them nearly orthogonal in HD space.
    """
    
    def __init__(self, hd_encoder: Optional[HyperdimensionalEncoder] = None):
        self.encoder = hd_encoder or HyperdimensionalEncoder()
        
        # Projection matrices for analog/digital subspaces
        self.analog_projection = None
        self.digital_projection = None
        
        # Initialize subspace bases
        self._init_subspaces()
        
    def _init_subspaces(self):
        """Initialize orthogonal subspaces for analog and digital patterns."""
        # Create analog subspace from smooth basis functions
        analog_basis = []
        for i in range(100):  # 100 smooth basis functions
            basis = np.zeros(self.encoder.dim)
            # Create smooth sinusoidal pattern
            freq = (i + 1) * 0.1
            phase = self.encoder.rng.random() * 2 * np.pi
            for j in range(self.encoder.dim):
                basis[j] = np.sin(freq * j / self.encoder.dim + phase)
            analog_basis.append(basis / np.linalg.norm(basis))
            
        self.analog_projection = np.array(analog_basis).T
        
        # Create digital subspace from discrete patterns
        digital_basis = []
        for i in range(100):  # 100 discrete basis functions
            basis = self.encoder.rng.choice([-1, 0, 1], size=self.encoder.dim)
            # Make sparse and blocky
            block_size = self.encoder.rng.randint(10, 50)
            for j in range(0, self.encoder.dim, block_size):
                if j + block_size < self.encoder.dim:
                    basis[j:j+block_size] = basis[j]
            digital_basis.append(basis / np.linalg.norm(basis))
            
        self.digital_projection = np.array(digital_basis).T
        
    def separate_errors(self, 
                       image: np.ndarray,
                       vintage_hv: np.ndarray,
                       digital_hv: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Separate vintage and digital errors from the combined image.
        
        Args:
            image: Input image with mixed errors
            vintage_hv: Hypervector encoding of expected vintage defects
            digital_hv: Hypervector encoding of expected digital errors
            
        Returns:
            Dictionary with 'clean', 'vintage_errors', and 'digital_errors'
        """
        # Convert image to frequency domain for analysis
        if len(image.shape) == 3:
            # Process each channel separately
            results = []
            for c in range(image.shape[2]):
                channel_result = self._separate_channel(
                    image[:, :, c], vintage_hv, digital_hv
                )
                results.append(channel_result)
            
            # Combine channels
            clean = np.stack([r['clean'] for r in results], axis=-1)
            vintage = np.stack([r['vintage'] for r in results], axis=-1)
            digital = np.stack([r['digital'] for r in results], axis=-1)
            
        else:
            result = self._separate_channel(image, vintage_hv, digital_hv)
            clean = result['clean']
            vintage = result['vintage'] 
            digital = result['digital']
            
        return {
            'clean': clean,
            'vintage_errors': vintage,
            'digital_errors': digital
        }
    
    def _separate_channel(self, 
                         channel: np.ndarray,
                         vintage_hv: np.ndarray,
                         digital_hv: np.ndarray) -> Dict[str, np.ndarray]:
        """Separate errors in a single channel."""
        h, w = channel.shape
        
        # Extract features from channel
        features = self._extract_features(channel)
        
        # Project to hyperdimensional space
        feature_hv = self.encoder._create_pattern_vector(features)
        
        # Compute projections onto analog and digital subspaces
        analog_component = self._project_analog(feature_hv, vintage_hv)
        digital_component = self._project_digital(feature_hv, digital_hv)
        
        # Reconstruct error patterns
        vintage_pattern = self._reconstruct_analog_errors(channel, analog_component)
        digital_pattern = self._reconstruct_digital_errors(channel, digital_component)
        
        # Apply constraint satisfaction
        clean = self._apply_constraints(channel, vintage_pattern, digital_pattern)
        
        return {
            'clean': clean,
            'vintage': vintage_pattern,
            'digital': digital_pattern
        }
    
    def _extract_features(self, channel: np.ndarray) -> np.ndarray:
        """Extract multi-scale features from image channel."""
        features = []
        
        # Gradient features (edge detection)
        grad_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        features.extend([grad_x.flatten(), grad_y.flatten()])
        
        # Frequency features
        dft = np.fft.fft2(channel)
        features.append(np.abs(dft).flatten())
        
        # Local statistics
        kernel_sizes = [3, 5, 7]
        for ks in kernel_sizes:
            local_mean = cv2.blur(channel, (ks, ks))
            local_var = cv2.blur(channel**2, (ks, ks)) - local_mean**2
            features.extend([local_mean.flatten(), local_var.flatten()])
            
        # Concatenate all features
        all_features = np.concatenate(features)
        
        return all_features
    
    def _project_analog(self, feature_hv: np.ndarray, vintage_hv: np.ndarray) -> np.ndarray:
        """Project features onto analog subspace guided by vintage hypervector."""
        # Weight projection by similarity to vintage pattern
        similarity = self.encoder.similarity(feature_hv, vintage_hv)
        
        # Project onto analog subspace
        coeffs = self.analog_projection.T @ feature_hv
        
        # Weight by vintage similarity
        coeffs *= (1 + similarity) / 2
        
        # Reconstruct in original space
        return self.analog_projection @ coeffs
    
    def _project_digital(self, feature_hv: np.ndarray, digital_hv: np.ndarray) -> np.ndarray:
        """Project features onto digital subspace guided by digital hypervector."""
        # Weight projection by similarity to digital pattern
        similarity = self.encoder.similarity(feature_hv, digital_hv)
        
        # Project onto digital subspace
        coeffs = self.digital_projection.T @ feature_hv
        
        # Weight by digital similarity
        coeffs *= (1 + similarity) / 2
        
        # Reconstruct in original space
        return self.digital_projection @ coeffs
    
    def _reconstruct_analog_errors(self, channel: np.ndarray, 
                                  analog_component: np.ndarray) -> np.ndarray:
        """Reconstruct analog error pattern from hyperdimensional representation."""
        h, w = channel.shape
        
        # Use low-rank approximation for smooth analog errors
        # Reshape channel for SVD
        U, s, Vt = svds(channel, k=min(20, min(h, w) - 1))
        
        # Weight singular values by analog component strength
        weights = analog_component[:len(s)] / np.max(np.abs(analog_component[:len(s)]))
        s_weighted = s * np.abs(weights)
        
        # Reconstruct smooth component
        analog_pattern = U @ np.diag(s_weighted) @ Vt
        
        # Apply smoothing to ensure continuity
        analog_pattern = cv2.GaussianBlur(analog_pattern, (5, 5), 1.0)
        
        return analog_pattern
    
    def _reconstruct_digital_errors(self, channel: np.ndarray,
                                   digital_component: np.ndarray) -> np.ndarray:
        """Reconstruct digital error pattern from hyperdimensional representation."""
        h, w = channel.shape
        
        # Digital errors are typically blocky and discrete
        # Use median filter to extract sharp transitions
        median = cv2.medianBlur(channel.astype(np.uint8), 3)
        digital_pattern = channel - median
        
        # Threshold to make discrete
        threshold = np.std(digital_pattern) * 0.5
        digital_pattern[np.abs(digital_pattern) < threshold] = 0
        
        # Weight by digital component strength
        strength = np.linalg.norm(digital_component) / self.encoder.dim
        digital_pattern *= strength
        
        return digital_pattern
    
    def _apply_constraints(self, channel: np.ndarray,
                          vintage_pattern: np.ndarray,
                          digital_pattern: np.ndarray) -> np.ndarray:
        """Apply constraint satisfaction to separate true signal from errors."""
        # Constraint 1: Analog errors are smooth
        vintage_smooth = cv2.bilateralFilter(
            vintage_pattern.astype(np.float32), 9, 75, 75
        )
        
        # Constraint 2: Digital errors are sparse and discrete
        digital_sparse = np.where(
            np.abs(digital_pattern) > np.std(digital_pattern), 
            digital_pattern, 
            0
        )
        
        # Constraint 3: Errors shouldn't overlap significantly
        overlap = np.abs(vintage_smooth) * np.abs(digital_sparse)
        overlap_penalty = overlap / (np.max(overlap) + 1e-8)
        
        # Adjust patterns based on overlap
        vintage_adjusted = vintage_smooth * (1 - overlap_penalty)
        digital_adjusted = digital_sparse * (1 - overlap_penalty)
        
        # Clean signal is original minus adjusted errors
        clean = channel - vintage_adjusted - digital_adjusted
        
        # Ensure valid range
        clean = np.clip(clean, 0, np.max(channel))
        
        return clean
    
    def adaptive_separation(self, image: np.ndarray, 
                          confidence_threshold: float = 0.7) -> Dict[str, np.ndarray]:
        """
        Perform adaptive error separation without prior knowledge of defects.
        
        Args:
            image: Input image
            confidence_threshold: Minimum confidence for error detection
            
        Returns:
            Separated components with confidence scores
        """
        # Detect potential defects adaptively
        vintage_defects = self._detect_analog_patterns(image)
        digital_errors = self._detect_digital_patterns(image)
        
        # Encode detected patterns
        vintage_hv = self.encoder.encode_vintage_defects(vintage_defects)
        digital_hv = self.encoder.encode_sensor_errors(digital_errors)
        
        # Compute confidence scores
        vintage_confidence = self._compute_analog_confidence(image, vintage_defects)
        digital_confidence = self._compute_digital_confidence(image, digital_errors)
        
        # Perform separation if confidence is high enough
        if vintage_confidence > confidence_threshold or digital_confidence > confidence_threshold:
            result = self.separate_errors(image, vintage_hv, digital_hv)
            result['vintage_confidence'] = vintage_confidence
            result['digital_confidence'] = digital_confidence
            return result
        else:
            # Low confidence - return original image
            return {
                'clean': image.copy(),
                'vintage_errors': np.zeros_like(image),
                'digital_errors': np.zeros_like(image),
                'vintage_confidence': vintage_confidence,
                'digital_confidence': digital_confidence
            }
    
    def _detect_analog_patterns(self, image: np.ndarray) -> list:
        """Detect potential analog/vintage defect patterns."""
        from ..analysis.lens_characterizer import DefectSignature
        
        defects = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect dust/spots (circular patterns)
        circles = cv2.HoughCircles(
            gray.astype(np.uint8), 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=20,
            param1=50, 
            param2=30, 
            minRadius=2, 
            maxRadius=50
        )
        
        if circles is not None:
            for circle in circles[0]:
                x, y, r = circle
                defects.append(DefectSignature(
                    type='dust',
                    magnitude=r / max(image.shape[:2]),
                    location=(x / image.shape[1], y / image.shape[0]),
                    spatial_extent=r
                ))
        
        # Detect haze (low contrast regions)
        contrast = cv2.Laplacian(gray, cv2.CV_64F).var()
        if contrast < 100:  # Low contrast indicates haze
            defects.append(DefectSignature(
                type='haze',
                magnitude=1 - contrast / 100,
                location=(0.5, 0.5),
                spatial_extent=1.0
            ))
            
        return defects
    
    def _detect_digital_patterns(self, image: np.ndarray) -> list:
        """Detect potential digital artifact patterns."""
        errors = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect hot/dead pixels
        median = cv2.medianBlur(gray.astype(np.uint8), 3)
        diff = np.abs(gray.astype(np.float32) - median.astype(np.float32))
        hot_pixels = diff > 3 * np.std(diff)
        
        if np.any(hot_pixels):
            errors.append({
                'type': 'hot_pixel',
                'severity': np.sum(hot_pixels) / hot_pixels.size,
                'pattern': hot_pixels.astype(np.float32)
            })
            
        # Detect banding
        fft = np.fft.fft2(gray)
        fft_mag = np.abs(fft)
        
        # Look for periodic patterns in frequency domain
        h, w = fft_mag.shape
        center_h, center_w = h // 2, w // 2
        
        # Check for horizontal/vertical bands
        h_line = fft_mag[center_h, :]
        v_line = fft_mag[:, center_w]
        
        h_peaks = signal.find_peaks(h_line, height=np.mean(h_line) * 3)[0]
        v_peaks = signal.find_peaks(v_line, height=np.mean(v_line) * 3)[0]
        
        if len(h_peaks) > 2 or len(v_peaks) > 2:
            errors.append({
                'type': 'banding',
                'severity': 0.5,
                'pattern': np.ones_like(gray) * 0.1
            })
            
        return errors
    
    def _compute_analog_confidence(self, image: np.ndarray, defects: list) -> float:
        """Compute confidence score for analog defect detection."""
        if not defects:
            return 0.0
            
        # Base confidence on number and severity of defects
        total_magnitude = sum(d.magnitude for d in defects)
        defect_coverage = sum(d.spatial_extent for d in defects) / max(image.shape[:2])
        
        confidence = min(1.0, total_magnitude * defect_coverage * len(defects) / 10)
        return confidence
    
    def _compute_digital_confidence(self, image: np.ndarray, errors: list) -> float:
        """Compute confidence score for digital error detection."""
        if not errors:
            return 0.0
            
        # Base confidence on error severity and coverage
        total_severity = sum(e['severity'] for e in errors)
        
        confidence = min(1.0, total_severity * len(errors) / 5)
        return confidence
