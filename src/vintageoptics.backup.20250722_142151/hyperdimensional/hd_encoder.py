"""
Hyperdimensional vector encoder for lens characteristics and defects.

This module provides high-dimensional vector representations that are robust
to noise and enable efficient similarity computations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DefectSignature:
    """Represents a lens defect with its characteristics."""
    type: str
    magnitude: float
    location: Tuple[float, float]
    spatial_extent: float
    
    
class HyperdimensionalEncoder:
    """
    Encodes lens defects and sensor errors as hyperdimensional vectors.
    
    Key properties:
    - High dimensionality (default 10,000) ensures quasi-orthogonality
    - Distributed representation is robust to noise
    - Supports both continuous (analog) and discrete (digital) patterns
    """
    
    def __init__(self, dimension: int = 10000, seed: int = 42):
        self.dim = dimension
        self.rng = np.random.RandomState(seed)
        
        # Initialize basis vectors for different defect types
        self.analog_bases = self._init_analog_bases()
        self.digital_bases = self._init_digital_bases()
        
        # Spatial encoding for location-dependent effects
        self.spatial_encoder = self._init_spatial_encoder()
        
    def _init_analog_bases(self) -> Dict[str, np.ndarray]:
        """Initialize smooth basis vectors for analog defects."""
        defect_types = [
            'dust', 'scratch', 'fungus', 'haze', 'separation',
            'coating_damage', 'oil', 'bubble', 'chip', 'cleaning_marks'
        ]
        
        bases = {}
        for defect in defect_types:
            # Create smooth, continuous patterns
            base = self.rng.randn(self.dim)
            # Apply smoothing via low-pass filter in frequency domain
            fft = np.fft.fft(base)
            # Keep only low frequencies for smooth variation
            cutoff = self.dim // 20
            fft[cutoff:-cutoff] = 0
            bases[defect] = np.real(np.fft.ifft(fft))
            bases[defect] /= np.linalg.norm(bases[defect])
            
        return bases
    
    def _init_digital_bases(self) -> Dict[str, np.ndarray]:
        """Initialize discrete basis vectors for digital artifacts."""
        artifact_types = [
            'hot_pixel', 'dead_pixel', 'banding', 'quantization',
            'demosaic_artifact', 'compression_block', 'readout_noise'
        ]
        
        bases = {}
        for artifact in artifact_types:
            # Create discrete, binary-like patterns
            base = self.rng.choice([-1, 1], size=self.dim)
            # Make it sparse to represent discrete nature
            mask = self.rng.random(self.dim) > 0.8
            base[~mask] = 0
            bases[artifact] = base / np.linalg.norm(base)
            
        return bases
    
    def _init_spatial_encoder(self) -> np.ndarray:
        """Initialize spatial encoding matrix for location-aware encoding."""
        # Create 2D frequency basis for spatial encoding
        freq_x = np.linspace(0, 10, int(np.sqrt(self.dim)))
        freq_y = np.linspace(0, 10, int(np.sqrt(self.dim)))
        
        spatial_basis = []
        for fx in freq_x[:10]:  # Use first 10 frequencies
            for fy in freq_y[:10]:
                basis = np.zeros(self.dim)
                # Create sinusoidal pattern
                for i in range(self.dim):
                    x = (i % int(np.sqrt(self.dim))) / np.sqrt(self.dim)
                    y = (i // int(np.sqrt(self.dim))) / np.sqrt(self.dim)
                    basis[i] = np.sin(2 * np.pi * fx * x) * np.cos(2 * np.pi * fy * y)
                spatial_basis.append(basis / np.linalg.norm(basis))
                
        return np.array(spatial_basis)
    
    def encode_vintage_defects(self, defects: List[DefectSignature]) -> np.ndarray:
        """
        Encode vintage lens defects as a hyperdimensional vector.
        
        Args:
            defects: List of defect signatures
            
        Returns:
            Normalized hyperdimensional vector
        """
        hv = np.zeros(self.dim)
        
        for defect in defects:
            if defect.type not in self.analog_bases:
                logger.warning(f"Unknown defect type: {defect.type}")
                continue
                
            # Get base vector for defect type
            base = self.analog_bases[defect.type].copy()
            
            # Modulate by magnitude
            base *= defect.magnitude
            
            # Add spatial encoding
            spatial_idx = int(defect.location[0] * 10) * 10 + int(defect.location[1] * 10)
            if spatial_idx < len(self.spatial_encoder):
                base = self._bind(base, self.spatial_encoder[spatial_idx])
            
            # Accumulate (bundle) all defects
            hv += base
            
        # Normalize to unit hypersphere
        if np.linalg.norm(hv) > 0:
            hv /= np.linalg.norm(hv)
            
        return hv
    
    def encode_sensor_errors(self, errors: List[Dict]) -> np.ndarray:
        """
        Encode digital sensor errors as a hyperdimensional vector.
        
        Args:
            errors: List of error descriptors
            
        Returns:
            Normalized hyperdimensional vector
        """
        hv = np.zeros(self.dim)
        
        for error in errors:
            error_type = error.get('type', 'unknown')
            if error_type not in self.digital_bases:
                logger.warning(f"Unknown error type: {error_type}")
                continue
                
            # Get base vector
            base = self.digital_bases[error_type].copy()
            
            # For digital errors, use XOR-like binding for discrete patterns
            if 'pattern' in error:
                pattern_hv = self._create_pattern_vector(error['pattern'])
                base = self._bind_discrete(base, pattern_hv)
            
            # Weight by severity
            severity = error.get('severity', 1.0)
            base *= severity
            
            hv += base
            
        # Normalize
        if np.linalg.norm(hv) > 0:
            hv /= np.linalg.norm(hv)
            
        return hv
    
    def _bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Bind two hypervectors using circular convolution."""
        return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))
    
    def _bind_discrete(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Bind discrete patterns using element-wise multiplication."""
        return a * np.sign(b)
    
    def _create_pattern_vector(self, pattern: np.ndarray) -> np.ndarray:
        """Convert a pattern array to a hypervector."""
        # Flatten and pad/truncate to dimension
        flat = pattern.flatten()
        if len(flat) < self.dim:
            # Repeat pattern to fill dimension
            hv = np.tile(flat, self.dim // len(flat) + 1)[:self.dim]
        else:
            # Subsample to fit dimension
            indices = np.linspace(0, len(flat)-1, self.dim, dtype=int)
            hv = flat[indices]
            
        return hv / np.linalg.norm(hv)
    
    def similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Compute cosine similarity between two hypervectors."""
        return np.dot(hv1, hv2)
    
    def bundle(self, hypervectors: List[np.ndarray]) -> np.ndarray:
        """Bundle multiple hypervectors by normalized addition."""
        result = np.sum(hypervectors, axis=0)
        return result / np.linalg.norm(result)
    
    def unbind(self, bound: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Unbind a hypervector using the inverse of the key."""
        # For circular convolution, unbinding uses correlation
        return np.real(np.fft.ifft(np.fft.fft(bound) / np.fft.fft(key)))
