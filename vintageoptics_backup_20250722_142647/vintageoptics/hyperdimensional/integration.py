"""
Integration module for hyperdimensional computing features with VintageOptics.

This module provides high-level functions to use HD computing for enhanced
error correction and defect analysis.
"""

import numpy as np
from typing import Dict, Optional, Union, List
import logging

from .hd_encoder import HyperdimensionalEncoder
from .error_separator import OrthogonalErrorSeparator
from .defect_topology import TopologicalDefectAnalyzer
from .constraint_solver import ConstraintBasedCorrector

logger = logging.getLogger(__name__)


class HyperdimensionalLensAnalyzer:
    """
    High-level interface for hyperdimensional lens analysis and correction.
    
    This class integrates HD computing capabilities into the VintageOptics
    workflow for advanced error correction and pattern recognition.
    """
    
    def __init__(self, dimension: int = 10000):
        """
        Initialize the HD analyzer.
        
        Args:
            dimension: Hypervector dimension (higher = more accurate but slower)
        """
        self.encoder = HyperdimensionalEncoder(dimension=dimension)
        self.separator = OrthogonalErrorSeparator(self.encoder)
        self.topology = TopologicalDefectAnalyzer(self.encoder)
        self.corrector = ConstraintBasedCorrector()
        
        # Store lens signatures for matching
        self.lens_signatures = {}
        
    def analyze_and_correct(self, 
                          image: np.ndarray,
                          mode: str = 'auto',
                          strength: float = 0.8) -> Dict[str, any]:
        """
        Perform comprehensive analysis and correction.
        
        Args:
            image: Input image
            mode: 'auto', 'vintage', 'digital', or 'hybrid'
            strength: Correction strength (0-1)
            
        Returns:
            Dictionary with corrected image and analysis results
        """
        results = {}
        
        # Topological analysis
        logger.info("Performing topological defect analysis...")
        topo = self.topology.analyze_defects(image)
        results['topology'] = topo
        results['defect_hypervector'] = topo['hypervector']
        
        # Determine correction mode
        if mode == 'auto':
            # Use adaptive separation to detect error types
            logger.info("Running adaptive error separation...")
            separation = self.separator.adaptive_separation(image)
            
            # Choose mode based on confidence
            if separation['vintage_confidence'] > 0.7:
                mode = 'vintage'
            elif separation['digital_confidence'] > 0.7:
                mode = 'digital'
            else:
                mode = 'hybrid'
                
            results['detected_mode'] = mode
            results['separation'] = separation
            
        # Apply appropriate correction
        if mode == 'vintage':
            results['corrected'] = self._correct_vintage_only(image, strength)
        elif mode == 'digital':
            results['corrected'] = self._correct_digital_only(image, strength)
        else:  # hybrid
            logger.info("Applying hybrid HD correction...")
            correction = self.corrector.correct_image(image, correction_strength=strength)
            results.update(correction)
            
        # Add quality metrics
        results['quality_score'] = self._compute_quality_score(
            image, 
            results['corrected']
        )
        
        return results
    
    def _correct_vintage_only(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Correct only vintage/analog defects."""
        # Create dummy digital hypervector (minimal digital errors)
        dummy_digital = np.zeros(self.encoder.dim)
        
        # Use topology to create vintage hypervector
        topo = self.topology.analyze_defects(image)
        vintage_hv = topo['hypervector']
        
        # Separate and correct
        separation = self.separator.separate_errors(image, vintage_hv, dummy_digital)
        
        # Blend correction
        corrected = image - separation['vintage_errors'] * strength
        return np.clip(corrected, 0, 255).astype(image.dtype)
    
    def _correct_digital_only(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Correct only digital/sensor defects."""
        # Create dummy vintage hypervector
        dummy_vintage = np.zeros(self.encoder.dim)
        
        # Detect digital patterns
        digital_errors = self.separator._detect_digital_patterns(image)
        digital_hv = self.encoder.encode_sensor_errors(digital_errors)
        
        # Separate and correct
        separation = self.separator.separate_errors(image, dummy_vintage, digital_hv)
        
        # Blend correction
        corrected = image - separation['digital_errors'] * strength
        return np.clip(corrected, 0, 255).astype(image.dtype)
    
    def _compute_quality_score(self, original: np.ndarray, corrected: np.ndarray) -> float:
        """Compute quality improvement score."""
        # Simple metric based on noise reduction and detail preservation
        import cv2
        
        # Noise estimation (lower is better)
        noise_orig = cv2.Laplacian(original, cv2.CV_64F).var()
        noise_corr = cv2.Laplacian(corrected, cv2.CV_64F).var()
        noise_improvement = (noise_orig - noise_corr) / (noise_orig + 1e-8)
        
        # Detail preservation (higher is better)
        edges_orig = cv2.Canny(original.astype(np.uint8), 50, 150)
        edges_corr = cv2.Canny(corrected.astype(np.uint8), 50, 150)
        detail_preservation = np.sum(edges_corr & edges_orig) / (np.sum(edges_orig) + 1e-8)
        
        # Combined score
        score = 0.6 * noise_improvement + 0.4 * detail_preservation
        return np.clip(score, 0, 1)
    
    def create_lens_signature(self, 
                             lens_name: str,
                             sample_images: List[np.ndarray]) -> np.ndarray:
        """
        Create a hyperdimensional signature for a specific lens.
        
        Args:
            lens_name: Name/ID of the lens
            sample_images: List of sample images from this lens
            
        Returns:
            Hypervector signature for the lens
        """
        logger.info(f"Creating HD signature for lens: {lens_name}")
        
        signatures = []
        for img in sample_images:
            # Analyze each sample
            topo = self.topology.analyze_defects(img)
            signatures.append(topo['hypervector'])
            
        # Bundle all signatures
        lens_signature = self.encoder.bundle(signatures)
        
        # Store for future matching
        self.lens_signatures[lens_name] = lens_signature
        
        return lens_signature
    
    def match_lens(self, 
                  image: np.ndarray,
                  threshold: float = 0.75) -> Optional[str]:
        """
        Match an image to known lens signatures.
        
        Args:
            image: Image to analyze
            threshold: Similarity threshold for matching
            
        Returns:
            Matched lens name or None
        """
        if not self.lens_signatures:
            logger.warning("No lens signatures available for matching")
            return None
            
        # Get image signature
        topo = self.topology.analyze_defects(image)
        img_signature = topo['hypervector']
        
        # Compare with known signatures
        best_match = None
        best_similarity = 0
        
        for lens_name, lens_sig in self.lens_signatures.items():
            similarity = self.encoder.similarity(img_signature, lens_sig)
            if similarity > best_similarity and similarity > threshold:
                best_similarity = similarity
                best_match = lens_name
                
        if best_match:
            logger.info(f"Matched lens: {best_match} (similarity: {best_similarity:.3f})")
        
        return best_match
    
    def iterative_enhancement(self,
                            image: np.ndarray,
                            target_quality: float = 0.8,
                            max_iterations: int = 5) -> Dict[str, any]:
        """
        Iteratively enhance image until target quality is reached.
        
        Args:
            image: Input image
            target_quality: Target quality score (0-1)
            max_iterations: Maximum iterations
            
        Returns:
            Enhanced image with iteration history
        """
        current = image.copy()
        history = []
        
        for i in range(max_iterations):
            # Apply correction with adaptive strength
            strength = 0.5 + 0.1 * i  # Increase strength each iteration
            result = self.analyze_and_correct(current, mode='hybrid', strength=strength)
            
            current = result['corrected']
            quality = result['quality_score']
            
            history.append({
                'iteration': i + 1,
                'quality_score': quality,
                'strength': strength
            })
            
            if quality >= target_quality:
                logger.info(f"Target quality reached after {i+1} iterations")
                break
                
        return {
            'enhanced': current,
            'final_quality': quality,
            'iterations': len(history),
            'history': history
        }


# Convenience functions for easy integration

def quick_hd_correction(image: np.ndarray, strength: float = 0.8) -> np.ndarray:
    """
    Quick one-line HD correction for vintage optics.
    
    Args:
        image: Input image
        strength: Correction strength (0-1)
        
    Returns:
        Corrected image
    """
    analyzer = HyperdimensionalLensAnalyzer()
    result = analyzer.analyze_and_correct(image, mode='auto', strength=strength)
    return result['corrected']


def analyze_lens_defects(image: np.ndarray) -> Dict[str, any]:
    """
    Analyze lens defects using topological and HD methods.
    
    Args:
        image: Input image
        
    Returns:
        Detailed defect analysis
    """
    analyzer = HyperdimensionalLensAnalyzer()
    topo = analyzer.topology.analyze_defects(image)
    
    # Format results for easy reading
    return {
        'dust_spots': len(topo['dust_features']),
        'scratches': len(topo['scratch_features']),
        'regions': len(topo['region_features']),
        'total_defects': topo['total_features'],
        'hypervector': topo['hypervector'],
        'details': topo
    }


def separate_vintage_digital_errors(image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Separate vintage and digital errors in an image.
    
    Args:
        image: Input image with mixed errors
        
    Returns:
        Dictionary with separated error components
    """
    analyzer = HyperdimensionalLensAnalyzer()
    separation = analyzer.separator.adaptive_separation(image)
    
    return {
        'clean': separation['clean'],
        'vintage_errors': separation['vintage_errors'],
        'digital_errors': separation['digital_errors'],
        'vintage_confidence': separation['vintage_confidence'],
        'digital_confidence': separation['digital_confidence']
    }
