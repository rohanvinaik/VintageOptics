"""
Topological analysis of lens defects for robust feature detection.

This module uses topological data analysis to identify persistent features
in lens defects that are invariant to noise and transformations.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import cv2
from scipy.spatial.distance import cdist
from scipy import ndimage
from dataclasses import dataclass
import logging

from .hd_encoder import HyperdimensionalEncoder

logger = logging.getLogger(__name__)


@dataclass 
class TopologicalFeature:
    """Represents a topological feature with persistence."""
    dimension: int  # 0 = point, 1 = loop, 2 = void
    birth: float
    death: float
    location: Tuple[float, float]
    persistence: float
    feature_vector: Optional[np.ndarray] = None
    
    @property
    def lifetime(self):
        return self.death - self.birth


class TopologicalDefectAnalyzer:
    """
    Analyzes lens defects using topological data analysis.
    
    Identifies persistent topological features that correspond to:
    - 0-dimensional: dust, spots, point defects
    - 1-dimensional: scratches, cracks, linear defects  
    - 2-dimensional: regions of haze, separation, area defects
    """
    
    def __init__(self, hd_encoder: Optional[HyperdimensionalEncoder] = None):
        self.encoder = hd_encoder or HyperdimensionalEncoder()
        self.noise_threshold = 0.05  # Minimum persistence to consider
        
    def analyze_defects(self, image: np.ndarray) -> Dict[str, any]:
        """
        Perform topological analysis of defects in the image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with topological features and hyperdimensional encoding
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Compute persistence diagram
        features = self._compute_persistence(gray)
        
        # Filter by persistence threshold
        significant_features = [
            f for f in features 
            if f.persistence > self.noise_threshold * (np.max(gray) - np.min(gray))
        ]
        
        # Classify by dimension
        dust = [f for f in significant_features if f.dimension == 0]
        scratches = [f for f in significant_features if f.dimension == 1]
        regions = [f for f in significant_features if f.dimension == 2]
        
        # Encode as hypervector
        hv = self._encode_topological_features(dust, scratches, regions)
        
        return {
            'dust_features': dust,
            'scratch_features': scratches,
            'region_features': regions,
            'total_features': len(significant_features),
            'hypervector': hv,
            'persistence_diagram': features
        }
    
    def _compute_persistence(self, gray: np.ndarray) -> List[TopologicalFeature]:
        """Compute persistence diagram using sublevel set filtration."""
        features = []
        
        # Normalize image
        normalized = (gray - np.min(gray)) / (np.max(gray) - np.min(gray))
        
        # 0-dimensional features (local minima)
        features.extend(self._find_0d_features(normalized))
        
        # 1-dimensional features (ridges/valleys)
        features.extend(self._find_1d_features(normalized))
        
        # 2-dimensional features (regions)
        features.extend(self._find_2d_features(normalized))
        
        return features
    
    def _find_0d_features(self, image: np.ndarray) -> List[TopologicalFeature]:
        """Find 0-dimensional topological features (points)."""
        features = []
        
        # Find local extrema
        # Local minima (dark spots - dust, dirt)
        local_min = ndimage.minimum_filter(image, size=5)
        minima_mask = (image == local_min) & (image < 0.5)
        minima_coords = np.column_stack(np.where(minima_mask))
        
        for coord in minima_coords:
            y, x = coord
            birth = image[y, x]
            
            # Find death time (when component merges)
            death = self._find_merge_time(image, (y, x), 'min')
            
            if death - birth > self.noise_threshold:
                features.append(TopologicalFeature(
                    dimension=0,
                    birth=birth,
                    death=death,
                    location=(x / image.shape[1], y / image.shape[0]),
                    persistence=death - birth
                ))
        
        # Local maxima (bright spots - bubbles, reflections)
        local_max = ndimage.maximum_filter(image, size=5)
        maxima_mask = (image == local_max) & (image > 0.5)
        maxima_coords = np.column_stack(np.where(maxima_mask))
        
        for coord in maxima_coords:
            y, x = coord
            birth = 1 - image[y, x]  # Invert for maxima
            
            # Find death time
            death = 1 - self._find_merge_time(image, (y, x), 'max')
            
            if death - birth > self.noise_threshold:
                features.append(TopologicalFeature(
                    dimension=0,
                    birth=birth,
                    death=death,
                    location=(x / image.shape[1], y / image.shape[0]),
                    persistence=death - birth
                ))
                
        return features
    
    def _find_1d_features(self, image: np.ndarray) -> List[TopologicalFeature]:
        """Find 1-dimensional topological features (curves)."""
        features = []
        
        # Detect edges/ridges using Canny
        edges = cv2.Canny((image * 255).astype(np.uint8), 30, 100)
        
        # Find contours (1D features)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) < 10:  # Skip small contours
                continue
                
            # Compute birth and death values along contour
            values = []
            for point in contour:
                x, y = point[0]
                values.append(image[y, x])
                
            birth = np.min(values)
            death = np.max(values)
            
            if death - birth > self.noise_threshold:
                # Compute centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    
                    features.append(TopologicalFeature(
                        dimension=1,
                        birth=birth,
                        death=death,
                        location=(cx / image.shape[1], cy / image.shape[0]),
                        persistence=death - birth
                    ))
                    
        return features
    
    def _find_2d_features(self, image: np.ndarray) -> List[TopologicalFeature]:
        """Find 2-dimensional topological features (regions)."""
        features = []
        
        # Use watershed or connected components for region detection
        # Threshold at multiple levels
        thresholds = np.linspace(0.1, 0.9, 9)
        
        for i, thresh in enumerate(thresholds[:-1]):
            # Binary mask at current threshold
            mask = image > thresh
            
            # Find connected components
            num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
            
            for label in range(1, num_labels):
                region = (labels == label)
                
                # Check if region persists to next threshold
                next_mask = image > thresholds[i + 1]
                if np.any(region & next_mask):
                    # Compute region properties
                    area = np.sum(region)
                    if area < 100:  # Skip small regions
                        continue
                        
                    y_coords, x_coords = np.where(region)
                    cx = np.mean(x_coords)
                    cy = np.mean(y_coords)
                    
                    birth = thresh
                    death = np.max(image[region])
                    
                    if death - birth > self.noise_threshold:
                        features.append(TopologicalFeature(
                            dimension=2,
                            birth=birth,
                            death=death,
                            location=(cx / image.shape[1], cy / image.shape[0]),
                            persistence=death - birth
                        ))
                        
        return features
    
    def _find_merge_time(self, image: np.ndarray, 
                        start: Tuple[int, int], 
                        mode: str) -> float:
        """Find when a component merges with another."""
        y, x = start
        value = image[y, x]
        
        # Flood fill to find merge threshold
        if mode == 'min':
            # For minima, increase threshold until component merges
            for thresh in np.linspace(value, 1.0, 100):
                mask = image <= thresh
                num_components = cv2.connectedComponents(mask.astype(np.uint8))[0]
                if num_components < self._count_initial_components(image, value, 'min'):
                    return thresh
        else:
            # For maxima, decrease threshold
            for thresh in np.linspace(value, 0.0, 100):
                mask = image >= thresh
                num_components = cv2.connectedComponents(mask.astype(np.uint8))[0]
                if num_components < self._count_initial_components(image, value, 'max'):
                    return thresh
                    
        return 1.0 if mode == 'min' else 0.0
    
    def _count_initial_components(self, image: np.ndarray, 
                                 threshold: float, 
                                 mode: str) -> int:
        """Count connected components at initial threshold."""
        if mode == 'min':
            mask = image <= threshold
        else:
            mask = image >= threshold
            
        return cv2.connectedComponents(mask.astype(np.uint8))[0]
    
    def _encode_topological_features(self, 
                                   dust: List[TopologicalFeature],
                                   scratches: List[TopologicalFeature], 
                                   regions: List[TopologicalFeature]) -> np.ndarray:
        """Encode topological features as a hypervector."""
        hv = np.zeros(self.encoder.dim)
        
        # Encode 0-dimensional features (dust/spots)
        if dust:
            dust_hv = np.zeros(self.encoder.dim)
            for feature in dust:
                # Create feature vector based on location and persistence
                loc_vec = self._location_to_vector(feature.location)
                pers_vec = self._persistence_to_vector(feature.persistence)
                feature_hv = self.encoder._bind(loc_vec, pers_vec)
                dust_hv += feature_hv
                
            dust_hv /= np.linalg.norm(dust_hv)
            hv += dust_hv * 0.4  # Weight for dust features
            
        # Encode 1-dimensional features (scratches)
        if scratches:
            scratch_hv = np.zeros(self.encoder.dim)
            for feature in scratches:
                loc_vec = self._location_to_vector(feature.location)
                pers_vec = self._persistence_to_vector(feature.persistence)
                feature_hv = self.encoder._bind(loc_vec, pers_vec)
                scratch_hv += feature_hv
                
            scratch_hv /= np.linalg.norm(scratch_hv)
            hv += scratch_hv * 0.4  # Weight for scratches
            
        # Encode 2-dimensional features (regions)
        if regions:
            region_hv = np.zeros(self.encoder.dim)
            for feature in regions:
                loc_vec = self._location_to_vector(feature.location)
                pers_vec = self._persistence_to_vector(feature.persistence)
                feature_hv = self.encoder._bind(loc_vec, pers_vec)
                region_hv += feature_hv
                
            region_hv /= np.linalg.norm(region_hv)
            hv += region_hv * 0.2  # Lower weight for regions
            
        # Normalize final hypervector
        if np.linalg.norm(hv) > 0:
            hv /= np.linalg.norm(hv)
            
        return hv
    
    def _location_to_vector(self, location: Tuple[float, float]) -> np.ndarray:
        """Convert spatial location to hypervector."""
        x, y = location
        vec = np.zeros(self.encoder.dim)
        
        # Use spatial frequencies to encode location
        for i in range(self.encoder.dim):
            freq_x = (i % 100) / 10
            freq_y = (i // 100) / 10
            vec[i] = np.sin(2 * np.pi * freq_x * x) * np.cos(2 * np.pi * freq_y * y)
            
        return vec / np.linalg.norm(vec)
    
    def _persistence_to_vector(self, persistence: float) -> np.ndarray:
        """Convert persistence value to hypervector."""
        vec = np.zeros(self.encoder.dim)
        
        # Use exponential decay to encode persistence
        for i in range(self.encoder.dim):
            vec[i] = np.exp(-i / (persistence * self.encoder.dim))
            
        return vec / np.linalg.norm(vec)
    
    def match_defect_pattern(self, 
                           image: np.ndarray,
                           reference_hv: np.ndarray,
                           threshold: float = 0.7) -> Dict[str, any]:
        """
        Match defects in image against a reference hypervector pattern.
        
        Args:
            image: Input image to analyze
            reference_hv: Reference hypervector encoding known defect pattern
            threshold: Similarity threshold for matching
            
        Returns:
            Match results with similarity score and identified features
        """
        # Analyze current image
        analysis = self.analyze_defects(image)
        current_hv = analysis['hypervector']
        
        # Compute similarity
        similarity = self.encoder.similarity(current_hv, reference_hv)
        
        # If similarity is high, identify matching features
        matches = []
        if similarity > threshold:
            # Find which features contribute most to similarity
            for feature_type in ['dust_features', 'scratch_features', 'region_features']:
                for feature in analysis[feature_type]:
                    # Create individual feature hypervector
                    loc_vec = self._location_to_vector(feature.location)
                    pers_vec = self._persistence_to_vector(feature.persistence)
                    feature_hv = self.encoder._bind(loc_vec, pers_vec)
                    
                    # Check similarity with reference
                    feature_sim = self.encoder.similarity(feature_hv, reference_hv)
                    if feature_sim > threshold * 0.8:
                        matches.append({
                            'feature': feature,
                            'type': feature_type.replace('_features', ''),
                            'similarity': feature_sim
                        })
                        
        return {
            'overall_similarity': similarity,
            'matches': matches,
            'is_match': similarity > threshold,
            'analysis': analysis
        }
