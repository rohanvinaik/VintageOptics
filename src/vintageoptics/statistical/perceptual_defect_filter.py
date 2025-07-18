"""
Perceptual Defect Filter for VintageOptics
Implements Gestalt principles to avoid treating coherent objects as defects
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage import measure, morphology
from typing import Dict, List, Tuple
import logging

class PerceptualDefectFilter:
    """
    Filters defect detections based on human visual perception principles.
    Implements Gestalt principles to avoid treating coherent objects as defects.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Thresholds based on human perception
        self.size_threshold_ratio = 0.05  # Defects > 5% of image are likely objects
        self.coherence_threshold = 0.7    # How "together" regions move
        self.texture_regularity = 0.6     # Regular patterns aren't defects
        self.min_defect_size = 3         # Minimum pixels for a defect
        self.max_defect_size_ratio = 0.01 # Max 1% of image for single defect
        
    def filter_defects(self, image: np.ndarray, raw_defects: Dict) -> Dict:
        """
        Apply perceptual filtering to raw defect detections.
        
        Args:
            image: Original image
            raw_defects: Raw defect detections from various analyzers
            
        Returns:
            Filtered defects that are actual image problems
        """
        h, w = image.shape[:2]
        total_pixels = h * w
        
        filtered_defects = {
            'dust': {'regions': [], 'count': 0, 'locations': []},
            'scratches': {'regions': [], 'count': 0, 'scratches': []},
            'fungus': {'regions': [], 'count': 0},
            'haze': raw_defects.get('haze', {'level': 0, 'severity': 0}),
            'separation': raw_defects.get('separation', {'detected': False, 'severity': 0}),
            'coating': raw_defects.get('coating', {'detected': False, 'severity': 0})
        }
        
        # Process dust defects
        if 'dust' in raw_defects:
            filtered_dust = self._filter_dust_defects(image, raw_defects['dust'], total_pixels)
            filtered_defects['dust'] = filtered_dust
            
        # Process scratch defects
        if 'scratches' in raw_defects:
            filtered_scratches = self._filter_scratch_defects(image, raw_defects['scratches'], total_pixels)
            filtered_defects['scratches'] = filtered_scratches
            
        # Process fungus defects
        if 'fungus' in raw_defects:
            filtered_fungus = self._filter_fungus_defects(image, raw_defects['fungus'], total_pixels)
            filtered_defects['fungus'] = filtered_fungus
        
        # Calculate total defect score
        total_count = (filtered_defects['dust']['count'] + 
                      filtered_defects['scratches']['count'] + 
                      filtered_defects['fungus']['count'])
        
        # Apply sanity check on total count
        if total_count > 100:
            self.logger.warning(f"Excessive defect count ({total_count}), likely false positives. Applying stricter filtering.")
            # Re-filter with stricter criteria
            filtered_defects = self._apply_stricter_filtering(image, raw_defects, filtered_defects)
            total_count = (filtered_defects['dust']['count'] + 
                          filtered_defects['scratches']['count'] + 
                          filtered_defects['fungus']['count'])
        
        filtered_defects['total_defect_score'] = min(total_count / 50, 1.0)
        
        self.logger.info(f"Filtered defects from {self._count_total(raw_defects)} to {total_count}")
        
        return filtered_defects
    
    def _filter_dust_defects(self, image: np.ndarray, dust_data: Dict, total_pixels: int) -> Dict:
        """Filter dust defects using perceptual criteria."""
        filtered_locations = []
        
        for dust in dust_data.get('locations', []):
            if self._is_actual_dust(dust, image, total_pixels):
                filtered_locations.append(dust)
        
        return {
            'locations': filtered_locations,
            'count': len(filtered_locations),
            'severity': min(len(filtered_locations) / 50, 1.0)
        }
    
    def _filter_scratch_defects(self, image: np.ndarray, scratch_data: Dict, total_pixels: int) -> Dict:
        """Filter scratch defects using perceptual criteria."""
        filtered_scratches = []
        
        for scratch in scratch_data.get('scratches', []):
            if self._is_actual_scratch(scratch, image, total_pixels):
                filtered_scratches.append(scratch)
        
        return {
            'scratches': filtered_scratches,
            'count': len(filtered_scratches),
            'severity': min(len(filtered_scratches) / 10, 1.0)
        }
    
    def _filter_fungus_defects(self, image: np.ndarray, fungus_data: Dict, total_pixels: int) -> Dict:
        """Filter fungus defects using perceptual criteria."""
        filtered_regions = []
        
        for region in fungus_data.get('regions', []):
            if self._is_actual_fungus(region, image, total_pixels):
                filtered_regions.append(region)
        
        return {
            'regions': filtered_regions,
            'count': len(filtered_regions),
            'severity': min(sum(r.get('area', 0) for r in filtered_regions) / total_pixels, 1.0)
        }
    
    def _is_actual_dust(self, dust: Dict, image: np.ndarray, total_pixels: int) -> bool:
        """
        Determine if a dust detection is actual dust or part of image content.
        """
        area = dust.get('area', 0)
        position = dust.get('position', (0, 0))
        
        # Size constraints
        if area < self.min_defect_size:
            return False  # Too small, likely noise
        
        if area > total_pixels * self.max_defect_size_ratio:
            return False  # Too large, likely part of the image
        
        # Check local context
        x, y = position
        h, w = image.shape[:2]
        
        # Define context window
        window_size = 50
        x1 = max(0, x - window_size)
        x2 = min(w, x + window_size)
        y1 = max(0, y - window_size)
        y2 = min(h, y + window_size)
        
        if x2 - x1 < 10 or y2 - y1 < 10:
            return False  # Too close to edge
        
        # Analyze local texture
        local_region = image[y1:y2, x1:x2]
        if len(local_region.shape) == 3:
            local_region = cv2.cvtColor(local_region, cv2.COLOR_BGR2GRAY)
        
        # Check if the "dust" is part of a texture pattern
        local_std = np.std(local_region)
        dust_intensity = dust.get('intensity', 0)
        local_mean = np.mean(local_region)
        
        # If the dust intensity is within 1 std dev of local mean, it's likely texture
        if abs(dust_intensity - local_mean) < local_std:
            return False
        
        # Check for regularity (Gestalt principle of similarity)
        if self._is_part_of_pattern(position, image):
            return False
        
        return True
    
    def _is_actual_scratch(self, scratch: Dict, image: np.ndarray, total_pixels: int) -> bool:
        """
        Determine if a scratch detection is actual scratch or part of image content.
        """
        length = scratch.get('length', 0)
        start = scratch.get('start', (0, 0))
        end = scratch.get('end', (0, 0))
        
        # Length constraints
        h, w = image.shape[:2]
        max_dimension = max(h, w)
        
        if length < 20:  # Too short
            return False
        
        if length > max_dimension * 0.8:  # Too long, likely an edge
            return False
        
        # Check if scratch follows image contours (likely not a defect)
        if self._follows_image_edge(start, end, image):
            return False
        
        # Check contrast along the scratch
        if not self._has_consistent_contrast(start, end, image):
            return False
        
        return True
    
    def _is_actual_fungus(self, region: Dict, image: np.ndarray, total_pixels: int) -> bool:
        """
        Determine if a fungus detection is actual fungus or part of image content.
        """
        area = region.get('area', 0)
        
        # Size constraints
        if area < 100:  # Too small for fungus
            return False
        
        if area > total_pixels * 0.05:  # Too large (>5% of image)
            return False
        
        # Check if it's part of a larger texture pattern
        bbox = region.get('bbox', None)
        if bbox is not None:
            x, y, w, h = bbox
            if self._is_regular_texture(image[y:y+h, x:x+w]):
                return False
        
        return True
    
    def _is_part_of_pattern(self, position: Tuple[int, int], image: np.ndarray) -> bool:
        """
        Check if a point is part of a regular pattern (using FFT).
        """
        x, y = position
        h, w = image.shape[:2]
        
        # Extract local region
        size = 64
        x1 = max(0, x - size//2)
        x2 = min(w, x + size//2)
        y1 = max(0, y - size//2)
        y2 = min(h, y + size//2)
        
        if x2 - x1 < size//2 or y2 - y1 < size//2:
            return False
        
        region = image[y1:y2, x1:x2]
        if len(region.shape) == 3:
            region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # FFT to detect regularity
        f_transform = np.fft.fft2(region)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Look for strong frequency peaks (indicating pattern)
        threshold = np.mean(magnitude) + 2 * np.std(magnitude)
        peaks = magnitude > threshold
        
        # If multiple peaks exist, it's likely a pattern
        peak_count = np.sum(peaks)
        return peak_count > 5
    
    def _follows_image_edge(self, start: Tuple[int, int], end: Tuple[int, int], image: np.ndarray) -> bool:
        """
        Check if a line follows an edge in the image.
        """
        # Sample points along the line
        num_samples = 10
        x_samples = np.linspace(start[0], end[0], num_samples).astype(int)
        y_samples = np.linspace(start[1], end[1], num_samples).astype(int)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150)
        
        # Check how many sample points are on edges
        edge_count = 0
        for x, y in zip(x_samples, y_samples):
            if 0 <= y < edges.shape[0] and 0 <= x < edges.shape[1]:
                # Check in a small window around the point
                window = edges[max(0, y-2):min(edges.shape[0], y+3), 
                              max(0, x-2):min(edges.shape[1], x+3)]
                if np.any(window > 0):
                    edge_count += 1
        
        # If most points are on edges, it's following an edge
        return edge_count > num_samples * 0.7
    
    def _has_consistent_contrast(self, start: Tuple[int, int], end: Tuple[int, int], image: np.ndarray) -> bool:
        """
        Check if a line has consistent contrast (actual scratch vs image feature).
        """
        # Sample perpendicular to the line
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Calculate line direction
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)
        
        if length == 0:
            return False
        
        # Normal direction
        nx = -dy / length
        ny = dx / length
        
        # Sample points along the line
        contrasts = []
        for t in np.linspace(0, 1, 10):
            px = int(start[0] + t * dx)
            py = int(start[1] + t * dy)
            
            # Sample perpendicular points
            offset = 3
            p1x = int(px + offset * nx)
            p1y = int(py + offset * ny)
            p2x = int(px - offset * nx)
            p2y = int(py - offset * ny)
            
            # Check bounds
            if (0 <= p1x < gray.shape[1] and 0 <= p1y < gray.shape[0] and
                0 <= p2x < gray.shape[1] and 0 <= p2y < gray.shape[0] and
                0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]):
                
                # Calculate contrast
                center_val = float(gray[py, px])
                side_val = (float(gray[p1y, p1x]) + float(gray[p2y, p2x])) / 2
                contrast = abs(center_val - side_val)
                contrasts.append(contrast)
        
        if not contrasts:
            return False
        
        # Check consistency
        mean_contrast = np.mean(contrasts)
        std_contrast = np.std(contrasts)
        
        # Actual scratches have consistent contrast
        return mean_contrast > 10 and std_contrast / mean_contrast < 0.5
    
    def _is_regular_texture(self, region: np.ndarray) -> bool:
        """
        Check if a region contains regular texture (not fungus).
        """
        if region.size == 0:
            return False
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        
        # Calculate texture features
        # Use Laws' texture energy measures
        l5 = np.array([1, 4, 6, 4, 1]) / 16  # Low-pass
        e5 = np.array([-1, -2, 0, 2, 1]) / 6  # Edge
        
        # Convolve with texture filters
        texture_response = cv2.sepFilter2D(gray, -1, l5, e5)
        
        # Regular textures have uniform response
        uniformity = 1 - (np.std(texture_response) / (np.mean(np.abs(texture_response)) + 1e-6))
        
        return uniformity > self.texture_regularity
    
    def _apply_stricter_filtering(self, image: np.ndarray, raw_defects: Dict, current_filtered: Dict) -> Dict:
        """
        Apply stricter filtering when too many defects are detected.
        """
        # Keep only the most significant defects
        strict_filtered = current_filtered.copy()
        
        # For dust: keep only larger, high-contrast spots
        if strict_filtered['dust']['count'] > 20:
            dust_list = strict_filtered['dust']['locations']
            # Sort by area and keep top 20
            dust_list.sort(key=lambda x: x.get('area', 0), reverse=True)
            strict_filtered['dust']['locations'] = dust_list[:20]
            strict_filtered['dust']['count'] = 20
        
        # For scratches: keep only longer, more prominent ones
        if strict_filtered['scratches']['count'] > 5:
            scratch_list = strict_filtered['scratches']['scratches']
            # Sort by length and keep top 5
            scratch_list.sort(key=lambda x: x.get('length', 0), reverse=True)
            strict_filtered['scratches']['scratches'] = scratch_list[:5]
            strict_filtered['scratches']['count'] = 5
        
        # For fungus: keep only larger regions
        if strict_filtered['fungus']['count'] > 3:
            fungus_list = strict_filtered['fungus']['regions']
            # Sort by area and keep top 3
            fungus_list.sort(key=lambda x: x.get('area', 0), reverse=True)
            strict_filtered['fungus']['regions'] = fungus_list[:3]
            strict_filtered['fungus']['count'] = 3
        
        return strict_filtered
    
    def _count_total(self, defects: Dict) -> int:
        """Count total defects in raw detection."""
        total = 0
        if 'dust' in defects:
            total += defects['dust'].get('count', 0)
        if 'scratches' in defects:
            total += defects['scratches'].get('count', 0)
        if 'fungus' in defects:
            total += defects['fungus'].get('count', 0)
        return total
