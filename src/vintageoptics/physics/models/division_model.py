# src/vintageoptics/physics/models/division_model.py
"""
Division Model for distortion correction.
Efficient for moderate symmetric distortions common in vintage lenses.
"""

import numpy as np
import cv2
from typing import Optional, Tuple
from dataclasses import dataclass
import logging

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = np
    CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DivisionModelParams:
    """Parameters for division distortion model"""
    lambda_param: float = 0.0  # Main distortion parameter
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    
    # Manufacturing variations
    variation_percent: float = 0.0
    
    def apply_manufacturing_variation(self, percent: float = 10.0):
        """Apply manufacturing variations"""
        variation = 1 + (np.random.randn() * percent / 100)
        self.lambda_param *= variation


class DivisionModel:
    """
    Division model for distortion correction.
    Simpler than rational function but effective for many vintage lenses.
    
    Model: r_u = r_d / (1 + λ * r_d²)
    where r_u is undistorted radius, r_d is distorted radius
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.params: Optional[DivisionModelParams] = None
        self._mapping_cache = {}
    
    def apply_correction(self, image: np.ndarray,
                        params: Optional[DivisionModelParams] = None) -> np.ndarray:
        """Apply division model distortion correction"""
        if params:
            self.params = params
        
        if self.params is None:
            raise ValueError("No parameters set for correction")
        
        h, w = image.shape[:2]
        
        # Set center if not specified
        if self.params.center_x is None:
            self.params.center_x = w / 2.0
        if self.params.center_y is None:
            self.params.center_y = h / 2.0
        
        # Skip if no distortion
        if abs(self.params.lambda_param) < 1e-8:
            return image
        
        # Use GPU for large images
        if self.use_gpu and w * h > 500000:
            return self._apply_correction_gpu(image)
        else:
            return self._apply_correction_cpu(image)
    
    def _apply_correction_cpu(self, image: np.ndarray) -> np.ndarray:
        """CPU implementation"""
        h, w = image.shape[:2]
        cx, cy = self.params.center_x, self.params.center_y
        
        # Check cache
        cache_key = (w, h, cx, cy, self.params.lambda_param)
        if cache_key in self._mapping_cache:
            map_x, map_y = self._mapping_cache[cache_key]
        else:
            # Generate mapping
            map_x, map_y = self._generate_inverse_mapping_cpu(w, h, cx, cy)
            self._mapping_cache[cache_key] = (map_x, map_y)
        
        # Apply remapping
        return cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REFLECT)
    
    def _apply_correction_gpu(self, image: np.ndarray) -> np.ndarray:
        """GPU implementation"""
        h, w = image.shape[:2]
        cx, cy = self.params.center_x, self.params.center_y
        
        # Transfer to GPU
        gpu_image = cp.asarray(image)
        
        # Generate mapping
        map_x, map_y = self._generate_inverse_mapping_gpu(w, h, cx, cy)
        
        # Apply correction
        corrected = self._remap_gpu(gpu_image, map_x, map_y)
        
        # Transfer back
        return cp.asnumpy(corrected).astype(image.dtype)
    
    def _generate_inverse_mapping_cpu(self, w: int, h: int, 
                                    cx: float, cy: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate inverse mapping for undistortion on CPU"""
        # Create coordinate grids
        x_grid, y_grid = np.meshgrid(np.arange(w, dtype=np.float32),
                                     np.arange(h, dtype=np.float32))
        
        # Normalize coordinates
        x_norm = (x_grid - cx) / cx
        y_norm = (y_grid - cy) / cy
        
        # Undistorted radius
        r_u = np.sqrt(x_norm**2 + y_norm**2)
        
        # Solve for distorted radius using Newton's method
        r_d = self._solve_distorted_radius_vectorized(r_u, self.params.lambda_param)
        
        # Scale factor
        scale = np.where(r_u > 1e-6, r_d / r_u, 1.0)
        
        # Calculate source coordinates
        x_src = (x_grid - cx) * scale + cx
        y_src = (y_grid - cy) * scale + cy
        
        return x_src, y_src
    
    def _generate_inverse_mapping_gpu(self, w: int, h: int,
                                    cx: float, cy: float) -> Tuple[cp.ndarray, cp.ndarray]:
        """Generate inverse mapping on GPU"""
        # Create coordinate grids
        x_grid, y_grid = cp.meshgrid(cp.arange(w, dtype=cp.float32),
                                     cp.arange(h, dtype=cp.float32))
        
        # Normalize coordinates
        x_norm = (x_grid - cx) / cx
        y_norm = (y_grid - cy) / cy
        
        # Undistorted radius
        r_u = cp.sqrt(x_norm**2 + y_norm**2)
        
        # Solve for distorted radius
        r_d = self._solve_distorted_radius_gpu(r_u, self.params.lambda_param)
        
        # Scale factor
        scale = cp.where(r_u > 1e-6, r_d / r_u, 1.0)
        
        # Calculate source coordinates
        x_src = (x_grid - cx) * scale + cx
        y_src = (y_grid - cy) * scale + cy
        
        return x_src, y_src
    
    def _solve_distorted_radius_vectorized(self, r_u: np.ndarray, 
                                         lambda_param: float) -> np.ndarray:
        """Solve r_d from r_u = r_d / (1 + λ * r_d²) using Newton's method"""
        # Initial guess
        r_d = r_u.copy()
        
        # Newton iterations
        for _ in range(5):  # Usually converges in 3-4 iterations
            f = r_d / (1 + lambda_param * r_d**2) - r_u
            f_prime = (1 - lambda_param * r_d**2) / (1 + lambda_param * r_d**2)**2
            
            # Avoid division by zero
            f_prime = np.where(np.abs(f_prime) > 1e-10, f_prime, 1e-10)
            
            # Newton update
            r_d = r_d - f / f_prime
            
            # Ensure positive radius
            r_d = np.maximum(r_d, 0)
        
        return r_d
    
    def _solve_distorted_radius_gpu(self, r_u: cp.ndarray, 
                                  lambda_param: float) -> cp.ndarray:
        """GPU version of radius solver"""
        r_d = r_u.copy()
        
        for _ in range(5):
            f = r_d / (1 + lambda_param * r_d**2) - r_u
            f_prime = (1 - lambda_param * r_d**2) / (1 + lambda_param * r_d**2)**2
            f_prime = cp.where(cp.abs(f_prime) > 1e-10, f_prime, 1e-10)
            r_d = r_d - f / f_prime
            r_d = cp.maximum(r_d, 0)
        
        return r_d
    
    def _remap_gpu(self, image: cp.ndarray, map_x: cp.ndarray, 
                  map_y: cp.ndarray) -> cp.ndarray:
        """GPU remapping using bilinear interpolation"""
        h, w = image.shape[:2]
        
        # Ensure coordinates are within bounds
        map_x = cp.clip(map_x, 0, w - 1)
        map_y = cp.clip(map_y, 0, h - 1)
        
        # Get integer and fractional parts
        x0 = cp.floor(map_x).astype(cp.int32)
        y0 = cp.floor(map_y).astype(cp.int32)
        x1 = cp.minimum(x0 + 1, w - 1)
        y1 = cp.minimum(y0 + 1, h - 1)
        
        # Fractional parts
        fx = map_x - x0
        fy = map_y - y0
        
        # Bilinear interpolation
        if len(image.shape) == 3:
            result = cp.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = (
                    image[y0, x0, c] * (1 - fx) * (1 - fy) +
                    image[y0, x1, c] * fx * (1 - fy) +
                    image[y1, x0, c] * (1 - fx) * fy +
                    image[y1, x1, c] * fx * fy
                )
        else:
            result = (
                image[y0, x0] * (1 - fx) * (1 - fy) +
                image[y0, x1] * fx * (1 - fy) +
                image[y1, x0] * (1 - fx) * fy +
                image[y1, x1] * fx * fy
            )
        
        return result
    
    def estimate_from_lines(self, image: np.ndarray, 
                          expected_straight: bool = True) -> DivisionModelParams:
        """Estimate division model parameters from detected lines"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        cx, cy = w / 2, h / 2
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                               minLineLength=50, maxLineGap=10)
        
        if lines is None or len(lines) < 5:
            logger.warning("Insufficient lines detected for parameter estimation")
            return DivisionModelParams(lambda_param=0.0, center_x=cx, center_y=cy)
        
        # Analyze line curvature
        curvatures = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Line midpoint
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Distance from center
            r = np.sqrt((mid_x - cx)**2 + (mid_y - cy)**2) / min(w, h)
            
            # Estimate curvature (simplified)
            if abs(x2 - x1) > 10:
                # Check if line should be vertical/horizontal
                angle = np.arctan2(y2 - y1, x2 - x1)
                expected_angle = np.arctan2(mid_y - cy, mid_x - cx)
                
                # Angular deviation as proxy for curvature
                deviation = np.sin(angle - expected_angle)
                curvatures.append((r, deviation))
        
        if not curvatures:
            return DivisionModelParams(lambda_param=0.0, center_x=cx, center_y=cy)
        
        # Fit simple model
        radii = np.array([c[0] for c in curvatures])
        deviations = np.array([c[1] for c in curvatures])
        
        # Estimate lambda using least squares
        # Simplified: assume deviation proportional to lambda * r²
        if len(radii) > 3:
            lambda_estimate = np.sum(deviations * radii**2) / np.sum(radii**4)
        else:
            lambda_estimate = 0.0
        
        return DivisionModelParams(
            lambda_param=float(lambda_estimate),
            center_x=cx,
            center_y=cy
        )
