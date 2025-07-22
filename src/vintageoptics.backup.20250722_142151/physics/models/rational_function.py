# src/vintageoptics/physics/models/rational_function.py
"""
Enhanced Rational Function Distortion Model with GPU acceleration support.
Implements high-precision correction for complex distortions found in vintage lenses.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict
from dataclasses import dataclass, field
import logging

try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage
    CUPY_AVAILABLE = True
except ImportError:
    cp = np
    CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RationalFunctionParams:
    """Parameters for rational function distortion model"""
    # Numerator coefficients
    a0: float = 1.0
    a1: float = 0.0
    a2: float = 0.0
    a3: float = 0.0
    a4: float = 0.0  # Extended for higher order
    
    # Denominator coefficients  
    b0: float = 1.0
    b1: float = 0.0
    b2: float = 0.0
    b3: float = 0.0
    b4: float = 0.0  # Extended for higher order
    
    # Center point (None = image center)
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    
    # Manufacturing variations
    variation_seed: Optional[int] = None
    variation_percent: float = 0.0
    
    def apply_manufacturing_variation(self, percent: float = 10.0):
        """Apply realistic manufacturing variations"""
        if self.variation_seed:
            np.random.seed(self.variation_seed)
        
        # Apply variations with physical constraints
        variation = 1 + (np.random.randn() * percent / 100)
        
        # Higher order terms have more variation
        self.a1 *= variation
        self.a2 *= variation * 1.1
        self.a3 *= variation * 1.2
        self.a4 *= variation * 1.3
        
        self.b1 *= variation
        self.b2 *= variation * 1.1
        self.b3 *= variation * 1.2
        self.b4 *= variation * 1.3


class RationalFunctionDistortion:
    """
    Rational function distortion model for complex optical aberrations.
    Suitable for lenses with non-polynomial distortion patterns.
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        if self.use_gpu:
            logger.info("GPU acceleration enabled for rational function model")
        
        self.params: Optional[RationalFunctionParams] = None
        self._cache = {}
    
    def apply_correction(self, image: np.ndarray, 
                        params: Optional[RationalFunctionParams] = None) -> np.ndarray:
        """Apply rational function distortion correction"""
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
        
        # Use GPU if available and beneficial
        if self.use_gpu and w * h > 1000000:  # Use GPU for images > 1MP
            return self._apply_correction_gpu(image)
        else:
            return self._apply_correction_cpu(image)
    
    def _apply_correction_cpu(self, image: np.ndarray) -> np.ndarray:
        """CPU implementation of rational function correction"""
        h, w = image.shape[:2]
        cx, cy = self.params.center_x, self.params.center_y
        
        # Generate or retrieve cached mapping
        cache_key = (w, h, id(self.params))
        if cache_key in self._cache:
            map_x, map_y = self._cache[cache_key]
        else:
            map_x, map_y = self._generate_mapping_cpu(w, h, cx, cy)
            self._cache[cache_key] = (map_x, map_y)
        
        # Apply remapping
        return cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC, 
                        borderMode=cv2.BORDER_REFLECT)
    
    def _apply_correction_gpu(self, image: np.ndarray) -> np.ndarray:
        """GPU implementation using CuPy"""
        h, w = image.shape[:2]
        cx, cy = self.params.center_x, self.params.center_y
        
        # Transfer to GPU
        gpu_image = cp.asarray(image)
        
        # Generate mapping on GPU
        map_x, map_y = self._generate_mapping_gpu(w, h, cx, cy)
        
        # Apply correction channel by channel (CuPy doesn't have cv2.remap equivalent)
        corrected = cp.zeros_like(gpu_image)
        
        if len(image.shape) == 3:
            for c in range(image.shape[2]):
                corrected[:, :, c] = cp_ndimage.map_coordinates(
                    gpu_image[:, :, c], 
                    [map_y.ravel(), map_x.ravel()],
                    order=3,  # Cubic interpolation
                    mode='reflect'
                ).reshape(h, w)
        else:
            corrected = cp_ndimage.map_coordinates(
                gpu_image,
                [map_y.ravel(), map_x.ravel()],
                order=3,
                mode='reflect'
            ).reshape(h, w)
        
        # Transfer back to CPU
        return cp.asnumpy(corrected).astype(image.dtype)
    
    def _generate_mapping_cpu(self, w: int, h: int, cx: float, cy: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate distortion mapping on CPU"""
        # Create coordinate grids
        x_grid, y_grid = np.meshgrid(np.arange(w, dtype=np.float32),
                                     np.arange(h, dtype=np.float32))
        
        # Normalize coordinates
        x_norm = (x_grid - cx) / cx
        y_norm = (y_grid - cy) / cy
        
        # Calculate radius squared
        r2 = x_norm**2 + y_norm**2
        r4 = r2**2
        r6 = r4 * r2
        r8 = r4**2
        
        # Rational function distortion
        numerator = (self.params.a0 + 
                    self.params.a1 * r2 + 
                    self.params.a2 * r4 + 
                    self.params.a3 * r6 +
                    self.params.a4 * r8)
        
        denominator = (self.params.b0 + 
                      self.params.b1 * r2 + 
                      self.params.b2 * r4 + 
                      self.params.b3 * r6 +
                      self.params.b4 * r8)
        
        # Prevent division by zero
        denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
        
        # Apply distortion
        distortion_factor = numerator / denominator
        
        # Calculate source coordinates
        x_src = x_norm * distortion_factor * cx + cx
        y_src = y_norm * distortion_factor * cy + cy
        
        return x_src.astype(np.float32), y_src.astype(np.float32)
    
    def _generate_mapping_gpu(self, w: int, h: int, cx: float, cy: float) -> Tuple[cp.ndarray, cp.ndarray]:
        """Generate distortion mapping on GPU"""
        # Create coordinate grids on GPU
        x_grid, y_grid = cp.meshgrid(cp.arange(w, dtype=cp.float32),
                                     cp.arange(h, dtype=cp.float32))
        
        # Normalize coordinates
        x_norm = (x_grid - cx) / cx
        y_norm = (y_grid - cy) / cy
        
        # Calculate radius squared
        r2 = x_norm**2 + y_norm**2
        r4 = r2**2
        r6 = r4 * r2
        r8 = r4**2
        
        # Rational function distortion
        numerator = (self.params.a0 + 
                    self.params.a1 * r2 + 
                    self.params.a2 * r4 + 
                    self.params.a3 * r6 +
                    self.params.a4 * r8)
        
        denominator = (self.params.b0 + 
                      self.params.b1 * r2 + 
                      self.params.b2 * r4 + 
                      self.params.b3 * r6 +
                      self.params.b4 * r8)
        
        # Prevent division by zero
        denominator = cp.where(cp.abs(denominator) < 1e-10, 1e-10, denominator)
        
        # Apply distortion
        distortion_factor = numerator / denominator
        
        # Calculate source coordinates
        x_src = x_norm * distortion_factor * cx + cx
        y_src = y_norm * distortion_factor * cy + cy
        
        return x_src, y_src
    
    def estimate_from_points(self, distorted_points: np.ndarray, 
                           undistorted_points: np.ndarray) -> RationalFunctionParams:
        """Estimate rational function parameters from point correspondences"""
        # This is a complex optimization problem
        # For now, we'll use a simplified approach
        
        # Normalize points
        cx = np.mean(undistorted_points[:, 0])
        cy = np.mean(undistorted_points[:, 1])
        
        norm_distorted = distorted_points - np.array([cx, cy])
        norm_undistorted = undistorted_points - np.array([cx, cy])
        
        # Calculate radii
        r_distorted = np.sqrt(np.sum(norm_distorted**2, axis=1))
        r_undistorted = np.sqrt(np.sum(norm_undistorted**2, axis=1))
        
        # Simple estimation (in practice, use optimization)
        distortion_ratios = r_distorted / (r_undistorted + 1e-10)
        
        # Fit polynomial to distortion ratios
        r_norm = r_undistorted / np.max(r_undistorted)
        coeffs = np.polyfit(r_norm**2, distortion_ratios, 3)
        
        # Convert to rational function parameters
        params = RationalFunctionParams(
            a0=1.0,
            a1=coeffs[2],
            a2=coeffs[1],
            a3=coeffs[0],
            b0=1.0,
            center_x=cx,
            center_y=cy
        )
        
        return params
    
    def get_inverse_mapping(self, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get inverse mapping for undistortion"""
        # For rational functions, we need to solve the inverse numerically
        # This is computationally expensive, so we use approximation
        
        if self.params is None:
            raise ValueError("No parameters set")
        
        cx = self.params.center_x or w / 2.0
        cy = self.params.center_y or h / 2.0
        
        # Generate forward mapping
        map_x_fwd, map_y_fwd = self._generate_mapping_cpu(w, h, cx, cy)
        
        # Approximate inverse using scattered interpolation
        # Sample points from forward mapping
        sample_step = max(1, min(w, h) // 50)
        y_samples, x_samples = np.mgrid[0:h:sample_step, 0:w:sample_step]
        
        # Get corresponding source points
        src_x = map_x_fwd[y_samples, x_samples].ravel()
        src_y = map_y_fwd[y_samples, x_samples].ravel()
        dst_x = x_samples.ravel()
        dst_y = y_samples.ravel()
        
        # Create inverse mapping using griddata
        from scipy.interpolate import griddata
        
        x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
        points = np.column_stack((src_x, src_y))
        values_x = dst_x
        values_y = dst_y
        
        map_x_inv = griddata(points, values_x, (x_grid, y_grid), 
                            method='cubic', fill_value=0).astype(np.float32)
        map_y_inv = griddata(points, values_y, (x_grid, y_grid), 
                            method='cubic', fill_value=0).astype(np.float32)
        
        return map_x_inv, map_y_inv
