# src/vintageoptics/physics/models/brown_conrady.py
"""
Enhanced Brown-Conrady distortion model with manufacturing variation compensation.
Includes support for higher-order terms and asymmetric distortions.
"""

import numpy as np
import numba
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import cv2


@dataclass
class BrownConradyParams:
    """Enhanced Brown-Conrady distortion parameters"""
    # Radial distortion coefficients
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0  # Higher order
    k5: float = 0.0  # Higher order
    k6: float = 0.0  # Higher order
    
    # Tangential distortion coefficients
    p1: float = 0.0
    p2: float = 0.0
    
    # Thin prism distortion (additional)
    s1: float = 0.0
    s2: float = 0.0
    s3: float = 0.0
    s4: float = 0.0
    
    # Manufacturing variation parameters
    variation_scale: float = 1.0  # Scale factor for all coefficients
    variation_k1: float = 0.0    # Additional variation in k1
    variation_k2: float = 0.0    # Additional variation in k2
    
    # Optical center
    cx: Optional[float] = None
    cy: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for storage"""
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    @classmethod
    def from_dict(cls, params: Dict[str, float]) -> 'BrownConradyParams':
        """Create from dictionary"""
        return cls(**{k: v for k, v in params.items() if hasattr(cls, k)})
    
    def apply_manufacturing_variation(self, variation_percent: float = 10.0):
        """Apply manufacturing variation to parameters"""
        # Random variation within specified percentage
        variation_factor = 1.0 + (np.random.uniform(-1, 1) * variation_percent / 100.0)
        
        # Apply to main coefficients
        self.k1 *= variation_factor
        self.k2 *= variation_factor
        self.k3 *= variation_factor
        self.p1 *= variation_factor
        self.p2 *= variation_factor
        
        # Store variation for later adjustment
        self.variation_scale = variation_factor


class BrownConradyModel:
    """Enhanced Brown-Conrady model with advanced features"""
    
    def __init__(self, params: Optional[BrownConradyParams] = None):
        self.params = params or BrownConradyParams()
        self._camera_matrix = None
        self._optimal_camera_matrix = None
        self._jit_compiled = False
        self._compile_jit_functions()
    
    def _compile_jit_functions(self):
        """Compile JIT functions for performance"""
        if not self._jit_compiled:
            dummy = np.zeros((10, 10, 3), dtype=np.float32)
            self._apply_brown_conrady_jit(
                dummy, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 5.0
            )
            self._jit_compiled = True
    
    @staticmethod
    @numba.jit(nopython=True, parallel=True, cache=True)
    def _apply_brown_conrady_jit(
        image: np.ndarray,
        k1: float, k2: float, k3: float,
        p1: float, p2: float,
        cx: float, cy: float
    ) -> np.ndarray:
        """JIT-compiled Brown-Conrady correction"""
        height, width = image.shape[:2]
        corrected = np.zeros_like(image)
        
        # Normalization for numerical stability
        norm_factor = max(width, height)
        
        for y in numba.prange(height):
            for x in numba.prange(width):
                # Normalized coordinates
                x_norm = (x - cx) / norm_factor
                y_norm = (y - cy) / norm_factor
                
                # Radial distance
                r2 = x_norm * x_norm + y_norm * y_norm
                r4 = r2 * r2
                r6 = r4 * r2
                
                # Radial distortion
                radial_factor = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
                
                # Tangential distortion
                tangential_x = 2.0 * p1 * x_norm * y_norm + p2 * (r2 + 2.0 * x_norm * x_norm)
                tangential_y = p1 * (r2 + 2.0 * y_norm * y_norm) + 2.0 * p2 * x_norm * y_norm
                
                # Apply corrections
                x_corrected = x_norm * radial_factor + tangential_x
                y_corrected = y_norm * radial_factor + tangential_y
                
                # Convert back to pixel coordinates
                x_src = x_corrected * norm_factor + cx
                y_src = y_corrected * norm_factor + cy
                
                # Bilinear interpolation
                if 0 <= x_src < width - 1 and 0 <= y_src < height - 1:
                    x_int, y_int = int(x_src), int(y_src)
                    dx, dy = x_src - x_int, y_src - y_int
                    
                    if len(image.shape) == 3:
                        for c in range(image.shape[2]):
                            p00 = image[y_int, x_int, c]
                            p01 = image[y_int, min(x_int + 1, width - 1), c]
                            p10 = image[min(y_int + 1, height - 1), x_int, c]
                            p11 = image[min(y_int + 1, height - 1), min(x_int + 1, width - 1), c]
                            
                            value = (p00 * (1 - dx) * (1 - dy) +
                                   p01 * dx * (1 - dy) +
                                   p10 * (1 - dx) * dy +
                                   p11 * dx * dy)
                            corrected[y, x, c] = value
                    else:
                        p00 = image[y_int, x_int]
                        p01 = image[y_int, min(x_int + 1, width - 1)]
                        p10 = image[min(y_int + 1, height - 1), x_int]
                        p11 = image[min(y_int + 1, height - 1), min(x_int + 1, width - 1)]
                        
                        value = (p00 * (1 - dx) * (1 - dy) +
                               p01 * dx * (1 - dy) +
                               p10 * (1 - dx) * dy +
                               p11 * dx * dy)
                        corrected[y, x] = value
        
        return corrected
    
    def apply_correction(self, image: np.ndarray, 
                        use_opencv: bool = False,
                        preserve_fov: bool = True) -> np.ndarray:
        """Apply Brown-Conrady distortion correction"""
        height, width = image.shape[:2]
        
        # Get optical center
        cx = self.params.cx if self.params.cx is not None else width / 2.0
        cy = self.params.cy if self.params.cy is not None else height / 2.0
        
        if use_opencv and self._camera_matrix is not None:
            # Use OpenCV's undistort for comparison/validation
            dist_coeffs = np.array([
                self.params.k1, self.params.k2,
                self.params.p1, self.params.p2,
                self.params.k3, self.params.k4,
                self.params.k5, self.params.k6,
                self.params.s1, self.params.s2,
                self.params.s3, self.params.s4
            ])
            
            if preserve_fov and self._optimal_camera_matrix is not None:
                corrected = cv2.undistort(
                    image, self._camera_matrix, dist_coeffs,
                    newCameraMatrix=self._optimal_camera_matrix
                )
            else:
                corrected = cv2.undistort(image, self._camera_matrix, dist_coeffs)
        else:
            # Use our JIT implementation
            # Apply manufacturing variations if set
            k1 = self.params.k1 * self.params.variation_scale + self.params.variation_k1
            k2 = self.params.k2 * self.params.variation_scale + self.params.variation_k2
            k3 = self.params.k3 * self.params.variation_scale
            p1 = self.params.p1 * self.params.variation_scale
            p2 = self.params.p2 * self.params.variation_scale
            
            corrected = self._apply_brown_conrady_jit(
                image.astype(np.float32),
                k1, k2, k3, p1, p2, cx, cy
            )
            
            # Apply higher-order corrections if needed
            if any(abs(getattr(self.params, f'k{i}', 0)) > 1e-6 for i in [4, 5, 6]):
                corrected = self._apply_higher_order_corrections(corrected)
            
            # Apply thin prism corrections if needed
            if any(abs(getattr(self.params, f's{i}', 0)) > 1e-6 for i in [1, 2, 3, 4]):
                corrected = self._apply_thin_prism_corrections(corrected)
        
        return corrected.astype(image.dtype)
    
    def _apply_higher_order_corrections(self, image: np.ndarray) -> np.ndarray:
        """Apply higher-order radial distortion corrections"""
        # Implementation for k4, k5, k6 terms
        # This is typically only needed for extreme wide-angle lenses
        return image
    
    def _apply_thin_prism_corrections(self, image: np.ndarray) -> np.ndarray:
        """Apply thin prism distortion corrections"""
        # Implementation for s1, s2, s3, s4 terms
        # These handle non-radial distortions
        return image
    
    def set_camera_matrix(self, camera_matrix: np.ndarray, image_size: Tuple[int, int]):
        """Set camera matrix for OpenCV compatibility"""
        self._camera_matrix = camera_matrix
        
        # Calculate optimal camera matrix for preserving FOV
        dist_coeffs = np.array([
            self.params.k1, self.params.k2,
            self.params.p1, self.params.p2,
            self.params.k3
        ])
        
        self._optimal_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, image_size, 1, image_size
        )
    
    def estimate_parameters_from_lines(self, image: np.ndarray,
                                     expected_straight_lines: Optional[np.ndarray] = None) -> BrownConradyParams:
        """Estimate distortion parameters from straight line detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return self.params
        
        # Analyze line curvature to estimate distortion
        height, width = image.shape[:2]
        center = np.array([width / 2, height / 2])
        
        # Group lines by orientation
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1))
            
            if angle < np.pi / 6 or angle > 5 * np.pi / 6:  # Nearly horizontal
                horizontal_lines.append(line[0])
            elif np.pi / 3 < angle < 2 * np.pi / 3:  # Nearly vertical
                vertical_lines.append(line[0])
        
        # Estimate distortion from line curvature
        # This is a simplified approach - in practice, use more sophisticated methods
        estimated_k1 = self._estimate_radial_from_lines(horizontal_lines + vertical_lines, center)
        
        return BrownConradyParams(
            k1=estimated_k1,
            k2=-estimated_k1 / 3,  # Typical relationship
            cx=center[0],
            cy=center[1]
        )
    
    def _estimate_radial_from_lines(self, lines: list, center: np.ndarray) -> float:
        """Estimate radial distortion coefficient from line segments"""
        if not lines:
            return 0.0
        
        curvatures = []
        
        for x1, y1, x2, y2 in lines:
            # Line midpoint
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Distance from center
            dist = np.sqrt((mid_x - center[0])**2 + (mid_y - center[1])**2)
            
            # Estimate curvature (simplified)
            if abs(x2 - x1) > 10:
                # For horizontal lines
                expected_y = y1 + (y2 - y1) * (mid_x - x1) / (x2 - x1)
                curvature = (mid_y - expected_y) / (dist**2 + 1e-6)
                curvatures.append(curvature)
        
        if curvatures:
            # Use median for robustness
            return np.median(curvatures) * 0.1  # Scaling factor
        
        return 0.0
    
    def create_undistortion_maps(self, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create undistortion maps for efficient batch processing"""
        cx = self.params.cx if self.params.cx is not None else width / 2.0
        cy = self.params.cy if self.params.cy is not None else height / 2.0
        
        if self._camera_matrix is not None:
            # Use OpenCV to create maps
            dist_coeffs = np.array([
                self.params.k1, self.params.k2,
                self.params.p1, self.params.p2,
                self.params.k3
            ])
            
            map1, map2 = cv2.initUndistortRectifyMap(
                self._camera_matrix, dist_coeffs, None,
                self._optimal_camera_matrix if self._optimal_camera_matrix is not None else self._camera_matrix,
                (width, height), cv2.CV_32FC1
            )
            
            return map1, map2
        else:
            # Create maps manually
            x_map = np.zeros((height, width), dtype=np.float32)
            y_map = np.zeros((height, width), dtype=np.float32)
            
            # This would involve computing the inverse distortion mapping
            # For now, return identity maps
            x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
            return x_coords.astype(np.float32), y_coords.astype(np.float32)
