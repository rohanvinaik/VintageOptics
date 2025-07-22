# src/vintageoptics/calibration/methods/opencv_chess.py
"""
OpenCV-based lens calibration using chessboard patterns.
Implements comprehensive calibration with manufacturing variation support.
"""

import cv2
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Complete calibration result with metadata"""
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    reprojection_error: float
    rvecs: List[np.ndarray]  # Rotation vectors
    tvecs: List[np.ndarray]  # Translation vectors
    image_points: List[np.ndarray]
    object_points: List[np.ndarray]
    calibration_date: str
    num_images: int
    image_paths: List[str]
    chessboard_size: Tuple[int, int]
    square_size: float
    flags_used: int
    optimal_camera_matrix: Optional[np.ndarray] = None
    roi: Optional[Tuple[int, int, int, int]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'camera_matrix': self.camera_matrix.tolist(),
            'distortion_coefficients': self.distortion_coefficients.tolist(),
            'reprojection_error': self.reprojection_error,
            'calibration_date': self.calibration_date,
            'num_images': self.num_images,
            'image_paths': self.image_paths,
            'chessboard_size': self.chessboard_size,
            'square_size': self.square_size,
            'flags_used': self.flags_used,
            'optimal_camera_matrix': self.optimal_camera_matrix.tolist() if self.optimal_camera_matrix is not None else None,
            'roi': self.roi
        }


class OpenCVChessboardCalibrator:
    """Advanced OpenCV-based calibration using chessboard patterns"""
    
    def __init__(self, 
                 chessboard_size: Tuple[int, int] = (9, 6),
                 square_size: float = 25.0,  # millimeters
                 enable_refinement: bool = True):
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.enable_refinement = enable_refinement
        
        # Termination criteria for corner refinement
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Corner refinement window
        self.refinement_window = (11, 11)
        
    def calibrate_from_images(self, 
                            image_paths: List[str],
                            show_progress: bool = True,
                            save_debug_images: bool = False,
                            debug_dir: str = "calibration_debug") -> CalibrationResult:
        """Perform comprehensive calibration from image files"""
        
        if save_debug_images:
            os.makedirs(debug_dir, exist_ok=True)
        
        # Prepare calibration pattern points
        pattern_points = self._create_pattern_points()
        
        # Collect calibration data
        object_points = []  # 3D points in real world space
        image_points = []   # 2D points in image plane
        valid_images = []
        image_size = None
        
        for idx, image_path in enumerate(image_paths):
            if show_progress:
                print(f"Processing image {idx + 1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"Could not load image: {image_path}")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if image_size is None:
                image_size = gray.shape[::-1]
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            
            if ret:
                # Refine corner positions
                if self.enable_refinement:
                    corners = cv2.cornerSubPix(
                        gray, corners, self.refinement_window, (-1, -1), self.criteria
                    )
                
                object_points.append(pattern_points)
                image_points.append(corners)
                valid_images.append(image_path)
                
                # Save debug image if requested
                if save_debug_images:
                    debug_img = img.copy()
                    cv2.drawChessboardCorners(debug_img, self.chessboard_size, corners, ret)
                    debug_path = os.path.join(debug_dir, f"corners_{idx:03d}.jpg")
                    cv2.imwrite(debug_path, debug_img)
            else:
                logger.warning(f"Could not find chessboard in: {image_path}")
        
        if len(object_points) < 3:
            raise ValueError(f"Insufficient valid calibration images: {len(object_points)}/3 minimum")
        
        print(f"Using {len(object_points)} valid images for calibration")
        
        # Perform calibration
        calibration_flags = self._get_calibration_flags()
        
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, image_size, None, None, flags=calibration_flags
        )
        
        # Calculate reprojection error
        reprojection_error = self._calculate_reprojection_error(
            object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs
        )
        
        # Get optimal camera matrix
        optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, image_size, 1, image_size
        )
        
        return CalibrationResult(
            camera_matrix=camera_matrix,
            distortion_coefficients=dist_coeffs,
            reprojection_error=reprojection_error,
            rvecs=rvecs,
            tvecs=tvecs,
            image_points=image_points,
            object_points=object_points,
            calibration_date=datetime.now().isoformat(),
            num_images=len(valid_images),
            image_paths=valid_images,
            chessboard_size=self.chessboard_size,
            square_size=self.square_size,
            flags_used=calibration_flags,
            optimal_camera_matrix=optimal_camera_matrix,
            roi=roi
        )
    
    def calibrate_with_manufacturing_variation(self,
                                             base_calibration: CalibrationResult,
                                             variation_images: List[str],
                                             variation_percent: float = 10.0) -> Dict[str, CalibrationResult]:
        """Calibrate lens copy with manufacturing variations"""
        
        # Use base calibration as initial guess
        initial_matrix = base_calibration.camera_matrix
        initial_dist = base_calibration.distortion_coefficients
        
        # Calibrate with constraints
        variation_result = self.calibrate_from_images(
            variation_images,
            show_progress=True
        )
        
        # Calculate parameter variations
        matrix_variation = np.abs(variation_result.camera_matrix - initial_matrix) / (initial_matrix + 1e-10)
        dist_variation = np.abs(variation_result.distortion_coefficients - initial_dist) / (np.abs(initial_dist) + 1e-10)
        
        # Check if variations are within expected manufacturing tolerance
        max_matrix_variation = np.max(matrix_variation) * 100
        max_dist_variation = np.max(dist_variation) * 100
        
        logger.info(f"Maximum camera matrix variation: {max_matrix_variation:.2f}%")
        logger.info(f"Maximum distortion coefficient variation: {max_dist_variation:.2f}%")
        
        if max_dist_variation > variation_percent * 2:
            logger.warning(f"Large variation detected ({max_dist_variation:.2f}%), "
                         "this may indicate a different lens model")
        
        return {
            'base': base_calibration,
            'instance': variation_result,
            'variations': {
                'camera_matrix_variation_percent': matrix_variation * 100,
                'distortion_variation_percent': dist_variation * 100,
                'max_variation_percent': max(max_matrix_variation, max_dist_variation)
            }
        }
    
    def generate_calibration_pattern(self, 
                                   output_path: str,
                                   pattern_size_mm: Tuple[int, int] = (297, 210),  # A4
                                   dpi: int = 300) -> str:
        """Generate high-quality calibration pattern for printing"""
        
        # Calculate pattern dimensions in pixels
        width_px = int(pattern_size_mm[0] * dpi / 25.4)
        height_px = int(pattern_size_mm[1] * dpi / 25.4)
        
        # Calculate square size in pixels
        square_px = int(self.square_size * dpi / 25.4)
        
        # Calculate margins
        pattern_width = (self.chessboard_size[0] + 1) * square_px
        pattern_height = (self.chessboard_size[1] + 1) * square_px
        
        margin_x = (width_px - pattern_width) // 2
        margin_y = (height_px - pattern_height) // 2
        
        # Create white background
        pattern = np.ones((height_px, width_px), dtype=np.uint8) * 255
        
        # Draw chessboard pattern
        for row in range(self.chessboard_size[1] + 1):
            for col in range(self.chessboard_size[0] + 1):
                if (row + col) % 2 == 0:
                    y1 = margin_y + row * square_px
                    y2 = margin_y + (row + 1) * square_px
                    x1 = margin_x + col * square_px
                    x2 = margin_x + (col + 1) * square_px
                    pattern[y1:y2, x1:x2] = 0
        
        # Add border and text
        cv2.rectangle(pattern, (margin_x - 10, margin_y - 10),
                     (margin_x + pattern_width + 10, margin_y + pattern_height + 10),
                     0, 2)
        
        # Add calibration info
        info_text = f"Chessboard: {self.chessboard_size[0]}x{self.chessboard_size[1]}, Square: {self.square_size}mm"
        cv2.putText(pattern, info_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
        
        # Save pattern
        cv2.imwrite(output_path, pattern)
        
        # Generate instructions
        instructions_path = output_path.replace('.png', '_instructions.txt')
        self._generate_instructions(instructions_path)
        
        return output_path
    
    def validate_calibration(self, 
                           calibration: CalibrationResult,
                           test_images: List[str]) -> Dict[str, float]:
        """Validate calibration quality with test images"""
        
        metrics = {
            'mean_reprojection_error': 0.0,
            'max_reprojection_error': 0.0,
            'detection_success_rate': 0.0,
            'distortion_reduction': 0.0
        }
        
        pattern_points = self._create_pattern_points()
        successes = 0
        errors = []
        
        for image_path in test_images:
            img = cv2.imread(image_path)
            if img is None:
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            
            if ret:
                successes += 1
                
                # Refine corners
                if self.enable_refinement:
                    corners = cv2.cornerSubPix(
                        gray, corners, self.refinement_window, (-1, -1), self.criteria
                    )
                
                # Calculate reprojection error
                _, rvec, tvec = cv2.solvePnP(
                    pattern_points, corners,
                    calibration.camera_matrix, calibration.distortion_coefficients
                )
                
                projected, _ = cv2.projectPoints(
                    pattern_points, rvec, tvec,
                    calibration.camera_matrix, calibration.distortion_coefficients
                )
                
                error = cv2.norm(corners, projected, cv2.NORM_L2) / len(projected)
                errors.append(error)
                
                # Measure distortion reduction
                undistorted = cv2.undistort(
                    img, calibration.camera_matrix, calibration.distortion_coefficients
                )
                
                # Simple metric: compare edge straightness
                original_edges = cv2.Canny(gray, 50, 150)
                undist_gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
                undist_edges = cv2.Canny(undist_gray, 50, 150)
                
                metrics['distortion_reduction'] += np.sum(undist_edges) / (np.sum(original_edges) + 1e-6)
        
        if errors:
            metrics['mean_reprojection_error'] = np.mean(errors)
            metrics['max_reprojection_error'] = np.max(errors)
        
        metrics['detection_success_rate'] = successes / len(test_images) if test_images else 0
        metrics['distortion_reduction'] /= max(successes, 1)
        
        return metrics
    
    def _create_pattern_points(self) -> np.ndarray:
        """Create 3D points for calibration pattern"""
        pattern_points = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        pattern_points[:, :2] = np.mgrid[0:self.chessboard_size[0], 
                                       0:self.chessboard_size[1]].T.reshape(-1, 2)
        pattern_points *= self.square_size
        return pattern_points
    
    def _get_calibration_flags(self) -> int:
        """Get calibration flags based on configuration"""
        flags = 0
        
        # Can add various flags based on requirements
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        # flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_FIX_K3
        
        return flags
    
    def _calculate_reprojection_error(self,
                                    object_points: List[np.ndarray],
                                    image_points: List[np.ndarray],
                                    rvecs: List[np.ndarray],
                                    tvecs: List[np.ndarray],
                                    camera_matrix: np.ndarray,
                                    dist_coeffs: np.ndarray) -> float:
        """Calculate mean reprojection error"""
        total_error = 0
        total_points = 0
        
        for i in range(len(object_points)):
            projected, _ = cv2.projectPoints(
                object_points[i], rvecs[i], tvecs[i],
                camera_matrix, dist_coeffs
            )
            error = cv2.norm(image_points[i], projected, cv2.NORM_L2)
            total_error += error * error
            total_points += len(projected)
        
        return np.sqrt(total_error / total_points)
    
    def _generate_instructions(self, output_path: str):
        """Generate calibration instructions"""
        instructions = f"""
Lens Calibration Instructions
============================

1. Pattern Specifications:
   - Chessboard size: {self.chessboard_size[0]}x{self.chessboard_size[1]} internal corners
   - Square size: {self.square_size}mm
   - Print on A4 paper at 100% scale (no fit-to-page)

2. Printing Instructions:
   - Use a high-quality printer
   - Print on matte white paper (avoid glossy)
   - Verify square size with a ruler after printing
   - Mount on a flat, rigid surface (foam board recommended)

3. Image Capture Guidelines:
   - Capture 20-30 images minimum
   - Vary pattern position and orientation:
     * Different distances (fill 30-80% of frame)
     * Different angles (±45° horizontal and vertical)
     * All areas of the frame (center, corners, edges)
   - Ensure sharp focus on the pattern
   - Use consistent, even lighting
   - Avoid shadows and reflections

4. Camera Settings:
   - Use the aperture you most commonly shoot with
   - If the lens has variable focal length, calibrate at each major focal length
   - Use base ISO for minimal noise
   - Ensure sufficient shutter speed to avoid blur

5. File Format:
   - Save as JPEG (high quality) or RAW
   - Maintain consistent image dimensions

6. Quality Checks:
   - Pattern should be clearly visible in all images
   - No motion blur or out-of-focus areas
   - No overexposed or underexposed regions on pattern

Tips for Best Results:
- Take more images than needed and let the software select the best ones
- Include images with the pattern at the extreme edges
- Ensure pattern is flat (check for warping)
- For zoom lenses, create separate calibrations for each focal length
"""
        
        with open(output_path, 'w') as f:
            f.write(instructions)
