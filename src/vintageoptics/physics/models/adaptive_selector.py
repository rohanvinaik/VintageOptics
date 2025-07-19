# src/vintageoptics/physics/models/adaptive_selector.py
"""
Adaptive model selection for optimal distortion correction.
Automatically selects the best model based on lens characteristics and distortion analysis.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import cv2

from .brown_conrady import BrownConradyModel
from .rational_function import RationalFunctionDistortion, RationalFunctionParams
from .division_model import DivisionModel


class DistortionModelType(Enum):
    """Available distortion correction models"""
    RATIONAL_FUNCTION = "rational_function"
    DIVISION = "division"
    BROWN_CONRADY = "brown_conrady"
    ML_BASED = "ml_based"
    COMBINED = "combined"


@dataclass
class ModelSelectionResult:
    """Result of adaptive model selection"""
    selected_model: DistortionModelType
    confidence: float
    reason: str
    fallback_models: List[DistortionModelType]
    distortion_metrics: Dict[str, float]


class AdaptiveModelSelector:
    """Intelligent selection of distortion correction models"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.distortion_thresholds = {
            'low': 0.01,
            'medium': 0.05,
            'high': 0.1,
            'extreme': 0.2
        }
        
    def analyze_distortion_pattern(self, image: np.ndarray, 
                                  calibration_data: Optional[Dict] = None) -> Dict[str, float]:
        """Analyze distortion characteristics of the image"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        metrics = {}
        
        # Detect straight lines and measure curvature
        edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            # Analyze line curvature
            curvatures = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Calculate distance from center
                line_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                dist_from_center = np.sqrt((line_center[0] - center[0])**2 + 
                                         (line_center[1] - center[1])**2)
                
                # Estimate curvature (simplified)
                if abs(x2 - x1) > 10:
                    slope = (y2 - y1) / (x2 - x1)
                    curvatures.append(abs(slope) * dist_from_center / max(width, height))
            
            if curvatures:
                metrics['mean_curvature'] = np.mean(curvatures)
                metrics['max_curvature'] = np.max(curvatures)
                metrics['curvature_variance'] = np.var(curvatures)
        
        # Analyze radial distortion pattern
        metrics.update(self._analyze_radial_pattern(image))
        
        # Check for asymmetric distortion
        metrics['asymmetry'] = self._measure_asymmetry(image)
        
        # Estimate distortion complexity
        metrics['complexity'] = self._estimate_complexity(metrics)
        
        return metrics
    
    def _analyze_radial_pattern(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze radial distortion characteristics"""
        height, width = image.shape[:2]
        center = np.array([width / 2, height / 2])
        
        # Sample points at different radii
        radial_samples = []
        num_samples = 8
        num_radii = 5
        
        for r_idx in range(1, num_radii + 1):
            radius = (r_idx / num_radii) * min(width, height) / 2
            for angle in np.linspace(0, 2 * np.pi, num_samples, endpoint=False):
                x = int(center[0] + radius * np.cos(angle))
                y = int(center[1] + radius * np.sin(angle))
                
                if 0 <= x < width and 0 <= y < height:
                    radial_samples.append({
                        'radius': radius,
                        'position': (x, y),
                        'intensity': np.mean(image[y, x]) if len(image.shape) == 2 else np.mean(image[y, x, :])
                    })
        
        # Analyze radial falloff
        if radial_samples:
            radii = [s['radius'] for s in radial_samples]
            intensities = [s['intensity'] for s in radial_samples]
            
            # Fit polynomial to radial intensity profile
            if len(radii) > 3:
                coeffs = np.polyfit(radii, intensities, 3)
                return {
                    'radial_coefficient_1': coeffs[2],
                    'radial_coefficient_2': coeffs[1],
                    'radial_coefficient_3': coeffs[0],
                    'radial_nonlinearity': np.std(intensities) / (np.mean(intensities) + 1e-6)
                }
        
        return {'radial_nonlinearity': 0.0}
    
    def _measure_asymmetry(self, image: np.ndarray) -> float:
        """Measure asymmetric distortion in the image"""
        height, width = image.shape[:2]
        
        # Compare left-right and top-bottom symmetry
        left_half = image[:, :width//2]
        right_half = cv2.flip(image[:, width//2:], 1)
        
        # Resize to same dimensions if needed
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        horizontal_asymmetry = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
        
        # Similar for vertical
        top_half = image[:height//2, :]
        bottom_half = cv2.flip(image[height//2:, :], 0)
        
        min_height = min(top_half.shape[0], bottom_half.shape[0])
        top_half = top_half[:min_height, :]
        bottom_half = bottom_half[:min_height, :]
        
        vertical_asymmetry = np.mean(np.abs(top_half.astype(float) - bottom_half.astype(float)))
        
        return (horizontal_asymmetry + vertical_asymmetry) / 510.0  # Normalize to 0-1
    
    def _estimate_complexity(self, metrics: Dict[str, float]) -> float:
        """Estimate overall distortion complexity"""
        complexity = 0.0
        
        # Weight different factors
        if 'max_curvature' in metrics:
            complexity += metrics['max_curvature'] * 2.0
        
        if 'curvature_variance' in metrics:
            complexity += metrics['curvature_variance'] * 1.5
        
        if 'asymmetry' in metrics:
            complexity += metrics['asymmetry'] * 3.0
        
        if 'radial_nonlinearity' in metrics:
            complexity += metrics['radial_nonlinearity'] * 1.0
        
        return min(complexity, 1.0)  # Cap at 1.0
    
    def select_model(self, image: np.ndarray, 
                    lens_profile: Optional[Dict] = None,
                    calibration_data: Optional[Dict] = None) -> ModelSelectionResult:
        """Select optimal distortion model based on analysis"""
        
        # Analyze distortion characteristics
        distortion_metrics = self.analyze_distortion_pattern(image, calibration_data)
        
        # Check lens profile hints
        if lens_profile:
            known_model = lens_profile.get('preferred_model')
            if known_model:
                return ModelSelectionResult(
                    selected_model=DistortionModelType(known_model),
                    confidence=0.95,
                    reason=f"Using known optimal model for {lens_profile.get('name', 'this lens')}",
                    fallback_models=[DistortionModelType.BROWN_CONRADY],
                    distortion_metrics=distortion_metrics
                )
        
        # Model selection logic based on distortion analysis
        complexity = distortion_metrics.get('complexity', 0.0)
        asymmetry = distortion_metrics.get('asymmetry', 0.0)
        max_curvature = distortion_metrics.get('max_curvature', 0.0)
        
        # High complexity or extreme distortion -> Rational Function
        if complexity > 0.7 or max_curvature > self.distortion_thresholds['extreme']:
            return ModelSelectionResult(
                selected_model=DistortionModelType.RATIONAL_FUNCTION,
                confidence=0.85,
                reason="High distortion complexity requires rational function model",
                fallback_models=[
                    DistortionModelType.DIVISION,
                    DistortionModelType.BROWN_CONRADY
                ],
                distortion_metrics=distortion_metrics
            )
        
        # Moderate complexity with low asymmetry -> Division Model
        elif complexity > 0.4 and asymmetry < 0.2:
            return ModelSelectionResult(
                selected_model=DistortionModelType.DIVISION,
                confidence=0.80,
                reason="Moderate symmetric distortion suits division model",
                fallback_models=[
                    DistortionModelType.BROWN_CONRADY,
                    DistortionModelType.RATIONAL_FUNCTION
                ],
                distortion_metrics=distortion_metrics
            )
        
        # High asymmetry -> ML-based or Combined
        elif asymmetry > 0.3:
            return ModelSelectionResult(
                selected_model=DistortionModelType.COMBINED,
                confidence=0.75,
                reason="Asymmetric distortion requires combined approach",
                fallback_models=[
                    DistortionModelType.ML_BASED,
                    DistortionModelType.RATIONAL_FUNCTION
                ],
                distortion_metrics=distortion_metrics
            )
        
        # Default to Brown-Conrady for standard cases
        else:
            return ModelSelectionResult(
                selected_model=DistortionModelType.BROWN_CONRADY,
                confidence=0.90,
                reason="Standard distortion pattern suitable for Brown-Conrady model",
                fallback_models=[
                    DistortionModelType.DIVISION,
                    DistortionModelType.RATIONAL_FUNCTION
                ],
                distortion_metrics=distortion_metrics
            )
    
    def validate_model_performance(self, 
                                 original: np.ndarray,
                                 corrected: np.ndarray,
                                 model_type: DistortionModelType) -> Dict[str, float]:
        """Validate the performance of a correction model"""
        metrics = {}
        
        # Check for over-correction
        original_edges = cv2.Canny(cv2.cvtColor(original, cv2.COLOR_RGB2GRAY), 50, 150)
        corrected_edges = cv2.Canny(cv2.cvtColor(corrected, cv2.COLOR_RGB2GRAY), 50, 150)
        
        # Line straightness improvement
        original_lines = cv2.HoughLinesP(original_edges, 1, np.pi/180, 100, 
                                        minLineLength=100, maxLineGap=10)
        corrected_lines = cv2.HoughLinesP(corrected_edges, 1, np.pi/180, 100,
                                         minLineLength=100, maxLineGap=10)
        
        if original_lines is not None and corrected_lines is not None:
            metrics['line_improvement'] = len(corrected_lines) / (len(original_lines) + 1e-6)
        
        # Measure remaining distortion
        remaining_metrics = self.analyze_distortion_pattern(corrected)
        metrics['remaining_distortion'] = remaining_metrics.get('complexity', 0.0)
        
        # Check for artifacts
        metrics['artifact_score'] = self._detect_correction_artifacts(original, corrected)
        
        return metrics
    
    def _detect_correction_artifacts(self, original: np.ndarray, corrected: np.ndarray) -> float:
        """Detect artifacts introduced by correction"""
        # Simple artifact detection based on local variance changes
        original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(float)
        corrected_gray = cv2.cvtColor(corrected, cv2.COLOR_RGB2GRAY).astype(float)
        
        # Calculate local variance
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        
        original_var = cv2.filter2D(original_gray ** 2, -1, kernel) - cv2.filter2D(original_gray, -1, kernel) ** 2
        corrected_var = cv2.filter2D(corrected_gray ** 2, -1, kernel) - cv2.filter2D(corrected_gray, -1, kernel) ** 2
        
        # Look for areas where variance increased significantly (potential artifacts)
        variance_increase = np.maximum(0, corrected_var - original_var)
        artifact_score = np.mean(variance_increase) / (np.mean(original_var) + 1e-6)
        
        return min(artifact_score, 1.0)
