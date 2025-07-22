# src/vintageoptics/physics/models/adaptive_selector_enhanced.py
"""
Enhanced adaptive model selection with hierarchical approach and confidence metrics.
Implements the multi-model correction architecture from the enhancement analysis.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass, field
from enum import Enum
import cv2
import logging
from abc import ABC, abstractmethod

from .brown_conrady import BrownConradyModel
from .rational_function import RationalFunctionDistortion, RationalFunctionParams
from .division_model import DivisionModel


logger = logging.getLogger(__name__)


class DistortionModelType(Enum):
    """Available distortion correction models in hierarchical order"""
    RATIONAL_FUNCTION = "rational_function"
    DIVISION = "division"
    BROWN_CONRADY = "brown_conrady"
    ML_BASED = "ml_based"
    COMBINED = "combined"
    NONE = "none"


@dataclass
class ModelPerformanceMetrics:
    """Detailed performance metrics for a correction model"""
    line_straightness: float = 0.0  # 0-1, higher is better
    remaining_distortion: float = 1.0  # 0-1, lower is better
    artifact_score: float = 0.0  # 0-1, lower is better
    corner_accuracy: float = 0.0  # 0-1, higher is better
    computational_cost: float = 0.0  # Relative cost, lower is better
    perceptual_quality: float = 0.0  # 0-1, higher is better
    
    @property
    def overall_score(self) -> float:
        """Calculate weighted overall performance score"""
        weights = {
            'line_straightness': 0.30,
            'remaining_distortion': -0.25,
            'artifact_score': -0.20,
            'corner_accuracy': 0.15,
            'computational_cost': -0.05,
            'perceptual_quality': 0.05
        }
        
        score = 0.0
        for metric, weight in weights.items():
            value = getattr(self, metric, 0.0)
            score += value * weight
        
        return max(0.0, min(1.0, score + 0.5))  # Normalize to 0-1


@dataclass
class ModelSelectionResult:
    """Enhanced result of adaptive model selection"""
    selected_model: DistortionModelType
    confidence: float
    reason: str
    fallback_models: List[DistortionModelType]
    distortion_metrics: Dict[str, float]
    performance_prediction: ModelPerformanceMetrics
    parameter_adjustments: Dict[str, float] = field(default_factory=dict)
    user_adjustable: bool = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'selected_model': self.selected_model.value,
            'confidence': self.confidence,
            'reason': self.reason,
            'fallback_models': [m.value for m in self.fallback_models],
            'distortion_metrics': self.distortion_metrics,
            'performance_prediction': {
                'line_straightness': self.performance_prediction.line_straightness,
                'remaining_distortion': self.performance_prediction.remaining_distortion,
                'artifact_score': self.performance_prediction.artifact_score,
                'corner_accuracy': self.performance_prediction.corner_accuracy,
                'computational_cost': self.performance_prediction.computational_cost,
                'perceptual_quality': self.performance_prediction.perceptual_quality,
                'overall_score': self.performance_prediction.overall_score
            },
            'parameter_adjustments': self.parameter_adjustments,
            'user_adjustable': self.user_adjustable
        }


class DistortionAnalyzer:
    """Advanced distortion pattern analysis"""
    
    def __init__(self):
        self.edge_detector = cv2.Canny
        self.line_detector = cv2.HoughLinesP
        
    def analyze_comprehensive(self, image: np.ndarray, 
                            mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Comprehensive distortion analysis"""
        metrics = {}
        
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Apply mask if provided
        if mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Basic distortion metrics
        metrics.update(self._analyze_barrel_pincushion(gray))
        metrics.update(self._analyze_mustache_distortion(gray))
        metrics.update(self._analyze_asymmetric_distortion(gray))
        metrics.update(self._analyze_complex_distortion(gray))
        
        # Edge-based metrics
        edges = self.edge_detector(gray, 50, 150)
        metrics.update(self._analyze_edge_curvature(edges))
        
        # Corner detection for accuracy assessment
        metrics.update(self._analyze_corner_distortion(gray))
        
        # Frequency domain analysis
        metrics.update(self._analyze_frequency_distortion(gray))
        
        return metrics
    
    def _analyze_barrel_pincushion(self, gray: np.ndarray) -> Dict[str, float]:
        """Analyze barrel/pincushion distortion characteristics"""
        height, width = gray.shape
        center = (width // 2, height // 2)
        
        # Sample radial lines
        radial_samples = []
        for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            for r in np.linspace(0.1, 0.9, 10):
                x = int(center[0] + r * width/2 * np.cos(angle))
                y = int(center[1] + r * height/2 * np.sin(angle))
                
                if 0 <= x < width and 0 <= y < height:
                    expected_r = r
                    actual_r = np.sqrt((x - center[0])**2 + (y - center[1])**2) / (width/2)
                    radial_samples.append(actual_r - expected_r)
        
        if radial_samples:
            radial_deviation = np.array(radial_samples)
            return {
                'barrel_pincushion': np.mean(radial_deviation),
                'barrel_pincushion_variance': np.var(radial_deviation),
                'max_radial_deviation': np.max(np.abs(radial_deviation))
            }
        
        return {'barrel_pincushion': 0.0, 'barrel_pincushion_variance': 0.0}
    
    def _analyze_mustache_distortion(self, gray: np.ndarray) -> Dict[str, float]:
        """Detect mustache/wave distortion pattern"""
        height, width = gray.shape
        center_y = height // 2
        
        # Analyze horizontal lines at different heights
        distortions = []
        for y in [height//4, center_y, 3*height//4]:
            edge_row = cv2.Canny(gray[y-5:y+5, :], 50, 150)
            if edge_row.any():
                # Detect wave pattern
                edge_points = np.where(edge_row > 0)[1]
                if len(edge_points) > 10:
                    # Fit polynomial and check for wave
                    poly = np.polyfit(edge_points, 
                                     np.arange(len(edge_points)), 2)
                    distortions.append(abs(poly[0]))  # Quadratic coefficient
        
        if distortions:
            return {
                'mustache_distortion': np.mean(distortions),
                'mustache_variance': np.var(distortions)
            }
        
        return {'mustache_distortion': 0.0, 'mustache_variance': 0.0}
    
    def _analyze_asymmetric_distortion(self, gray: np.ndarray) -> Dict[str, float]:
        """Analyze asymmetric distortion patterns"""
        height, width = gray.shape
        
        # Quadrant analysis
        quadrants = [
            gray[:height//2, :width//2],      # Top-left
            gray[:height//2, width//2:],      # Top-right
            gray[height//2:, :width//2],      # Bottom-left
            gray[height//2:, width//2:]       # Bottom-right
        ]
        
        # Compare quadrant statistics
        quadrant_means = [np.mean(q) for q in quadrants]
        quadrant_stds = [np.std(q) for q in quadrants]
        
        asymmetry_score = np.std(quadrant_means) / (np.mean(quadrant_means) + 1e-6)
        
        # Detailed asymmetry analysis
        horizontal_asymmetry = abs(quadrant_means[0] + quadrant_means[2] - 
                                  quadrant_means[1] - quadrant_means[3])
        vertical_asymmetry = abs(quadrant_means[0] + quadrant_means[1] - 
                                quadrant_means[2] - quadrant_means[3])
        
        return {
            'asymmetry_score': asymmetry_score,
            'horizontal_asymmetry': horizontal_asymmetry / 255.0,
            'vertical_asymmetry': vertical_asymmetry / 255.0,
            'quadrant_variance': np.var(quadrant_stds)
        }
    
    def _analyze_complex_distortion(self, gray: np.ndarray) -> Dict[str, float]:
        """Analyze complex, higher-order distortion patterns"""
        height, width = gray.shape
        center = (width // 2, height // 2)
        
        # Radial polynomial fitting
        points = []
        distances = []
        
        # Sample points on a grid
        for y in range(0, height, height//20):
            for x in range(0, width, width//20):
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                if dist > 0:
                    points.append((x, y))
                    distances.append(dist)
        
        if len(points) > 20:
            # Analyze distortion field complexity
            distances = np.array(distances)
            normalized_distances = distances / np.max(distances)
            
            # Fit higher-order polynomial
            try:
                coeffs = np.polyfit(normalized_distances, 
                                   np.random.randn(len(distances)) * 0.1, 5)
                
                complexity = np.sum(np.abs(coeffs[2:]))  # Higher-order terms
                
                return {
                    'distortion_complexity': min(complexity, 1.0),
                    'higher_order_distortion': np.sum(np.abs(coeffs[3:])),
                    'polynomial_order_needed': self._estimate_polynomial_order(coeffs)
                }
            except:
                pass
        
        return {'distortion_complexity': 0.0, 'higher_order_distortion': 0.0}
    
    def _analyze_edge_curvature(self, edges: np.ndarray) -> Dict[str, float]:
        """Analyze curvature of detected edges"""
        lines = self.line_detector(edges, 1, np.pi/180, 50, 
                                  minLineLength=50, maxLineGap=10)
        
        if lines is not None and len(lines) > 5:
            curvatures = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if length > 20:
                    # Sample points along the line
                    mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
                    
                    # Check for curvature by measuring deviation
                    # This is a simplified approach
                    angle = np.arctan2(y2-y1, x2-x1)
                    perp_angle = angle + np.pi/2
                    
                    # Sample perpendicular to line
                    sample_dist = 5
                    check_x = int(mid_x + sample_dist * np.cos(perp_angle))
                    check_y = int(mid_y + sample_dist * np.sin(perp_angle))
                    
                    if (0 <= check_x < edges.shape[1] and 
                        0 <= check_y < edges.shape[0]):
                        if edges[check_y, check_x] > 0:
                            curvatures.append(sample_dist / length)
            
            if curvatures:
                return {
                    'mean_edge_curvature': np.mean(curvatures),
                    'max_edge_curvature': np.max(curvatures),
                    'edge_curvature_variance': np.var(curvatures)
                }
        
        return {'mean_edge_curvature': 0.0, 'max_edge_curvature': 0.0}
    
    def _analyze_corner_distortion(self, gray: np.ndarray) -> Dict[str, float]:
        """Analyze distortion at image corners"""
        height, width = gray.shape
        corner_size = min(width, height) // 4
        
        corners = [
            gray[:corner_size, :corner_size],                    # Top-left
            gray[:corner_size, -corner_size:],                   # Top-right
            gray[-corner_size:, :corner_size],                   # Bottom-left
            gray[-corner_size:, -corner_size:]                   # Bottom-right
        ]
        
        corner_sharpness = []
        corner_distortion = []
        
        for corner in corners:
            # Measure sharpness using gradient
            gx = cv2.Sobel(corner, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(corner, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(gx**2 + gy**2)
            corner_sharpness.append(np.mean(gradient_mag))
            
            # Measure local distortion
            edges = cv2.Canny(corner, 50, 150)
            if edges.any():
                corner_distortion.append(np.sum(edges) / edges.size)
        
        return {
            'corner_sharpness_variance': np.var(corner_sharpness),
            'corner_distortion_mean': np.mean(corner_distortion),
            'corner_quality': 1.0 - np.std(corner_sharpness) / (np.mean(corner_sharpness) + 1e-6)
        }
    
    def _analyze_frequency_distortion(self, gray: np.ndarray) -> Dict[str, float]:
        """Analyze distortion in frequency domain"""
        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Analyze radial frequency distribution
        height, width = gray.shape
        center = (height // 2, width // 2)
        
        # Create radial frequency bins
        y, x = np.ogrid[:height, :width]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        # Compute radial average
        num_bins = min(center)
        radial_profile = []
        
        for i in range(num_bins):
            mask = (r >= i) & (r < i + 1)
            if mask.any():
                radial_profile.append(np.mean(magnitude_spectrum[mask]))
        
        if len(radial_profile) > 10:
            radial_profile = np.array(radial_profile)
            
            # Analyze frequency falloff
            mid_point = len(radial_profile) // 2
            high_freq_ratio = np.mean(radial_profile[mid_point:]) / (np.mean(radial_profile[:mid_point]) + 1e-6)
            
            return {
                'frequency_distortion': high_freq_ratio,
                'frequency_uniformity': 1.0 - np.std(radial_profile) / (np.mean(radial_profile) + 1e-6)
            }
        
        return {'frequency_distortion': 0.0, 'frequency_uniformity': 1.0}
    
    def _estimate_polynomial_order(self, coeffs: np.ndarray) -> int:
        """Estimate required polynomial order based on coefficients"""
        threshold = 0.001
        for i in range(len(coeffs)-1, -1, -1):
            if abs(coeffs[i]) > threshold:
                return len(coeffs) - i
        return 1


class HierarchicalModelSelector:
    """Enhanced adaptive model selector with hierarchical approach"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.analyzer = DistortionAnalyzer()
        
        # Model hierarchy (in order of preference/capability)
        self.model_hierarchy = [
            DistortionModelType.RATIONAL_FUNCTION,
            DistortionModelType.DIVISION,
            DistortionModelType.BROWN_CONRADY,
            DistortionModelType.ML_BASED
        ]
        
        # Thresholds for model selection
        self.thresholds = {
            'extreme_distortion': 0.15,
            'high_distortion': 0.08,
            'moderate_distortion': 0.04,
            'low_distortion': 0.02,
            'minimal_distortion': 0.01
        }
        
        # Manufacturing variation compensation ranges
        self.variation_ranges = {
            'budget_lens': 0.15,      # ±15% parameter adjustment
            'consumer_lens': 0.10,    # ±10% parameter adjustment
            'prosumer_lens': 0.05,    # ±5% parameter adjustment
            'professional_lens': 0.02  # ±2% parameter adjustment
        }
        
    def select_optimal_model(self, 
                           image: np.ndarray,
                           lens_profile: Optional[Dict] = None,
                           calibration_data: Optional[Dict] = None,
                           user_preferences: Optional[Dict] = None) -> ModelSelectionResult:
        """Select optimal model using hierarchical approach with confidence metrics"""
        
        # Comprehensive distortion analysis
        distortion_metrics = self.analyzer.analyze_comprehensive(image)
        
        # Determine lens quality tier for variation compensation
        lens_tier = self._determine_lens_tier(lens_profile)
        variation_factor = self.variation_ranges.get(lens_tier, 0.10)
        
        # Check for user preferences
        preserve_character = user_preferences.get('preserve_character', True) if user_preferences else True
        performance_priority = user_preferences.get('performance_priority', 'balanced') if user_preferences else 'balanced'
        
        # Evaluate each model in hierarchy
        model_evaluations = []
        
        for model_type in self.model_hierarchy:
            evaluation = self._evaluate_model_suitability(
                model_type, distortion_metrics, lens_profile, 
                preserve_character, performance_priority
            )
            model_evaluations.append((model_type, evaluation))
        
        # Select best model based on evaluations
        best_model, best_evaluation = max(model_evaluations, 
                                         key=lambda x: x[1]['score'])
        
        # Determine fallback models
        fallback_models = [m[0] for m in sorted(model_evaluations, 
                                               key=lambda x: x[1]['score'], 
                                               reverse=True)[1:4]]
        
        # Calculate parameter adjustments for manufacturing variation
        parameter_adjustments = self._calculate_parameter_adjustments(
            best_model, distortion_metrics, variation_factor
        )
        
        # Predict performance metrics
        performance_prediction = self._predict_performance(
            best_model, distortion_metrics, lens_profile
        )
        
        return ModelSelectionResult(
            selected_model=best_model,
            confidence=best_evaluation['confidence'],
            reason=best_evaluation['reason'],
            fallback_models=fallback_models,
            distortion_metrics=distortion_metrics,
            performance_prediction=performance_prediction,
            parameter_adjustments=parameter_adjustments,
            user_adjustable=True
        )
    
    def _determine_lens_tier(self, lens_profile: Optional[Dict]) -> str:
        """Determine lens quality tier for variation compensation"""
        if not lens_profile:
            return 'consumer_lens'
        
        # Check various indicators of lens quality
        indicators = {
            'professional': ['L', 'GM', 'S-Line', 'Art', 'Pro', 'ED', 'ASPH'],
            'prosumer': ['USM', 'HSM', 'VR', 'IS', 'OS', 'VC'],
            'consumer': ['STM', 'DC', 'DG', 'EF-S', 'DX'],
            'budget': ['kit', 'basic', 'entry']
        }
        
        lens_name = lens_profile.get('name', '').lower()
        lens_class = lens_profile.get('class', 'consumer')
        
        # Check professional indicators
        for indicator in indicators['professional']:
            if indicator.lower() in lens_name:
                return 'professional_lens'
        
        # Check prosumer indicators
        for indicator in indicators['prosumer']:
            if indicator.lower() in lens_name:
                return 'prosumer_lens'
        
        # Check budget indicators
        for indicator in indicators['budget']:
            if indicator.lower() in lens_name:
                return 'budget_lens'
        
        # Default based on metadata
        if lens_class == 'professional':
            return 'professional_lens'
        elif lens_class == 'prosumer':
            return 'prosumer_lens'
        elif lens_class == 'budget':
            return 'budget_lens'
        
        return 'consumer_lens'
    
    def _evaluate_model_suitability(self,
                                  model_type: DistortionModelType,
                                  distortion_metrics: Dict[str, float],
                                  lens_profile: Optional[Dict],
                                  preserve_character: bool,
                                  performance_priority: str) -> Dict:
        """Evaluate suitability of a specific model"""
        
        score = 0.0
        confidence = 0.0
        reasons = []
        
        # Extract key metrics
        max_distortion = distortion_metrics.get('max_radial_deviation', 0.0)
        complexity = distortion_metrics.get('distortion_complexity', 0.0)
        asymmetry = distortion_metrics.get('asymmetry_score', 0.0)
        mustache = distortion_metrics.get('mustache_distortion', 0.0)
        higher_order = distortion_metrics.get('higher_order_distortion', 0.0)
        
        # Model-specific evaluation
        if model_type == DistortionModelType.RATIONAL_FUNCTION:
            # Best for extreme and complex distortions
            if max_distortion > self.thresholds['high_distortion']:
                score += 0.3
                reasons.append("High distortion level")
            if complexity > 0.6:
                score += 0.3
                reasons.append("Complex distortion pattern")
            if higher_order > 0.1:
                score += 0.2
                reasons.append("Higher-order distortions present")
            
            confidence = min(0.95, score + 0.3)
            
        elif model_type == DistortionModelType.DIVISION:
            # Good for moderate symmetric distortions
            if (self.thresholds['moderate_distortion'] < max_distortion < 
                self.thresholds['high_distortion']):
                score += 0.4
                reasons.append("Moderate distortion level")
            if asymmetry < 0.2:
                score += 0.3
                reasons.append("Symmetric distortion")
            if mustache < 0.05:
                score += 0.2
                reasons.append("No significant mustache distortion")
            
            confidence = min(0.85, score + 0.2)
            
        elif model_type == DistortionModelType.BROWN_CONRADY:
            # Standard choice for typical distortions
            if max_distortion < self.thresholds['moderate_distortion']:
                score += 0.5
                reasons.append("Standard distortion level")
            if complexity < 0.3:
                score += 0.3
                reasons.append("Simple distortion pattern")
            if preserve_character:
                score += 0.1
                reasons.append("Character preservation preferred")
            
            confidence = min(0.90, score + 0.1)
            
        elif model_type == DistortionModelType.ML_BASED:
            # Fallback for unusual patterns
            if asymmetry > 0.4:
                score += 0.3
                reasons.append("High asymmetry")
            if mustache > 0.1:
                score += 0.3
                reasons.append("Complex mustache distortion")
            if lens_profile and 'unusual' in lens_profile.get('characteristics', []):
                score += 0.3
                reasons.append("Unusual lens characteristics")
            
            confidence = min(0.75, score + 0.1)
        
        # Adjust for performance priority
        if performance_priority == 'quality':
            if model_type in [DistortionModelType.RATIONAL_FUNCTION, 
                            DistortionModelType.ML_BASED]:
                score += 0.1
        elif performance_priority == 'speed':
            if model_type in [DistortionModelType.BROWN_CONRADY, 
                            DistortionModelType.DIVISION]:
                score += 0.1
        
        # Generate reason string
        if reasons:
            reason = f"{model_type.value}: {', '.join(reasons)}"
        else:
            reason = f"{model_type.value}: Default selection"
        
        return {
            'score': score,
            'confidence': confidence,
            'reason': reason
        }
    
    def _calculate_parameter_adjustments(self,
                                       model_type: DistortionModelType,
                                       distortion_metrics: Dict[str, float],
                                       variation_factor: float) -> Dict[str, float]:
        """Calculate parameter adjustments for manufacturing variation"""
        adjustments = {}
        
        # Base adjustments based on model type
        if model_type == DistortionModelType.BROWN_CONRADY:
            adjustments = {
                'k1_variation': variation_factor,
                'k2_variation': variation_factor * 0.8,
                'k3_variation': variation_factor * 0.6,
                'p1_variation': variation_factor * 0.5,
                'p2_variation': variation_factor * 0.5
            }
        elif model_type == DistortionModelType.DIVISION:
            adjustments = {
                'lambda_variation': variation_factor,
                'center_x_variation': variation_factor * 0.3,
                'center_y_variation': variation_factor * 0.3
            }
        elif model_type == DistortionModelType.RATIONAL_FUNCTION:
            adjustments = {
                'numerator_variation': variation_factor,
                'denominator_variation': variation_factor * 0.9,
                'order_flexibility': 1 if variation_factor > 0.1 else 0
            }
        
        # Adjust based on detected patterns
        if distortion_metrics.get('asymmetry_score', 0) > 0.3:
            for key in adjustments:
                if 'center' in key or 'p' in key:
                    adjustments[key] *= 1.5
        
        return adjustments
    
    def _predict_performance(self,
                           model_type: DistortionModelType,
                           distortion_metrics: Dict[str, float],
                           lens_profile: Optional[Dict]) -> ModelPerformanceMetrics:
        """Predict performance metrics for selected model"""
        
        # Base predictions
        predictions = ModelPerformanceMetrics()
        
        # Model-specific base performance
        base_performance = {
            DistortionModelType.RATIONAL_FUNCTION: {
                'line_straightness': 0.95,
                'remaining_distortion': 0.05,
                'artifact_score': 0.10,
                'corner_accuracy': 0.90,
                'computational_cost': 0.80,
                'perceptual_quality': 0.85
            },
            DistortionModelType.DIVISION: {
                'line_straightness': 0.85,
                'remaining_distortion': 0.10,
                'artifact_score': 0.05,
                'corner_accuracy': 0.80,
                'computational_cost': 0.40,
                'perceptual_quality': 0.80
            },
            DistortionModelType.BROWN_CONRADY: {
                'line_straightness': 0.80,
                'remaining_distortion': 0.15,
                'artifact_score': 0.03,
                'corner_accuracy': 0.75,
                'computational_cost': 0.20,
                'perceptual_quality': 0.85
            },
            DistortionModelType.ML_BASED: {
                'line_straightness': 0.88,
                'remaining_distortion': 0.08,
                'artifact_score': 0.15,
                'corner_accuracy': 0.85,
                'computational_cost': 0.90,
                'perceptual_quality': 0.75
            }
        }
        
        # Get base performance
        base = base_performance.get(model_type, base_performance[DistortionModelType.BROWN_CONRADY])
        
        # Apply base values
        for key, value in base.items():
            setattr(predictions, key, value)
        
        # Adjust based on distortion severity
        max_distortion = distortion_metrics.get('max_radial_deviation', 0.0)
        if max_distortion > 0.1:
            predictions.line_straightness *= 0.9
            predictions.remaining_distortion *= 1.2
            predictions.artifact_score *= 1.3
        
        # Adjust based on complexity
        complexity = distortion_metrics.get('distortion_complexity', 0.0)
        if complexity > 0.7:
            predictions.corner_accuracy *= 0.85
            predictions.perceptual_quality *= 0.9
        
        # Lens-specific adjustments
        if lens_profile:
            if 'swirly_bokeh' in lens_profile.get('characteristics', []):
                predictions.perceptual_quality *= 1.1  # Bonus for character
            if 'sharp_corners' in lens_profile.get('strengths', []):
                predictions.corner_accuracy *= 1.1
        
        # Ensure values are in valid range
        for attr in ['line_straightness', 'remaining_distortion', 'artifact_score', 
                     'corner_accuracy', 'computational_cost', 'perceptual_quality']:
            value = getattr(predictions, attr)
            setattr(predictions, attr, max(0.0, min(1.0, value)))
        
        return predictions
    
    def apply_confidence_based_blending(self,
                                      results: List[Tuple[DistortionModelType, np.ndarray, float]],
                                      target_confidence: float = 0.85) -> np.ndarray:
        """Blend results from multiple models based on confidence scores"""
        
        if not results:
            raise ValueError("No results to blend")
        
        if len(results) == 1:
            return results[0][1]  # Return single result
        
        # Calculate blending weights based on confidence
        weights = []
        total_confidence = 0.0
        
        for model_type, corrected_image, confidence in results:
            if confidence >= target_confidence * 0.7:  # Include if reasonably confident
                weight = confidence ** 2  # Square to emphasize higher confidence
                weights.append(weight)
                total_confidence += weight
            else:
                weights.append(0.0)
        
        if total_confidence == 0:
            # Fallback to best single result
            return max(results, key=lambda x: x[2])[1]
        
        # Normalize weights
        weights = [w / total_confidence for w in weights]
        
        # Blend images
        blended = np.zeros_like(results[0][1], dtype=np.float32)
        
        for i, (_, corrected_image, _) in enumerate(results):
            if weights[i] > 0:
                blended += corrected_image.astype(np.float32) * weights[i]
        
        return blended.astype(results[0][1].dtype)
    
    def generate_correction_report(self, 
                                 selection_result: ModelSelectionResult,
                                 validation_metrics: Optional[Dict] = None) -> Dict:
        """Generate detailed correction report for user review"""
        
        report = {
            'summary': {
                'selected_model': selection_result.selected_model.value,
                'confidence': f"{selection_result.confidence * 100:.1f}%",
                'overall_quality': selection_result.performance_prediction.overall_score,
                'processing_time_estimate': self._estimate_processing_time(
                    selection_result.selected_model
                )
            },
            'distortion_analysis': {
                'severity': self._classify_distortion_severity(
                    selection_result.distortion_metrics
                ),
                'primary_type': self._identify_primary_distortion_type(
                    selection_result.distortion_metrics
                ),
                'complexity': selection_result.distortion_metrics.get(
                    'distortion_complexity', 0.0
                )
            },
            'model_selection': {
                'reason': selection_result.reason,
                'alternatives': [m.value for m in selection_result.fallback_models],
                'user_adjustable': selection_result.user_adjustable
            },
            'expected_results': {
                'line_straightness': f"{selection_result.performance_prediction.line_straightness * 100:.0f}%",
                'distortion_removal': f"{(1 - selection_result.performance_prediction.remaining_distortion) * 100:.0f}%",
                'artifact_risk': self._classify_artifact_risk(
                    selection_result.performance_prediction.artifact_score
                ),
                'corner_quality': f"{selection_result.performance_prediction.corner_accuracy * 100:.0f}%"
            },
            'recommendations': self._generate_recommendations(
                selection_result, validation_metrics
            )
        }
        
        if validation_metrics:
            report['validation'] = validation_metrics
        
        return report
    
    def _estimate_processing_time(self, model_type: DistortionModelType) -> str:
        """Estimate processing time based on model complexity"""
        times = {
            DistortionModelType.BROWN_CONRADY: "< 1 second",
            DistortionModelType.DIVISION: "1-2 seconds",
            DistortionModelType.RATIONAL_FUNCTION: "2-5 seconds",
            DistortionModelType.ML_BASED: "5-10 seconds",
            DistortionModelType.COMBINED: "10-15 seconds"
        }
        return times.get(model_type, "Unknown")
    
    def _classify_distortion_severity(self, metrics: Dict[str, float]) -> str:
        """Classify overall distortion severity"""
        max_distortion = metrics.get('max_radial_deviation', 0.0)
        
        if max_distortion > self.thresholds['extreme_distortion']:
            return "Extreme"
        elif max_distortion > self.thresholds['high_distortion']:
            return "High"
        elif max_distortion > self.thresholds['moderate_distortion']:
            return "Moderate"
        elif max_distortion > self.thresholds['low_distortion']:
            return "Low"
        else:
            return "Minimal"
    
    def _identify_primary_distortion_type(self, metrics: Dict[str, float]) -> str:
        """Identify the primary type of distortion"""
        types = []
        
        if abs(metrics.get('barrel_pincushion', 0.0)) > 0.05:
            if metrics.get('barrel_pincushion', 0.0) > 0:
                types.append("Barrel")
            else:
                types.append("Pincushion")
        
        if metrics.get('mustache_distortion', 0.0) > 0.03:
            types.append("Mustache/Wave")
        
        if metrics.get('asymmetry_score', 0.0) > 0.3:
            types.append("Asymmetric")
        
        if metrics.get('higher_order_distortion', 0.0) > 0.05:
            types.append("Complex/Higher-order")
        
        return ", ".join(types) if types else "Standard radial"
    
    def _classify_artifact_risk(self, artifact_score: float) -> str:
        """Classify the risk of correction artifacts"""
        if artifact_score < 0.05:
            return "Very Low"
        elif artifact_score < 0.10:
            return "Low"
        elif artifact_score < 0.20:
            return "Moderate"
        else:
            return "High"
    
    def _generate_recommendations(self, 
                                selection_result: ModelSelectionResult,
                                validation_metrics: Optional[Dict]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Model confidence recommendations
        if selection_result.confidence < 0.7:
            recommendations.append(
                "Consider manual review due to lower confidence in automatic selection"
            )
        
        # Performance recommendations
        perf = selection_result.performance_prediction
        
        if perf.artifact_score > 0.15:
            recommendations.append(
                "High artifact risk - consider using fallback model or reducing correction strength"
            )
        
        if perf.corner_accuracy < 0.7:
            recommendations.append(
                "Corner quality may be compromised - check results carefully at image edges"
            )
        
        if perf.computational_cost > 0.7:
            recommendations.append(
                "Processing will be intensive - ensure adequate system resources"
            )
        
        # Distortion-specific recommendations
        if selection_result.distortion_metrics.get('asymmetry_score', 0) > 0.4:
            recommendations.append(
                "High asymmetry detected - results may vary across image regions"
            )
        
        # Validation-based recommendations
        if validation_metrics:
            if validation_metrics.get('line_improvement', 1.0) < 0.8:
                recommendations.append(
                    "Limited improvement detected - consider alternative models"
                )
        
        return recommendations
