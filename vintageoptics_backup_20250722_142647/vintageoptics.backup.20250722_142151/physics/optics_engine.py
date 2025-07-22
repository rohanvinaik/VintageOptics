# src/vintageoptics/physics/optics_engine.py
"""
Enhanced physics-based optical correction engine with multi-model support.
Integrates adaptive model selection, manufacturing variation compensation,
and confidence metrics for informed user adjustments.
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import logging

from .models.adaptive_selector import AdaptiveModelSelector, DistortionModelType, ModelSelectionResult
from .models.brown_conrady import BrownConradyModel, BrownConradyParams
from .models.rational_function import RationalFunctionDistortion, RationalFunctionParams
from .models.division_model import DivisionModel, DivisionModelParams


logger = logging.getLogger(__name__)


@dataclass
class CorrectionResult:
    """Result of optical correction with metadata"""
    corrected_image: np.ndarray
    model_used: DistortionModelType
    confidence: float
    quality_metrics: Dict[str, float]
    parameters_applied: Dict[str, float]
    alternative_models: List[DistortionModelType]


class OpticsEngine:
    """Enhanced physics-based optical correction engine"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.adaptive_selector = AdaptiveModelSelector(config)
        
        # Initialize correction models
        self.models = {
            DistortionModelType.BROWN_CONRADY: BrownConradyModel(),
            DistortionModelType.RATIONAL_FUNCTION: RationalFunctionDistortion(),
            DistortionModelType.DIVISION: DivisionModel()
        }
        
        # Cache for model parameters
        self.parameter_cache = {}
        
    def apply_corrections(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """Legacy interface - applies corrections with automatic model selection"""
        result = self.apply_adaptive_corrections(image, params)
        return result.corrected_image
    
    def apply_adaptive_corrections(self, 
                                 image: np.ndarray, 
                                 params: Dict,
                                 lens_profile: Optional[Dict] = None,
                                 calibration_data: Optional[Dict] = None,
                                 preserve_character: bool = True) -> CorrectionResult:
        """Apply corrections with adaptive model selection and confidence metrics"""
        
        # Analyze image and select optimal model
        selection_result = self.adaptive_selector.select_model(
            image, lens_profile, calibration_data
        )
        
        logger.info(f"Selected {selection_result.selected_model.value} model "
                   f"with {selection_result.confidence:.2f} confidence: {selection_result.reason}")
        
        # Try primary model
        try:
            corrected = self._apply_model_corrections(
                image, params, selection_result.selected_model, preserve_character
            )
            
            # Validate correction quality
            quality_metrics = self.adaptive_selector.validate_model_performance(
                image, corrected, selection_result.selected_model
            )
            
            # Check if we should try fallback models
            if quality_metrics.get('remaining_distortion', 0) > 0.1 and selection_result.fallback_models:
                logger.info("Primary model insufficient, trying fallback models")
                corrected, selection_result.selected_model = self._try_fallback_models(
                    image, params, selection_result.fallback_models, preserve_character
                )
                
                # Re-evaluate quality
                quality_metrics = self.adaptive_selector.validate_model_performance(
                    image, corrected, selection_result.selected_model
                )
            
        except Exception as e:
            logger.error(f"Error applying {selection_result.selected_model.value}: {e}")
            # Try fallback models
            if selection_result.fallback_models:
                corrected, selection_result.selected_model = self._try_fallback_models(
                    image, params, selection_result.fallback_models, preserve_character
                )
                quality_metrics = {}
            else:
                # Last resort - return original
                corrected = image
                quality_metrics = {'error': str(e)}
        
        # Apply additional corrections
        corrected = self._apply_chromatic_aberration_correction(corrected, params)
        corrected = self._apply_vignetting_correction(corrected, params, preserve_character)
        
        return CorrectionResult(
            corrected_image=np.clip(corrected, 0, 255).astype(image.dtype),
            model_used=selection_result.selected_model,
            confidence=selection_result.confidence,
            quality_metrics={**selection_result.distortion_metrics, **quality_metrics},
            parameters_applied=params,
            alternative_models=selection_result.fallback_models
        )
    
    def _apply_model_corrections(self, 
                               image: np.ndarray, 
                               params: Dict,
                               model_type: DistortionModelType,
                               preserve_character: bool) -> np.ndarray:
        """Apply corrections using specified model"""
        
        if model_type == DistortionModelType.BROWN_CONRADY:
            return self._apply_brown_conrady(image, params, preserve_character)
        elif model_type == DistortionModelType.RATIONAL_FUNCTION:
            return self._apply_rational_function(image, params)
        elif model_type == DistortionModelType.DIVISION:
            return self._apply_division_model(image, params)
        elif model_type == DistortionModelType.COMBINED:
            return self._apply_combined_models(image, params, preserve_character)
        else:
            # ML_BASED would go here when implemented
            return image
    
    def _apply_brown_conrady(self, image: np.ndarray, params: Dict, preserve_character: bool) -> np.ndarray:
        """Apply Brown-Conrady correction with manufacturing variations"""
        model = self.models[DistortionModelType.BROWN_CONRADY]
        
        # Create parameters
        bc_params = BrownConradyParams(
            k1=params.get('distortion_k1', 0.0),
            k2=params.get('distortion_k2', 0.0),
            k3=params.get('distortion_k3', 0.0),
            p1=params.get('distortion_p1', 0.0),
            p2=params.get('distortion_p2', 0.0),
            cx=params.get('center_x'),
            cy=params.get('center_y')
        )
        
        # Apply manufacturing variations if specified
        if params.get('apply_manufacturing_variation', False):
            variation_percent = params.get('manufacturing_variation_percent', 10.0)
            bc_params.apply_manufacturing_variation(variation_percent)
        
        # Apply character preservation adjustments
        if preserve_character:
            bc_params = self._adjust_for_character_preservation(bc_params, params)
        
        model.params = bc_params
        return model.apply_correction(image)
    
    def _apply_rational_function(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """Apply rational function correction for complex distortions"""
        model = self.models[DistortionModelType.RATIONAL_FUNCTION]
        
        # Create parameters
        rf_params = RationalFunctionParams(
            a0=params.get('rf_a0', 1.0),
            a1=params.get('rf_a1', 0.0),
            a2=params.get('rf_a2', 0.0),
            a3=params.get('rf_a3', 0.0),
            b0=params.get('rf_b0', 1.0),
            b1=params.get('rf_b1', 0.0),
            b2=params.get('rf_b2', 0.0),
            b3=params.get('rf_b3', 0.0),
            center_x=params.get('center_x'),
            center_y=params.get('center_y')
        )
        
        model.params = rf_params
        return model.apply_correction(image)
    
    def _apply_division_model(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """Apply division model correction"""
        model = self.models[DistortionModelType.DIVISION]
        
        # Create parameters
        div_params = DivisionModelParams(
            lambda_param=params.get('division_lambda', 0.0),
            center_x=params.get('center_x'),
            center_y=params.get('center_y')
        )
        
        model.params = div_params
        return model.apply_correction(image)
    
    def _apply_combined_models(self, image: np.ndarray, params: Dict, preserve_character: bool) -> np.ndarray:
        """Apply combination of models for complex cases"""
        # Start with division model for bulk correction
        corrected = self._apply_division_model(image, params)
        
        # Fine-tune with Brown-Conrady for residual distortion
        residual_params = {
            'distortion_k1': params.get('residual_k1', 0.0),
            'distortion_k2': params.get('residual_k2', 0.0),
            'distortion_p1': params.get('residual_p1', 0.0),
            'distortion_p2': params.get('residual_p2', 0.0)
        }
        
        if any(abs(residual_params[k]) > 1e-6 for k in residual_params):
            corrected = self._apply_brown_conrady(corrected, residual_params, preserve_character)
        
        return corrected
    
    def _try_fallback_models(self, 
                           image: np.ndarray, 
                           params: Dict,
                           fallback_models: List[DistortionModelType],
                           preserve_character: bool) -> Tuple[np.ndarray, DistortionModelType]:
        """Try fallback models if primary fails"""
        best_corrected = image
        best_model = fallback_models[0] if fallback_models else DistortionModelType.BROWN_CONRADY
        best_score = float('inf')
        
        for model_type in fallback_models:
            try:
                corrected = self._apply_model_corrections(image, params, model_type, preserve_character)
                
                # Quick quality check
                quality = self.adaptive_selector.validate_model_performance(
                    image, corrected, model_type
                )
                
                score = quality.get('remaining_distortion', 1.0) + quality.get('artifact_score', 0.0)
                
                if score < best_score:
                    best_corrected = corrected
                    best_model = model_type
                    best_score = score
                    
            except Exception as e:
                logger.warning(f"Fallback model {model_type.value} failed: {e}")
                continue
        
        return best_corrected, best_model
    
    def _adjust_for_character_preservation(self, params: BrownConradyParams, config: Dict) -> BrownConradyParams:
        """Adjust correction parameters to preserve lens character"""
        preservation_strength = config.get('character_preservation_strength', 0.7)
        
        # Reduce correction strength to preserve character
        params.k1 *= (1 - preservation_strength * 0.3)
        params.k2 *= (1 - preservation_strength * 0.3)
        params.k3 *= (1 - preservation_strength * 0.4)
        params.p1 *= (1 - preservation_strength * 0.2)
        params.p2 *= (1 - preservation_strength * 0.2)
        
        return params
    
    def _apply_chromatic_aberration_correction(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """Apply wavelength-specific chromatic aberration correction"""
        if len(image.shape) != 3:
            return image
        
        h, w = image.shape[:2]
        center_x = params.get('center_x', w / 2.0)
        center_y = params.get('center_y', h / 2.0)
        
        # Wavelength-specific scaling factors
        # Standard wavelengths: 440nm (blue), 550nm (green), 640nm (red)
        red_scale = params.get('chromatic_red', 1.0)
        blue_scale = params.get('chromatic_blue', 1.0)
        
        # Sub-pixel correction for smooth transitions
        if params.get('use_subpixel_ca', True):
            return self._apply_subpixel_chromatic_correction(
                image, red_scale, blue_scale, center_x, center_y
            )
        else:
            # Standard correction
            corrected = image.copy()
            
            for channel, scale in [(0, red_scale), (2, blue_scale)]:
                if abs(scale - 1.0) > 1e-4:
                    scale_matrix = np.array([
                        [scale, 0, center_x * (1 - scale)],
                        [0, scale, center_y * (1 - scale)]
                    ], dtype=np.float32)
                    
                    corrected[:, :, channel] = cv2.warpAffine(
                        image[:, :, channel], scale_matrix, (w, h),
                        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT
                    )
            
            return corrected
    
    def _apply_subpixel_chromatic_correction(self, image: np.ndarray,
                                           red_scale: float, blue_scale: float,
                                           center_x: float, center_y: float) -> np.ndarray:
        """Apply sub-pixel chromatic aberration correction"""
        h, w = image.shape[:2]
        corrected = np.zeros_like(image, dtype=np.float32)
        
        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:h, :w]
        
        # For each channel, apply wavelength-specific correction
        for channel in range(3):
            if channel == 0:  # Red
                scale = red_scale
            elif channel == 2:  # Blue
                scale = blue_scale
            else:  # Green (reference)
                scale = 1.0
            
            if abs(scale - 1.0) < 1e-4:
                corrected[:, :, channel] = image[:, :, channel]
            else:
                # Sub-pixel interpolation using cv2.remap
                x_centered = (x_coords - center_x) * scale + center_x
                y_centered = (y_coords - center_y) * scale + center_y
                
                map_x, map_y = np.meshgrid(x_centered, y_centered)
                corrected[:, :, channel] = cv2.remap(
                    image[:, :, channel], 
                    map_x.astype(np.float32), 
                    map_y.astype(np.float32),
                    cv2.INTER_CUBIC, 
                    borderMode=cv2.BORDER_REFLECT
                )
        
        return corrected.astype(image.dtype)
    
    def _apply_vignetting_correction(self, image: np.ndarray, params: Dict, preserve_character: bool) -> np.ndarray:
        """Two-stage vignetting correction with character preservation"""
        h, w = image.shape[:2]
        center_x = params.get('center_x', w / 2.0)
        center_y = params.get('center_y', h / 2.0)
        
        # Stage 1: Optical vignetting (user-adjustable preservation)
        optical_params = {
            'a1': params.get('vignetting_a1', 0.0),
            'a2': params.get('vignetting_a2', 0.0),
            'a3': params.get('vignetting_a3', 0.0)
        }
        
        # Stage 2: Mechanical vignetting (automatic detection and removal)
        mechanical_vignetting_detected = params.get('mechanical_vignetting_detected', False)
        
        # Skip if no significant vignetting
        if all(abs(optical_params[k]) < 1e-4 for k in optical_params) and not mechanical_vignetting_detected:
            return image
        
        # Create radial distance map
        y_coords, x_coords = np.ogrid[:h, :w]
        x_centered = x_coords - center_x
        y_centered = y_coords - center_y
        max_radius = min(center_x, center_y)
        radius_norm = np.sqrt(x_centered**2 + y_centered**2) / max_radius
        
        # Calculate optical vignetting correction
        optical_correction = 1.0
        if any(abs(optical_params[k]) > 1e-4 for k in optical_params):
            optical_correction = (1.0 + 
                                optical_params['a1'] * radius_norm**2 + 
                                optical_params['a2'] * radius_norm**4 + 
                                optical_params['a3'] * radius_norm**6)
            
            # Apply character preservation
            if preserve_character:
                preservation_factor = params.get('vignetting_preservation', 0.5)
                optical_correction = 1.0 + (optical_correction - 1.0) * (1 - preservation_factor)
        
        # Apply mechanical vignetting correction if detected
        if mechanical_vignetting_detected:
            mechanical_mask = self._detect_mechanical_vignetting(image)
            mechanical_correction = 1.0 / (mechanical_mask + 1e-6)
        else:
            mechanical_correction = 1.0
        
        # Combined correction
        total_correction = optical_correction * mechanical_correction
        
        # Apply correction
        if len(image.shape) == 3:
            corrected = image * total_correction[:, :, np.newaxis]
        else:
            corrected = image * total_correction
        
        return corrected
    
    def _detect_mechanical_vignetting(self, image: np.ndarray) -> np.ndarray:
        """Detect mechanical vignetting pattern"""
        # Simplified detection - in practice, use more sophisticated methods
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Analyze brightness falloff
        h, w = gray.shape
        center = (w // 2, h // 2)
        
        # Sample brightness at different radii
        mask = np.ones((h, w), dtype=np.float32)
        
        # This is a placeholder - implement proper mechanical vignetting detection
        return mask
    
    def estimate_correction_confidence(self, 
                                     image: np.ndarray,
                                     corrected: np.ndarray,
                                     model_type: DistortionModelType) -> Dict[str, float]:
        """Estimate confidence metrics for user feedback"""
        metrics = {}
        
        # Line straightness confidence
        original_lines = self._detect_lines(image)
        corrected_lines = self._detect_lines(corrected)
        
        if original_lines and corrected_lines:
            metrics['line_straightness_confidence'] = min(
                len(corrected_lines) / (len(original_lines) + 1e-6), 1.0
            )
        
        # Distortion correction confidence
        remaining_distortion = self.adaptive_selector.analyze_distortion_pattern(corrected)
        metrics['distortion_correction_confidence'] = 1.0 - remaining_distortion.get('complexity', 0.0)
        
        # Model suitability confidence
        metrics['model_suitability'] = {
            DistortionModelType.BROWN_CONRADY: 0.9,
            DistortionModelType.DIVISION: 0.85,
            DistortionModelType.RATIONAL_FUNCTION: 0.8,
            DistortionModelType.COMBINED: 0.75,
            DistortionModelType.ML_BASED: 0.7
        }.get(model_type, 0.5)
        
        # Overall confidence
        metrics['overall_confidence'] = np.mean([
            metrics.get('line_straightness_confidence', 0.5),
            metrics.get('distortion_correction_confidence', 0.5),
            metrics.get('model_suitability', 0.5)
        ])
        
        return metrics
    
    def _detect_lines(self, image: np.ndarray) -> list:
        """Detect lines in image for quality assessment"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
        return lines if lines is not None else []
