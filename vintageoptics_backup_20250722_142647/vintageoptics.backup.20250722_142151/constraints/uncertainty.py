"""
Uncertainty tracking and propagation system.

Provides confidence intervals and uncertainty quantification throughout
the optical processing pipeline.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union, List
from dataclasses import dataclass
import logging


@dataclass
class UncertaintyEstimate:
    """Container for uncertainty information."""
    
    mean: np.ndarray
    std: np.ndarray
    confidence_level: float = 0.95
    source: str = "unknown"
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def lower_bound(self) -> np.ndarray:
        """Get lower confidence bound."""
        # For 95% confidence, use ~2 standard deviations
        z_score = 1.96 if self.confidence_level == 0.95 else 2.58
        return self.mean - z_score * self.std
    
    @property
    def upper_bound(self) -> np.ndarray:
        """Get upper confidence bound."""
        z_score = 1.96 if self.confidence_level == 0.95 else 2.58
        return self.mean + z_score * self.std
    
    @property
    def relative_uncertainty(self) -> np.ndarray:
        """Get relative uncertainty (coefficient of variation)."""
        return self.std / (np.abs(self.mean) + 1e-10)


class UncertaintyTracker:
    """
    Track and propagate uncertainty through optical processing operations.
    
    Based on principles of error propagation and Bayesian uncertainty quantification.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.uncertainty_sources = {
            'sensor_noise': self._sensor_noise_model,
            'calibration': self._calibration_uncertainty,
            'model': self._model_uncertainty,
            'numerical': self._numerical_uncertainty
        }
    
    def estimate_input_uncertainty(self, image: np.ndarray, 
                                 metadata: Optional[Dict] = None) -> UncertaintyEstimate:
        """Estimate uncertainty in input image."""
        if metadata is None:
            metadata = {}
        
        # Combine multiple uncertainty sources
        uncertainties = []
        
        # Sensor noise (shot noise + read noise)
        sensor_unc = self._sensor_noise_model(image, metadata)
        uncertainties.append(sensor_unc)
        
        # Quantization uncertainty
        bit_depth = metadata.get('bit_depth', 8)
        quant_unc = self._quantization_uncertainty(image, bit_depth)
        uncertainties.append(quant_unc)
        
        # Combine uncertainties (assuming independence)
        combined = self._combine_uncertainties(uncertainties)
        
        return combined
    
    def propagate_through_operation(self, 
                                  input_uncertainty: UncertaintyEstimate,
                                  operation: str,
                                  operation_params: Dict,
                                  jacobian: Optional[np.ndarray] = None) -> UncertaintyEstimate:
        """
        Propagate uncertainty through an operation using linearization.
        
        For operation y = f(x), uncertainty propagates as:
        σ_y² = J * σ_x² * J^T
        where J is the Jacobian of f.
        """
        if jacobian is None:
            # Estimate Jacobian numerically if not provided
            jacobian = self._estimate_jacobian(operation, operation_params)
        
        # Simple case: element-wise operations
        if operation in ['blur', 'sharpen', 'color_correction']:
            # These operations can be approximated as linear transforms
            output_var = jacobian**2 * input_uncertainty.std**2
            output_std = np.sqrt(output_var)
            
            return UncertaintyEstimate(
                mean=input_uncertainty.mean,  # Updated by operation elsewhere
                std=output_std,
                confidence_level=input_uncertainty.confidence_level,
                source=f"{input_uncertainty.source}→{operation}"
            )
        
        # Complex case: non-linear operations
        elif operation in ['distortion_correction', 'vignetting_correction']:
            # Add model uncertainty
            model_unc = self._model_uncertainty(operation, operation_params)
            
            # Combine with propagated uncertainty
            output_var = jacobian**2 * input_uncertainty.std**2 + model_unc**2
            output_std = np.sqrt(output_var)
            
            return UncertaintyEstimate(
                mean=input_uncertainty.mean,
                std=output_std,
                confidence_level=input_uncertainty.confidence_level,
                source=f"{input_uncertainty.source}→{operation}",
                metadata={'model_uncertainty': model_unc}
            )
        
        else:
            # Default: assume uncertainty increases by 10%
            self.logger.warning(f"Unknown operation {operation}, using default uncertainty propagation")
            return UncertaintyEstimate(
                mean=input_uncertainty.mean,
                std=input_uncertainty.std * 1.1,
                confidence_level=input_uncertainty.confidence_level,
                source=f"{input_uncertainty.source}→{operation}"
            )
    
    def propagate_through_pipeline(self, 
                                 initial_uncertainty: UncertaintyEstimate,
                                 pipeline_steps: List[Tuple[str, Dict]]) -> UncertaintyEstimate:
        """Propagate uncertainty through entire pipeline."""
        current_uncertainty = initial_uncertainty
        
        for operation, params in pipeline_steps:
            current_uncertainty = self.propagate_through_operation(
                current_uncertainty, operation, params
            )
        
        return current_uncertainty
    
    def _sensor_noise_model(self, image: np.ndarray, metadata: Dict) -> np.ndarray:
        """Model sensor noise (shot noise + read noise)."""
        # Shot noise follows Poisson statistics: variance = signal
        # Approximate for normalized images
        signal_level = image.mean()
        iso = metadata.get('iso', 100)
        
        # Higher ISO = more noise
        iso_factor = np.sqrt(iso / 100)
        
        # Shot noise standard deviation
        shot_noise_std = np.sqrt(signal_level) * iso_factor * 0.01
        
        # Read noise (constant across image)
        read_noise_std = 0.005 * iso_factor
        
        # Total noise (independent sources add in quadrature)
        total_std = np.sqrt(shot_noise_std**2 + read_noise_std**2)
        
        # Create spatial noise map (shot noise varies with signal)
        noise_map = np.sqrt(image * shot_noise_std**2 + read_noise_std**2)
        
        return noise_map
    
    def _quantization_uncertainty(self, image: np.ndarray, bit_depth: int) -> np.ndarray:
        """Estimate quantization uncertainty."""
        # Quantization step size
        levels = 2**bit_depth
        step_size = 1.0 / levels
        
        # Uniform distribution uncertainty: σ = step_size / sqrt(12)
        quant_std = step_size / np.sqrt(12)
        
        # Constant across image
        return np.full_like(image, quant_std)
    
    def _calibration_uncertainty(self, calibration_type: str, 
                                metadata: Dict) -> float:
        """Estimate uncertainty from calibration accuracy."""
        calibration_uncertainties = {
            'geometric': 0.5,  # pixels
            'photometric': 0.02,  # 2% relative
            'chromatic': 1.0,  # pixels
            'vignetting': 0.05  # 5% relative
        }
        
        base_uncertainty = calibration_uncertainties.get(calibration_type, 0.1)
        
        # Adjust based on calibration quality
        quality = metadata.get('calibration_quality', 'medium')
        quality_factors = {
            'low': 2.0,
            'medium': 1.0,
            'high': 0.5
        }
        
        return base_uncertainty * quality_factors.get(quality, 1.0)
    
    def _model_uncertainty(self, operation: str, params: Dict) -> Union[float, np.ndarray]:
        """Estimate uncertainty from model approximations."""
        model_uncertainties = {
            'distortion_correction': 0.3,  # pixels
            'vignetting_correction': 0.03,  # 3% relative
            'chromatic_correction': 0.5,  # pixels
            'blur_kernel_estimation': 0.1  # relative
        }
        
        base_uncertainty = model_uncertainties.get(operation, 0.05)
        
        # Adjust based on model complexity
        if 'model_order' in params:
            # Higher order models have lower bias but higher variance
            order_factor = 1.0 + 0.1 * params['model_order']
            base_uncertainty *= order_factor
        
        return base_uncertainty
    
    def _numerical_uncertainty(self, operation: str, precision: str = 'float32') -> float:
        """Estimate numerical precision limits."""
        precision_limits = {
            'float16': 1e-3,
            'float32': 1e-6,
            'float64': 1e-15
        }
        
        return precision_limits.get(precision, 1e-6)
    
    def _estimate_jacobian(self, operation: str, params: Dict) -> np.ndarray:
        """Estimate Jacobian matrix for uncertainty propagation."""
        # For most image operations, Jacobian is approximately diagonal
        # with values representing local gain/attenuation
        
        if operation == 'blur':
            # Blur attenuates high frequencies
            kernel_size = params.get('kernel_size', 5)
            attenuation = 1.0 / (kernel_size**2)
            return np.ones((1, 1)) * attenuation
        
        elif operation == 'sharpen':
            # Sharpening amplifies high frequencies
            strength = params.get('strength', 1.0)
            amplification = 1.0 + strength
            return np.ones((1, 1)) * amplification
        
        elif operation == 'color_correction':
            # Color matrix transform
            if 'color_matrix' in params:
                return params['color_matrix']
            else:
                return np.eye(3)  # Identity for RGB
        
        else:
            # Default: unity gain
            return np.ones((1, 1))
    
    def _combine_uncertainties(self, uncertainties: List[Union[float, np.ndarray]]) -> UncertaintyEstimate:
        """Combine multiple uncertainty sources."""
        # Assume independence - variances add
        total_variance = 0
        
        for unc in uncertainties:
            if isinstance(unc, np.ndarray):
                total_variance = total_variance + unc**2
            else:
                total_variance = total_variance + unc**2
        
        total_std = np.sqrt(total_variance)
        
        # Mean is typically the original value (uncertainties are zero-mean)
        if isinstance(total_std, np.ndarray):
            mean = np.zeros_like(total_std)
        else:
            mean = 0.0
        
        return UncertaintyEstimate(
            mean=mean,
            std=total_std,
            source='combined'
        )
    
    def visualize_uncertainty(self, image: np.ndarray, 
                            uncertainty: UncertaintyEstimate,
                            output_path: Optional[str] = None) -> np.ndarray:
        """Create visualization of uncertainty map."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray' if image.ndim == 2 else None)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Uncertainty map
        unc_map = uncertainty.std
        if unc_map.ndim == 3:
            unc_map = unc_map.mean(axis=2)
        
        im1 = axes[0, 1].imshow(unc_map, cmap='hot')
        axes[0, 1].set_title('Uncertainty Map')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Relative uncertainty
        rel_unc = uncertainty.relative_uncertainty
        if rel_unc.ndim == 3:
            rel_unc = rel_unc.mean(axis=2)
        
        im2 = axes[1, 0].imshow(rel_unc, cmap='viridis', vmin=0, vmax=0.5)
        axes[1, 0].set_title('Relative Uncertainty')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 1])
        
        # Confidence bounds visualization
        # Show regions where uncertainty exceeds threshold
        high_unc_mask = rel_unc > 0.2  # 20% relative uncertainty
        
        axes[1, 1].imshow(image, cmap='gray' if image.ndim == 2 else None, alpha=0.5)
        axes[1, 1].imshow(high_unc_mask, cmap='Reds', alpha=0.5)
        axes[1, 1].set_title('High Uncertainty Regions (>20%)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        # Convert figure to array for return
        fig.canvas.draw()
        viz_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        viz_array = viz_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return viz_array
    
    def create_uncertainty_report(self, 
                                uncertainty_estimates: Dict[str, UncertaintyEstimate]) -> Dict:
        """Create summary report of uncertainties in pipeline."""
        report = {
            'summary': {},
            'details': {},
            'recommendations': []
        }
        
        for stage, uncertainty in uncertainty_estimates.items():
            # Summary statistics
            mean_unc = float(np.mean(uncertainty.std))
            max_unc = float(np.max(uncertainty.std))
            mean_rel_unc = float(np.mean(uncertainty.relative_uncertainty))
            
            report['summary'][stage] = {
                'mean_uncertainty': mean_unc,
                'max_uncertainty': max_unc,
                'mean_relative_uncertainty': mean_rel_unc,
                'confidence_level': uncertainty.confidence_level
            }
            
            # Detailed analysis
            report['details'][stage] = {
                'source': uncertainty.source,
                'metadata': uncertainty.metadata,
                'high_uncertainty_fraction': float(
                    np.mean(uncertainty.relative_uncertainty > 0.2)
                )
            }
        
        # Generate recommendations
        for stage, stats in report['summary'].items():
            if stats['mean_relative_uncertainty'] > 0.15:
                report['recommendations'].append(
                    f"High uncertainty in {stage} stage - consider higher quality "
                    f"calibration or input data"
                )
            
            if stats['max_uncertainty'] > 10 * stats['mean_uncertainty']:
                report['recommendations'].append(
                    f"Large uncertainty variations in {stage} - check for "
                    f"problematic image regions"
                )
        
        return report
