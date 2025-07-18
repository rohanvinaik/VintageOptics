# src/vintageoptics/depth/__init__.py

"""
Depth-aware processing module
"""

from .depth_analyzer import DepthFromDefocusAnalyzer

class DepthAwareProcessor:
    """Depth-aware image processing"""
    
    def __init__(self, config):
        self.config = config
        self.depth_analyzer = DepthFromDefocusAnalyzer(config)
        self.depth_estimator = None
        
    def estimate_depth(self, image):
        """Estimate depth map from single image"""
        # Use MiDaS or similar for depth estimation
        try:
            from ..integrations.midas_integration import MiDaSIntegration
            if self.depth_estimator is None:
                self.depth_estimator = MiDaSIntegration(self.config)
            return self.depth_estimator.estimate_depth(image)
        except ImportError:
            # Fallback to defocus analysis
            return self.depth_analyzer.estimate_from_defocus(image)
    
    def process_with_layers(self, image, depth_map, params):
        """Process image with depth layer awareness"""
        if depth_map is None:
            # No depth info, process normally
            return image.copy()
            
        # Segment image into depth layers
        import numpy as np
        num_layers = self.config.get('depth', {}).get('num_layers', 3)
        
        # Quantize depth map
        depth_min, depth_max = depth_map.min(), depth_map.max()
        depth_normalized = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
        depth_quantized = np.floor(depth_normalized * num_layers).astype(int)
        depth_quantized = np.clip(depth_quantized, 0, num_layers - 1)
        
        # Process each layer separately
        result = np.zeros_like(image)
        
        for layer_idx in range(num_layers):
            # Create mask for current layer
            layer_mask = (depth_quantized == layer_idx).astype(np.float32)
            
            # Adjust correction strength based on depth
            layer_params = params.copy()
            depth_factor = 1.0 - (layer_idx / num_layers)  # Stronger correction for closer objects
            
            for key in ['distortion_k1', 'distortion_k2', 'chromatic_red', 'chromatic_blue']:
                if key in layer_params:
                    layer_params[key] *= depth_factor
            
            # Apply corrections to layer
            from ..physics import OpticsEngine
            optics = OpticsEngine(self.config)
            layer_corrected = optics.apply_corrections(image, layer_params)
            
            # Blend into result
            if len(layer_mask.shape) == 2:
                layer_mask = layer_mask[:, :, np.newaxis]
            result += layer_corrected * layer_mask
            
        return result.astype(image.dtype)

__all__ = ['DepthAwareProcessor', 'DepthFromDefocusAnalyzer']
