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
    
    def process_with_layers(self, image, depth_map, params):
        """Process image with depth layer awareness"""
        # Stub implementation
        return image.copy()

__all__ = ['DepthAwareProcessor', 'DepthFromDefocusAnalyzer']
