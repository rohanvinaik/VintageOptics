# src/vintageoptics/synthesis/fov_transformer.py

class FieldOfViewTransformer:
    """Transform images between different focal lengths"""
    
    def __init__(self):
        self.projection_engine = ProjectionEngine()
        self.inpainting_engine = InpaintingEngine()
    
    def transform_focal_length(self, image: np.ndarray,
                             source_focal: float,
                             target_focal: float,
                             sensor_size: str = 'FF',
                             method: str = 'adaptive') -> np.ndarray:
        """Transform image to appear shot at different focal length"""
        
        # Calculate field of view change
        source_fov = self._focal_to_fov(source_focal, sensor_size)
        target_fov = self._focal_to_fov(target_focal, sensor_size)
        
        if target_fov > source_fov:
            # Need wider FOV - must synthesize missing content
            return self._expand_field_of_view(image, source_fov, target_fov, method)
        else:
            # Narrower FOV - crop and adjust perspective
            return self._narrow_field_of_view(image, source_fov, target_fov)
    
    def _expand_field_of_view(self, image: np.ndarray,
                             source_fov: float,
                             target_fov: float,
                             method: str) -> np.ndarray:
        """Expand FOV using AI-assisted synthesis"""
        
        expansion_factor = target_fov / source_fov
        
        if method == 'adaptive':
            if expansion_factor < 1.2:
                # Small expansion - use smart padding
                return self._smart_padding_expansion(image, expansion_factor)
            elif expansion_factor < 1.5:
                # Medium expansion - use outpainting
                return self._outpainting_expansion(image, expansion_factor)
            else:
                # Large expansion - use neural synthesis
                return self._neural_synthesis_expansion(image, expansion_factor)
        
    def _outpainting_expansion(self, image: np.ndarray, 
                              factor: float) -> np.ndarray:
        """Use outpainting techniques to expand image"""
        
        h, w = image.shape[:2]
        new_h = int(h * factor)
        new_w = int(w * factor)
        
        # Create canvas
        canvas = np.zeros((new_h, new_w, 3), dtype=image.dtype)
        
        # Position original
        y_off = (new_h - h) // 2
        x_off = (new_w - w) // 2
        canvas[y_off:y_off+h, x_off:x_off+w] = image
        
        # Create gradient masks for smooth blending
        mask = np.zeros((new_h, new_w), dtype=np.float32)
        mask[y_off:y_off+h, x_off:x_off+w] = 1.0
        
        # Gradient borders
        border_width = 50
        for i in range(border_width):
            alpha = i / border_width
            mask[y_off-i-1, x_off:x_off+w] = alpha if y_off-i-1 >= 0 else 0
            mask[y_off+h+i, x_off:x_off+w] = alpha if y_off+h+i < new_h else 0
            mask[y_off:y_off+h, x_off-i-1] = alpha if x_off-i-1 >= 0 else 0
            mask[y_off:y_off+h, x_off+w+i] = alpha if x_off+w+i < new_w else 0
        
        # Use PatchMatch or deep learning for synthesis
        if self._has_stable_diffusion_inpaint():
            result = self._stable_diffusion_outpaint(canvas, mask)
        else:
            # Fallback to classical methods
            result = self._classical_outpaint(canvas, mask)
        
        return result