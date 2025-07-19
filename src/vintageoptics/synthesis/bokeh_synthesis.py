# src/vintageoptics/synthesis/bokeh_synthesis.py

class BokehSynthesisEngine:
    """Synthesize realistic lens bokeh"""
    
    def __init__(self):
        self.aperture_shapes = self._load_aperture_shapes()
        self.bokeh_kernels = {}
    
    def synthesize_bokeh(self, image: np.ndarray, depth_map: np.ndarray,
                        lens_profile: Dict, settings: Dict) -> np.ndarray:
        """Apply lens-specific bokeh to image based on depth"""
        
        # Get lens bokeh characteristics
        bokeh_params = lens_profile.get('bokeh_characteristics', {})
        aperture_blades = lens_profile.get('aperture_blades', 9)
        
        # Create aperture shape
        aperture_shape = self._create_aperture_shape(
            aperture_blades,
            settings.get('aperture_f'),
            bokeh_params
        )
        
        # Create bokeh kernel for this lens
        bokeh_kernel = self._create_bokeh_kernel(
            aperture_shape,
            bokeh_params,
            settings
        )
        
        # Apply depth-based bokeh
        result = self._apply_depth_based_blur(
            image, depth_map, bokeh_kernel, settings
        )
        
        # Add lens-specific bokeh characteristics
        if bokeh_params.get('swirly_bokeh'):
            result = self._add_swirl_effect(result, depth_map, bokeh_params)
        
        if bokeh_params.get('soap_bubble_bokeh'):
            result = self._add_soap_bubble_effect(result, depth_map, bokeh_params)
        
        if bokeh_params.get('cats_eye_bokeh'):
            result = self._add_cats_eye_vignetting(result, depth_map, bokeh_params)
        
        return result
    
    def _create_aperture_shape(self, num_blades: int, f_stop: float,
                             bokeh_params: Dict) -> np.ndarray:
        """Create realistic aperture shape"""
        
        # Base size depends on f-stop
        base_size = int(100 / np.sqrt(f_stop))
        if base_size % 2 == 0:
            base_size += 1
        
        aperture = np.zeros((base_size, base_size), dtype=np.float32)
        center = base_size // 2
        
        if num_blades < 6:
            # Nearly circular for few blades
            cv2.circle(aperture, (center, center), center-1, 1.0, -1)
        else:
            # Polygonal aperture
            angles = np.linspace(0, 2*np.pi, num_blades, endpoint=False)
            points = []
            for angle in angles:
                x = int(center + (center-1) * np.cos(angle))
                y = int(center + (center-1) * np.sin(angle))
                points.append([x, y])
            points = np.array(points)
            cv2.fillPoly(aperture, [points], 1.0)
        
        # Add aperture blade curvature
        if bokeh_params.get('curved_blades', False):
            aperture = self._curve_aperture_blades(aperture, num_blades)
        
        return aperture
    
    def _create_bokeh_kernel(self, aperture_shape: np.ndarray,
                           bokeh_params: Dict, settings: Dict) -> np.ndarray:
        """Create lens-specific bokeh kernel"""
        
        kernel = aperture_shape.copy()
        
        # Add optical characteristics
        if bokeh_params.get('spherical_aberration', 0) > 0:
            kernel = self._add_spherical_aberration(kernel, bokeh_params)
        
        if bokeh_params.get('onion_rings', False):
            kernel = self._add_onion_rings(kernel, bokeh_params)
        
        # Normalize
        kernel = kernel / np.sum(kernel)
        
        return kernel
    
    def _apply_depth_based_blur(self, image: np.ndarray, depth_map: np.ndarray,
                               bokeh_kernel: np.ndarray, settings: Dict) -> np.ndarray:
        """Apply varying blur based on depth"""
        
        h, w = image.shape[:2]
        result = np.zeros_like(image, dtype=np.float32)
        
        # Determine focus plane
        focus_distance = settings.get('focus_distance', 0.5)
        dof_scale = settings.get('depth_of_field_scale', 0.1)
        
        # Create blur amount map
        blur_map = np.abs(depth_map - focus_distance) / dof_scale
        blur_map = np.clip(blur_map, 0, 1)
        
        # Quantize into blur levels for efficiency
        num_levels = 10
        blur_levels = np.linspace(0, 1, num_levels)
        
        for i, blur_level in enumerate(blur_levels):
            if i == 0:
                # No blur for in-focus areas
                mask = blur_map < (blur_levels[1] / 2)
                result[mask] = image[mask]
                continue
            
            # Create mask for this blur level
            if i < num_levels - 1:
                mask = (blur_map >= blur_levels[i-1]) & (blur_map < blur_levels[i])
            else:
                mask = blur_map >= blur_levels[i-1]
            
            if not np.any(mask):
                continue
            
            # Scale kernel for this blur level
            kernel_size = int(bokeh_kernel.shape[0] * blur_level)
            if kernel_size % 2 == 0:
                kernel_size += 1
            if kernel_size < 3:
                kernel_size = 3
            
            scaled_kernel = cv2.resize(bokeh_kernel, (kernel_size, kernel_size))
            scaled_kernel = scaled_kernel / np.sum(scaled_kernel)
            
            # Apply convolution
            blurred = cv2.filter2D(image, -1, scaled_kernel)
            
            # Blend into result
            for c in range(3):
                result[:, :, c][mask] = blurred[:, :, c][mask]
        
        return result.astype(image.dtype)
    
    def _add_swirl_effect(self, image: np.ndarray, depth_map: np.ndarray,
                         params: Dict) -> np.ndarray:
        """Add swirly bokeh characteristic of certain vintage lenses"""
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Create swirl displacement map
        y, x = np.ogrid[:h, :w]
        x = x - center[0]
        y = y - center[1]
        
        # Radius and angle
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        # Swirl increases with radius and blur amount
        swirl_strength = params.get('swirl_strength', 0.3)
        blur_mask = depth_map > 0.6  # Only in bokeh areas
        
        # Apply swirl
        swirl_amount = swirl_strength * (r / np.max(r)) * blur_mask
        theta_new = theta + swirl_amount
        
        # Convert back to cartesian
        x_new = r * np.cos(theta_new) + center[0]
        y_new = r * np.sin(theta_new) + center[1]
        
        # Remap image
        return cv2.remap(image, x_new.astype(np.float32), 
                        y_new.astype(np.float32), cv2.INTER_LINEAR)