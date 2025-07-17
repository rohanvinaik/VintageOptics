# src/vintageoptics/synthesis/optical_characteristics.py

class OpticalCharacteristicEngine:
    """Apply specific optical characteristics of vintage lenses"""
    
    def apply_vintage_coating_effects(self, image: np.ndarray, 
                                    lens_profile: Dict) -> np.ndarray:
        """Simulate vintage lens coating characteristics"""
        
        coating_type = lens_profile.get('coating_type', 'single')
        coating_age = lens_profile.get('coating_age_factor', 0.0)
        
        result = image.copy().astype(np.float32)
        
        if coating_type == 'uncoated':
            # Lower contrast, more flare
            result = self._reduce_contrast(result, 0.8)
            result = self._add_veiling_flare(result, 0.1)
        elif coating_type == 'single':
            # Slight contrast reduction, color shift
            result = self._reduce_contrast(result, 0.9)
            result = self._add_color_cast(result, lens_profile.get('color_cast', 'warm'))
        
        # Age effects
        if coating_age > 0:
            result = self._simulate_coating_degradation(result, coating_age)
        
        return result.astype(image.dtype)
    
    def apply_vintage_glass_effects(self, image: np.ndarray,
                                  lens_profile: Dict) -> np.ndarray:
        """Simulate vintage glass characteristics"""
        
        glass_types = lens_profile.get('glass_types', [])
        
        result = image.copy()
        
        if 'thoriated' in glass_types:
            # Slight yellow tint from thorium
            result = self._add_thorium_tint(result, 0.05)
        
        if 'flint' in glass_types:
            # Higher dispersion
            result = self._enhance_chromatic_aberration(result, 1.2)
        
        if 'lanthanide' in glass_types:
            # Better color correction but different rendering
            result = self._adjust_color_rendering(result, 'lanthanide')
        
        return result
    
    def simulate_specific_lens_models(self, image: np.ndarray,
                                    lens_model: str,
                                    settings: Dict) -> np.ndarray:
        """Apply characteristics of specific legendary lenses"""
        
        if lens_model == "helios_44":
            return self._apply_helios_44_character(image, settings)
        elif lens_model == "petzval":
            return self._apply_petzval_character(image, settings)
        elif lens_model == "sonnar":
            return self._apply_sonnar_character(image, settings)
        elif lens_model == "summicron":
            return self._apply_summicron_character(image, settings)
        # Add more iconic lenses...
        
        return image
    
    def _apply_helios_44_character(self, image: np.ndarray, 
                                  settings: Dict) -> np.ndarray:
        """Apply Helios 44 swirly bokeh and rendering"""
        
        depth_map = settings.get('depth_map')
        if depth_map is None:
            depth_map = self._estimate_simple_depth(image)
        
        # Strong swirly bokeh
        result = self._add_helios_swirl(image, depth_map)
        
        # Characteristic contrast
        result = self._adjust_contrast_curve(result, 'helios')
        
        # Slight warm tint
        result = self._add_color_cast(result, 'warm', 0.03)
        
        return result