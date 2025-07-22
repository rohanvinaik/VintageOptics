# src/vintageoptics/depth/layer_processor.py

class LayerBasedProcessor:
    """Apply different corrections to different depth layers"""
    
    def __init__(self, physics_engine: OpticsEngine, 
                 statistical_cleaner: StatisticalCleaner):
        self.physics_engine = physics_engine
        self.statistical_cleaner = statistical_cleaner
        self.bokeh_preservers = BokehPreservationEngine()
    
    def process_with_depth_awareness(self, image: np.ndarray,
                                    depth_map: DepthMap,
                                    lens_params: Dict,
                                    preserve_bokeh: bool = True) -> np.ndarray:
        """Process image with depth-aware corrections"""
        
        processed = image.copy()
        h, w = image.shape[:2]
        
        # Process each depth layer differently
        for layer in depth_map.depth_layers:
            # Create smooth transition mask
            layer_mask = self._create_smooth_layer_mask(layer.mask)
            
            # Determine correction strength based on depth
            correction_params = self._adapt_corrections_for_layer(
                base_params=lens_params,
                layer=layer,
                preserve_bokeh=preserve_bokeh
            )
            
            # Apply physics corrections with adapted parameters
            layer_corrected = self._apply_layer_corrections(
                image=image,
                mask=layer_mask,
                params=correction_params,
                layer=layer
            )
            
            # Blend with original using mask
            processed = self._blend_layers(processed, layer_corrected, layer_mask)
        
        # Apply cross-layer consistency
        processed = self._ensure_layer_consistency(processed, depth_map)
        
        return processed
    
    def _adapt_corrections_for_layer(self, base_params: Dict, 
                                   layer: DepthLayer,
                                   preserve_bokeh: bool) -> Dict:
        """Adapt correction parameters based on layer depth"""
        
        adapted_params = base_params.copy()
        
        # Determine if layer is in focus
        focus_score = layer.blur_characteristics['focus_score']
        is_bokeh = focus_score < 0.3  # Heavily blurred
        is_transition = 0.3 <= focus_score <= 0.7
        is_sharp = focus_score > 0.7
        
        if preserve_bokeh and is_bokeh:
            # Minimal corrections for bokeh areas
            adapted_params.update({
                'distortion_strength': 0.2,  # Minimal distortion correction
                'ca_strength': 0.3,          # Light CA correction
                'vignetting_strength': 0.5,  # Moderate vignetting
                'sharpening': 0.0,           # No sharpening
                'defect_removal_sensitivity': 0.7  # Less aggressive
            })
        elif is_sharp:
            # Full corrections for in-focus areas
            adapted_params.update({
                'distortion_strength': 1.0,
                'ca_strength': 1.0,
                'vignetting_strength': 0.8,
                'sharpening': 0.3,
                'defect_removal_sensitivity': 1.0
            })
        else:  # Transition zone
            # Graduated corrections
            adapted_params.update({
                'distortion_strength': 0.5 + focus_score * 0.5,
                'ca_strength': 0.4 + focus_score * 0.6,
                'vignetting_strength': 0.6 + focus_score * 0.2,
                'sharpening': focus_score * 0.2,
                'defect_removal_sensitivity': 0.8
            })
        
        # Special handling for specific bokeh types
        if layer.blur_characteristics.get('swirly_bokeh'):
            adapted_params['preserve_swirl'] = True
            adapted_params['distortion_strength'] *= 0.5
        
        if layer.blur_characteristics.get('soap_bubble_bokeh'):
            adapted_params['preserve_outlining'] = True
            adapted_params['ca_strength'] *= 0.7
        
        return adapted_params
    
    def _apply_layer_corrections(self, image: np.ndarray, mask: np.ndarray,
                               params: Dict, layer: DepthLayer) -> np.ndarray:
        """Apply corrections to a specific layer"""
        
        # Extract layer region with padding for edge handling
        padded_mask, crop_coords = self._extract_layer_with_padding(mask)
        layer_region = self._extract_region(image, crop_coords)
        
        # Apply physics corrections with adapted parameters
        if params.get('distortion_strength', 1.0) > 0:
            layer_region = self.physics_engine.apply_distortion_correction(
                layer_region, 
                strength=params['distortion_strength']
            )
        
        if params.get('ca_strength', 1.0) > 0:
            layer_region = self.physics_engine.apply_chromatic_correction(
                layer_region,
                strength=params['ca_strength']
            )
        
        # Different defect handling for bokeh vs sharp areas
        if layer.blur_characteristics['focus_score'] > 0.7:
            # Sharp area - full defect removal
            layer_region = self.statistical_cleaner.remove_defects(
                layer_region,
                sensitivity=params['defect_removal_sensitivity']
            )
        else:
            # Bokeh area - preserve circular artifacts, remove only linear defects
            layer_region = self._remove_bokeh_safe_defects(
                layer_region,
                bokeh_type=layer.blur_characteristics.get('bokeh_type', 'standard')
            )
        
        # Bokeh enhancement (if applicable)
        if params.get('enhance_bokeh') and layer.blur_characteristics['is_bokeh']:
            layer_region = self.bokeh_preservers.enhance_bokeh(
                layer_region,
                bokeh_characteristics=layer.blur_characteristics
            )
        
        return layer_region
    
    def _remove_bokeh_safe_defects(self, image: np.ndarray, 
                                  bokeh_type: str) -> np.ndarray:
        """Remove defects while preserving bokeh characteristics"""
        
        # Detect linear scratches (safe to remove in bokeh)
        scratch_mask = self._detect_linear_defects(image)
        
        # Avoid removing circular features in bokeh
        if bokeh_type in ['soap_bubble', 'swirly']:
            circular_features = self._detect_circular_features(image)
            scratch_mask = scratch_mask & ~circular_features
        
        # Apply targeted inpainting
        if np.any(scratch_mask):
            image = cv2.inpaint(image, scratch_mask, 3, cv2.INPAINT_TELEA)
        
        return image