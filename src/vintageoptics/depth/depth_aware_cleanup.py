# src/vintageoptics/depth/depth_aware_cleanup.py

class DepthAwareDefectHandler:
    """Handle defects differently based on depth context"""
    
    def __init__(self):
        self.defect_classifier = DefectClassifier()
        self.bokeh_safe_remover = BokehSafeDefectRemover()
        self.sharp_area_remover = SharpAreaDefectRemover()
    
    def remove_defects_with_depth_context(self, image: np.ndarray,
                                         depth_map: DepthMap) -> np.ndarray:
        """Remove defects considering their depth context"""
        
        # Classify defects by type and location
        defects = self.defect_classifier.detect_all_defects(image)
        
        corrected = image.copy()
        
        for defect in defects:
            # Get depth at defect location
            defect_depth = self._get_defect_depth(defect, depth_map)
            defect_layer = self._get_layer_for_depth(defect_depth, depth_map)
            
            # Determine removal strategy based on context
            if defect_layer.blur_characteristics['is_bokeh']:
                # In bokeh - be very careful
                if self._should_remove_in_bokeh(defect, defect_layer):
                    corrected = self.bokeh_safe_remover.remove_defect(
                        corrected, defect, defect_layer
                    )
            else:
                # In sharp area - standard removal
                corrected = self.sharp_area_remover.remove_defect(
                    corrected, defect
                )
        
        return corrected
    
    def _should_remove_in_bokeh(self, defect: Defect, layer: DepthLayer) -> bool:
        """Determine if defect should be removed from bokeh area"""
        
        # Always remove obvious sensor dust and scratches
        if defect.type in ['dust', 'scratch', 'hair']:
            return True
        
        # Preserve features that could be bokeh characteristics
        if defect.type == 'circular_artifact':
            # Could be soap bubble bokeh
            if layer.blur_characteristics.get('bokeh_type') == 'soap_bubble':
                return False
            
            # Check if it matches bokeh ball characteristics
            if self._matches_bokeh_signature(defect, layer):
                return False
        
        # Fungus patterns - remove if confidence is high
        if defect.type == 'fungus':
            return defect.confidence > 0.8
        
        return True