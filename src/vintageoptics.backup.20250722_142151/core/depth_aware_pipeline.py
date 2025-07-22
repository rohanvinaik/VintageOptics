# src/vintageoptics/core/depth_aware_pipeline.py

class DepthAwarePipeline:
    """Enhanced pipeline with depth-aware processing"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.depth_analyzer = DepthFromDefocusAnalyzer(config)
        self.layer_processor = LayerBasedProcessor(
            physics_engine=OpticsEngine(),
            statistical_cleaner=StatisticalCleaner()
        )
        self.bokeh_analyzer = BokehAnalyzer()
        self.depth_cache = {}
    
    def process_with_depth_intelligence(self, image_path: str,
                                      lens_data: Dict,
                                      output_path: str = None) -> ProcessingResult:
        """Process image with full depth awareness"""
        
        # Load image
        image = self.load_image(image_path)
        
        # Extract or compute depth map
        depth_map = self._get_or_compute_depth_map(image, lens_data)
        
        # Analyze bokeh characteristics
        bokeh_analysis = self.bokeh_analyzer.analyze_bokeh_characteristics(
            image, depth_map.depth_map
        )
        
        # Process with layer awareness
        processed = self.layer_processor.process_with_depth_awareness(
            image=image,
            depth_map=depth_map,
            lens_params=lens_data,
            preserve_bokeh=self.config.get('preserve_bokeh', True)
        )
        
        # Optional: enhance depth effect
        if self.config.get('enhance_depth_separation', False):
            processed = self._enhance_depth_separation(
                processed, depth_map, bokeh_analysis
            )
        
        # Save with metadata
        result = ProcessingResult(
            corrected_image=processed,
            depth_map=depth_map,
            bokeh_analysis=bokeh_analysis,
            processing_metadata={
                'depth_layers': len(depth_map.depth_layers),
                'bokeh_preserved': self.config.get('preserve_bokeh', True),
                'layer_specific_corrections': True
            }
        )
        
        if output_path:
            self.save_with_depth_metadata(result, output_path)
        
        return result
    
    def _enhance_depth_separation(self, image: np.ndarray,
                                depth_map: DepthMap,
                                bokeh_analysis: Dict) -> np.ndarray:
        """Optionally enhance depth separation"""
        
        enhanced = image.copy()
        
        # Subtle contrast adjustment based on depth
        for layer in depth_map.depth_layers:
            if layer.blur_characteristics['focus_score'] > 0.8:
                # Slightly boost contrast in sharp areas
                layer_region = image * layer.mask[:, :, np.newaxis]
                layer_region = self._adaptive_contrast_enhancement(
                    layer_region, strength=0.1
                )
                enhanced = self._blend_regions(enhanced, layer_region, layer.mask)
            elif layer.blur_characteristics['is_bokeh']:
                # Optionally enhance bokeh rendering
                if bokeh_analysis.get('quality_score', 0) < 0.7:
                    layer_region = self._enhance_bokeh_quality(
                        image * layer.mask[:, :, np.newaxis],
                        bokeh_analysis
                    )
                    enhanced = self._blend_regions(enhanced, layer_region, layer.mask)
        
        return enhanced
    
    def save_with_depth_metadata(self, result: ProcessingResult, output_path: str):
        """Save processed image with depth information"""
        
        # Save main image
        cv2.imwrite(output_path, cv2.cvtColor(result.corrected_image, cv2.COLOR_RGB2BGR))
        
        # Save depth map visualization
        depth_vis = self._visualize_depth_map(result.depth_map)
        depth_path = output_path.rsplit('.', 1)[0] + '_depth.png'
        cv2.imwrite(depth_path, depth_vis)
        
        # Save processing metadata
        import json
        metadata = {
            'depth_analysis': {
                'num_layers': len(result.depth_map.depth_layers),
                'focus_points': [fp.to_dict() for fp in result.depth_map.focus_points],
                'bokeh_type': result.bokeh_analysis.get('bokeh_type', 'none'),
                'bokeh_quality': result.bokeh_analysis.get('quality_score', 0)
            },
            'processing': result.processing_metadata
        }
        
        metadata_path = output_path.rsplit('.', 1)[0] + '_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)