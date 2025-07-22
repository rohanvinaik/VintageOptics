# src/vintageoptics/core/synthesis_pipeline.py

class SynthesisPipeline:
    """Main pipeline for applying lens characteristics"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.synthesizer = LensCharacteristicSynthesizer(config['database_path'])
        self.fov_transformer = FieldOfViewTransformer()
        self.depth_estimator = MiDaSDepthEstimator()
    
    def apply_lens_to_image(self, image_path: str, target_lens: str,
                          settings: Dict = None) -> SynthesisResult:
        """Apply lens characteristics to an image"""
        
        # Load image and extract metadata
        image = cv2.imread(image_path)
        metadata = self.extract_metadata(image_path)
        
        # Detect source lens (if any)
        source_lens = self.detect_source_lens(metadata)
        
        # Estimate depth if needed
        if settings.get('use_depth', True):
            depth_map = self.depth_estimator.estimate_depth(image)
            settings['depth_map'] = depth_map
        
        # Apply lens characteristics
        result = self.synthesizer.apply_lens_characteristics(
            image, source_lens, target_lens, settings
        )
        
        return SynthesisResult(
            synthesized_image=result,
            source_lens=source_lens,
            target_lens=target_lens,
            depth_map=depth_map,
            settings=settings
        )
    
    def transform_focal_length(self, image_path: str,
                             target_focal: float,
                             preserve_size: bool = True) -> np.ndarray:
        """Change apparent focal length of image"""
        
        image = cv2.imread(image_path)
        metadata = self.extract_metadata(image_path)
        
        # Get source focal length
        source_focal = metadata.get('focal_length', 50)  # Default to 50mm
        
        # Transform
        transformed = self.fov_transformer.transform_focal_length(
            image, source_focal, target_focal
        )
        
        if preserve_size:
            # Resize back to original dimensions
            transformed = cv2.resize(transformed, (image.shape[1], image.shape[0]))
        
        return transformed