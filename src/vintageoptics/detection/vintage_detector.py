# src/vintageoptics/detection/vintage_detector.py

from .base_detector import BaseLensDetector
from typing import Dict, Optional

class VintageLensDetector(BaseLensDetector):
    """Detector for vintage manual lenses using optical fingerprinting"""
    
    def detect(self, image_data) -> Optional[Dict]:
        """Detect vintage lens from optical characteristics"""
        
        # Import here to avoid circular dependency
        from .lens_fingerprinting import LensFingerprinter
        
        # Basic EXIF check first
        metadata = image_data.metadata if hasattr(image_data, 'metadata') else {}
        
        lens_info = {}
        
        # Check for manual lens indicators
        if 'LensModel' in metadata:
            model = metadata['LensModel'].lower()
            # Common vintage lens patterns
            vintage_patterns = ['helios', 'takumar', 'jupiter', 'mir', 'industar', 
                               'zeiss jena', 'meyer', 'vivitar', 'minolta rokkor']
            
            if any(pattern in model for pattern in vintage_patterns):
                lens_info['model'] = metadata['LensModel']
                lens_info['vintage_detected'] = True
        
        # Try optical fingerprinting
        fingerprinter = LensFingerprinter(self.config)
        optical_signature = fingerprinter.extract_signature(image_data.image)
        
        if optical_signature:
            # Match against known vintage lens database
            matched_lens = self._match_vintage_lens(optical_signature)
            if matched_lens:
                lens_info.update(matched_lens)
                
        if lens_info:
            return {
                'lens_type': 'vintage',
                'lens_id': self._generate_vintage_id(lens_info),
                'confidence': 0.7 if 'model' in lens_info else 0.4,
                **lens_info
            }
            
        return None
    
    def _match_vintage_lens(self, signature: Dict) -> Optional[Dict]:
        """Match optical signature against vintage lens database"""
        
        # Simplified matching - in production would query database
        # Check for characteristic swirly bokeh (Helios)
        if signature.get('bokeh_swirl', 0) > 0.7:
            return {
                'model': 'Helios 44-2 58mm f/2',
                'mount': 'M42',
                'character': 'swirly_bokeh'
            }
        
        # Check for radioactive yellowing (Takumar)
        if signature.get('yellow_cast', 0) > 0.3:
            return {
                'model': 'Super Takumar 50mm f/1.4',
                'mount': 'M42',
                'character': 'radioactive_yellow'
            }
            
        return None
    
    def _generate_vintage_id(self, lens_info: Dict) -> str:
        """Generate vintage lens identifier"""
        if 'model' in lens_info:
            model = lens_info['model'].lower()
            # Common normalizations for vintage lenses
            model = model.replace(' ', '_')
            model = model.replace('-', '_')
            model = model.replace('/', '_')
            
            # Extract focal length if present
            import re
            focal_match = re.search(r'(\d+)mm', model)
            if focal_match:
                focal = focal_match.group(1)
                return f"{model.split()[0]}_{focal}mm"
                
            return model
            
        return 'unknown_vintage'
