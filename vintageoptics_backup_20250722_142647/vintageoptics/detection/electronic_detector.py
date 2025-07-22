# src/vintageoptics/detection/electronic_detector.py

from .base_detector import BaseLensDetector
from typing import Dict, Optional

class ElectronicDetector(BaseLensDetector):
    """Detector for modern electronic lenses with communication protocols"""
    
    def detect(self, image_data) -> float:
        """Detect electronic lens from EXIF and communication data, return probability score"""
        
        metadata = image_data.metadata if hasattr(image_data, 'metadata') else {}
        
        # Extract lens information from EXIF
        lens_info = {}
        
        # Standard EXIF fields
        if 'LensModel' in metadata:
            lens_info['model'] = metadata['LensModel']
        elif 'LensMake' in metadata and 'LensInfo' in metadata:
            lens_info['model'] = f"{metadata['LensMake']} {metadata['LensInfo']}"
        
        # Extract additional electronic lens data
        if 'FocalLength' in metadata:
            lens_info['focal_length'] = metadata['FocalLength']
        
        if 'FNumber' in metadata:
            lens_info['aperture'] = metadata['FNumber']
            
        if 'LensSerialNumber' in metadata:
            lens_info['serial'] = metadata['LensSerialNumber']
            
        # Electronic lens specific features
        if 'LensStabilization' in metadata:
            lens_info['stabilization'] = metadata['LensStabilization']
            
        if lens_info:
            # Return confidence score for electronic detection
            return 0.9 if 'model' in lens_info else 0.5
        
        return 0.0
    
    def _generate_lens_id(self, lens_info: Dict) -> str:
        """Generate unique lens identifier"""
        if 'model' in lens_info:
            # Normalize model name
            model = lens_info['model'].lower()
            model = model.replace(' ', '_')
            model = model.replace('/', '_')
            return model
        return 'unknown_electronic'
