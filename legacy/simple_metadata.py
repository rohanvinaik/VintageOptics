"""
Simple metadata extractor using PIL/Pillow
Fallback when ExifTool is not available
"""

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import logging

logger = logging.getLogger(__name__)

class SimpleMetadataExtractor:
    """Extract basic metadata using PIL"""
    
    def extract_metadata(self, image_path):
        """Extract metadata from image file"""
        try:
            # Open image with PIL
            img = Image.open(image_path)
            
            # Get basic info
            metadata = {
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'width': img.width,
                'height': img.height
            }
            
            # Extract EXIF data
            exifdata = img.getexif()
            
            camera_info = {}
            lens_info = {}
            settings = {}
            
            if exifdata:
                # Process each EXIF tag
                for tag_id, value in exifdata.items():
                    tag = TAGS.get(tag_id, tag_id)
                    
                    # Camera info
                    if tag == 'Make':
                        camera_info['make'] = str(value).strip()
                    elif tag == 'Model':
                        camera_info['model'] = str(value).strip()
                    elif tag == 'BodySerialNumber':
                        camera_info['serial'] = str(value)
                    
                    # Lens info (basic - ExifTool would provide more)
                    elif tag == 'LensModel':
                        lens_info['model'] = str(value).strip()
                    elif tag == 'LensMake':
                        lens_info['make'] = str(value).strip()
                    elif tag == 'LensSerialNumber':
                        lens_info['serial'] = str(value)
                    
                    # Settings
                    elif tag == 'FocalLength':
                        if hasattr(value, 'real'):
                            settings['focal_length'] = f"{value.real}mm"
                        else:
                            settings['focal_length'] = f"{value}mm"
                    elif tag == 'FNumber':
                        if hasattr(value, 'real'):
                            settings['aperture'] = f"f/{value.real}"
                        else:
                            settings['aperture'] = f"f/{value}"
                    elif tag == 'ISOSpeedRatings':
                        settings['iso'] = str(value)
                    elif tag == 'ExposureTime':
                        if hasattr(value, 'numerator') and hasattr(value, 'denominator'):
                            if value.numerator == 1:
                                settings['shutter_speed'] = f"1/{value.denominator}"
                            else:
                                settings['shutter_speed'] = f"{value.numerator}/{value.denominator}"
                        else:
                            settings['shutter_speed'] = str(value)
            
            # Try to infer lens info from camera model if not found
            if not lens_info.get('model') and camera_info.get('model'):
                # Some cameras include lens info in model
                model = camera_info['model']
                if 'mm' in model.lower() or 'f/' in model.lower():
                    # Might be a fixed lens camera
                    lens_info['model'] = f"Integrated lens ({model})"
            
            return {
                'camera': camera_info,
                'lens': lens_info,
                'settings': settings,
                'basic': metadata
            }
            
        except Exception as e:
            logger.error(f"Metadata extraction error: {e}")
            return {
                'camera': {'make': 'Unknown', 'model': 'Unknown'},
                'lens': {'model': 'Unknown'},
                'settings': {},
                'basic': {}
            }
    
    def extract_lens_info(self, image_path):
        """Extract lens-specific information"""
        metadata = self.extract_metadata(image_path)
        
        lens_info = metadata.get('lens', {})
        settings = metadata.get('settings', {})
        
        # Parse focal length
        focal_length = None
        if 'focal_length' in settings:
            try:
                focal_str = settings['focal_length'].replace('mm', '').strip()
                focal_length = float(focal_str)
            except:
                pass
        
        # Parse aperture
        aperture = None
        if 'aperture' in settings:
            try:
                ap_str = settings['aperture'].replace('f/', '').strip()
                aperture = float(ap_str)
            except:
                pass
        
        return {
            'make': lens_info.get('make'),
            'model': lens_info.get('model', 'Unknown'),
            'focal_length': focal_length,
            'aperture': aperture,
            'serial_number': lens_info.get('serial')
        }
