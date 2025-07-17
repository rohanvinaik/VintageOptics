# src/vintageoptics/detection/metadata_extractor.py
"""
Comprehensive metadata extraction from images including EXIF, XMP, IPTC,
and maker notes for lens identification and profiling.
"""

import os
import json
import struct
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime
import logging
import xml.etree.ElementTree as ET

# Try to import various metadata libraries
try:
    import exifread
    EXIFREAD_AVAILABLE = True
except ImportError:
    EXIFREAD_AVAILABLE = False
    
try:
    from PIL import Image, ExifTags
    from PIL.ExifTags import TAGS, GPSTAGS
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import piexif
    PIEXIF_AVAILABLE = True
except ImportError:
    PIEXIF_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Comprehensive metadata extraction for lens detection"""
    
    def __init__(self):
        self.check_dependencies()
        
        # Lens model patterns for manual lens detection
        self.manual_lens_patterns = {
            'helios': ['helios', 'гелиос'],
            'jupiter': ['jupiter', 'юпитер'],
            'industar': ['industar', 'индустар'],
            'takumar': ['takumar', 'super-takumar', 'smc takumar'],
            'zeiss': ['zeiss', 'contax', 'planar', 'sonnar', 'biogon'],
            'meyer': ['meyer', 'görlitz', 'domiplan', 'oreston'],
            'canon_fd': ['canon fd', 'canon fl'],
            'nikon_ai': ['nikkor-', 'nikon ai', 'nikkor ai'],
            'minolta_md': ['minolta md', 'minolta mc', 'rokkor'],
            'pentax_k': ['pentax-k', 'pentax k', 'smc pentax'],
            'olympus_om': ['olympus om', 'zuiko'],
            'm42': ['m42', 'pentacon', 'chinon', 'vivitar']
        }
        
    def check_dependencies(self):
        """Check available metadata libraries"""
        available = []
        if EXIFREAD_AVAILABLE:
            available.append('exifread')
        if PIL_AVAILABLE:
            available.append('PIL')
        if PIEXIF_AVAILABLE:
            available.append('piexif')
            
        if not available:
            logger.warning("No metadata extraction libraries available. "
                         "Install exifread, Pillow, or piexif for full functionality.")
        else:
            logger.info(f"Available metadata libraries: {', '.join(available)}")
    
    def extract_all_metadata(self, image_path: str) -> Dict[str, Any]:
        """Extract all available metadata from image"""
        metadata = {
            'file_info': self._extract_file_info(image_path),
            'exif': {},
            'xmp': {},
            'iptc': {},
            'maker_notes': {},
            'lens_info': {},
            'camera_info': {},
            'shooting_info': {},
            'processing_info': {}
        }
        
        # Try multiple extraction methods
        if PIL_AVAILABLE:
            pil_data = self._extract_with_pil(image_path)
            metadata.update(pil_data)
            
        if EXIFREAD_AVAILABLE:
            exifread_data = self._extract_with_exifread(image_path)
            metadata['exif'].update(exifread_data.get('exif', {}))
            metadata['maker_notes'].update(exifread_data.get('maker_notes', {}))
            
        if PIEXIF_AVAILABLE:
            piexif_data = self._extract_with_piexif(image_path)
            metadata['exif'].update(piexif_data.get('exif', {}))
            
        # Extract XMP if present
        xmp_data = self._extract_xmp(image_path)
        if xmp_data:
            metadata['xmp'] = xmp_data
            
        # Process and organize metadata
        metadata['lens_info'] = self._extract_lens_info(metadata)
        metadata['camera_info'] = self._extract_camera_info(metadata)
        metadata['shooting_info'] = self._extract_shooting_info(metadata)
        
        # Detect manual/adapted lenses
        manual_lens = self._detect_manual_lens(metadata)
        if manual_lens:
            metadata['lens_info']['manual_lens_detected'] = manual_lens
            
        return metadata
    
    def _extract_file_info(self, image_path: str) -> Dict[str, Any]:
        """Extract basic file information"""
        stat = os.stat(image_path)
        return {
            'filename': os.path.basename(image_path),
            'size_bytes': stat.st_size,
            'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat()
        }
    
    def _extract_with_pil(self, image_path: str) -> Dict[str, Any]:
        """Extract metadata using PIL/Pillow"""
        metadata = {'exif': {}, 'basic': {}}
        
        try:
            with Image.open(image_path) as img:
                # Basic image info
                metadata['basic'] = {
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'width': img.width,
                    'height': img.height
                }
                
                # EXIF data
                exif_data = img._getexif()
                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        
                        # Decode GPS info
                        if tag == 'GPSInfo':
                            gps_data = {}
                            for gps_tag_id, gps_value in value.items():
                                gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                                gps_data[gps_tag] = gps_value
                            metadata['exif'][tag] = gps_data
                        else:
                            # Handle different value types
                            if isinstance(value, bytes):
                                try:
                                    value = value.decode('utf-8', errors='ignore')
                                except:
                                    value = str(value)
                            elif isinstance(value, (tuple, list)) and len(value) == 2:
                                # Likely a rational number
                                if isinstance(value[0], int) and isinstance(value[1], int):
                                    if value[1] != 0:
                                        value = value[0] / value[1]
                                    else:
                                        value = value[0]
                            
                            metadata['exif'][tag] = value
                            
        except Exception as e:
            logger.error(f"PIL extraction failed: {e}")
            
        return metadata
    
    def _extract_with_exifread(self, image_path: str) -> Dict[str, Any]:
        """Extract metadata using exifread library"""
        metadata = {'exif': {}, 'maker_notes': {}}
        
        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f, details=True)
                
                for tag, value in tags.items():
                    tag_name = tag.replace('EXIF ', '').replace('Image ', '')
                    
                    # Convert value to appropriate type
                    if hasattr(value, 'values'):
                        if len(value.values) == 1:
                            val = value.values[0]
                        else:
                            val = value.values
                    else:
                        val = str(value)
                        
                    # Separate maker notes
                    if tag.startswith('MakerNote'):
                        metadata['maker_notes'][tag_name] = val
                    else:
                        metadata['exif'][tag_name] = val
                        
        except Exception as e:
            logger.error(f"exifread extraction failed: {e}")
            
        return metadata
    
    def _extract_with_piexif(self, image_path: str) -> Dict[str, Any]:
        """Extract metadata using piexif library"""
        metadata = {'exif': {}}
        
        try:
            exif_dict = piexif.load(image_path)
            
            # Process each IFD
            for ifd_name in ['0th', 'Exif', 'GPS', '1st']:
                if ifd_name in exif_dict:
                    for tag, value in exif_dict[ifd_name].items():
                        tag_name = piexif.TAGS[ifd_name].get(tag, {}).get('name', str(tag))
                        
                        # Decode bytes
                        if isinstance(value, bytes):
                            try:
                                value = value.decode('utf-8', errors='ignore').rstrip('\x00')
                            except:
                                value = str(value)
                                
                        metadata['exif'][tag_name] = value
                        
        except Exception as e:
            logger.error(f"piexif extraction failed: {e}")
            
        return metadata
    
    def _extract_xmp(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Extract XMP metadata"""
        xmp_data = {}
        
        try:
            with open(image_path, 'rb') as f:
                content = f.read()
                
            # Look for XMP packet
            xmp_start = content.find(b'<x:xmpmeta')
            if xmp_start == -1:
                xmp_start = content.find(b'<xmp:xmpmeta')
                
            if xmp_start != -1:
                xmp_end = content.find(b'</x:xmpmeta>', xmp_start)
                if xmp_end == -1:
                    xmp_end = content.find(b'</xmp:xmpmeta>', xmp_start)
                    
                if xmp_end != -1:
                    xmp_str = content[xmp_start:xmp_end + 12].decode('utf-8', errors='ignore')
                    
                    # Parse XMP XML
                    try:
                        root = ET.fromstring(xmp_str)
                        xmp_data = self._parse_xmp_tree(root)
                    except ET.ParseError as e:
                        logger.error(f"XMP parsing failed: {e}")
                        
        except Exception as e:
            logger.error(f"XMP extraction failed: {e}")
            
        return xmp_data if xmp_data else None
    
    def _parse_xmp_tree(self, element: ET.Element, prefix: str = '') -> Dict[str, Any]:
        """Recursively parse XMP XML tree"""
        data = {}
        
        # Get attributes
        for key, value in element.attrib.items():
            clean_key = key.split('}')[-1] if '}' in key else key
            data[prefix + clean_key] = value
            
        # Get text content
        if element.text and element.text.strip():
            data[prefix + 'value'] = element.text.strip()
            
        # Process children
        for child in element:
            child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            child_data = self._parse_xmp_tree(child, prefix + child_tag + '.')
            data.update(child_data)
            
        return data
    
    def _extract_lens_info(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and organize lens information"""
        lens_info = {}
        
        # Standard EXIF lens fields
        exif = metadata.get('exif', {})
        
        # Lens make and model
        lens_make = exif.get('LensMake', exif.get('Lens Make', ''))
        lens_model = exif.get('LensModel', exif.get('Lens Model', ''))
        
        if lens_make:
            lens_info['make'] = lens_make
        if lens_model:
            lens_info['model'] = lens_model
            
        # Lens specifications
        lens_info['focal_length'] = self._parse_focal_length(
            exif.get('FocalLength', exif.get('Focal Length', None))
        )
        
        lens_info['focal_length_35mm'] = self._parse_focal_length(
            exif.get('FocalLengthIn35mmFilm', exif.get('Focal Length in 35mm Film', None))
        )
        
        lens_info['max_aperture'] = self._parse_aperture(
            exif.get('MaxApertureValue', exif.get('Max Aperture Value', None))
        )
        
        # Lens ID (for electronic lenses)
        lens_id = exif.get('LensID', exif.get('Lens ID', None))
        if lens_id:
            lens_info['lens_id'] = lens_id
            
        # Lens serial number
        lens_serial = exif.get('LensSerialNumber', exif.get('Lens Serial Number', None))
        if lens_serial:
            lens_info['serial_number'] = str(lens_serial)
            
        # From maker notes
        maker_notes = metadata.get('maker_notes', {})
        if 'LensType' in maker_notes:
            lens_info['lens_type'] = maker_notes['LensType']
            
        # From XMP
        xmp = metadata.get('xmp', {})
        for key, value in xmp.items():
            if 'lens' in key.lower():
                lens_info[f'xmp_{key}'] = value
                
        return lens_info
    
    def _extract_camera_info(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract camera information"""
        camera_info = {}
        exif = metadata.get('exif', {})
        
        # Camera make and model
        camera_info['make'] = exif.get('Make', exif.get('Camera Make', ''))
        camera_info['model'] = exif.get('Model', exif.get('Camera Model', ''))
        
        # Serial number
        camera_info['serial_number'] = exif.get('BodySerialNumber', 
                                               exif.get('Serial Number', ''))
        
        # Firmware
        camera_info['firmware'] = exif.get('Software', exif.get('Firmware', ''))
        
        return camera_info
    
    def _extract_shooting_info(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract shooting parameters"""
        shooting_info = {}
        exif = metadata.get('exif', {})
        
        # Exposure settings
        shooting_info['aperture'] = self._parse_aperture(
            exif.get('FNumber', exif.get('F-Number', exif.get('ApertureValue', None)))
        )
        
        shooting_info['shutter_speed'] = self._parse_shutter_speed(
            exif.get('ExposureTime', exif.get('Exposure Time', None))
        )
        
        shooting_info['iso'] = self._parse_iso(
            exif.get('ISOSpeedRatings', exif.get('ISO Speed Ratings', 
                    exif.get('ISO', None)))
        )
        
        # Focus information  
        shooting_info['focus_distance'] = exif.get('FocusDistance', 
                                                  exif.get('Subject Distance', None))
        shooting_info['focus_mode'] = exif.get('FocusMode', exif.get('Focus Mode', None))
        
        # Other settings
        shooting_info['white_balance'] = exif.get('WhiteBalance', 
                                                 exif.get('White Balance', None))
        shooting_info['metering_mode'] = exif.get('MeteringMode', 
                                                 exif.get('Metering Mode', None))
        
        # Date and time
        date_taken = exif.get('DateTimeOriginal', exif.get('DateTime', None))
        if date_taken:
            shooting_info['date_taken'] = str(date_taken)
            
        return shooting_info
    
    def _detect_manual_lens(self, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect manual/adapted lenses from metadata patterns"""
        
        # Check for missing electronic lens data
        lens_info = metadata.get('lens_info', {})
        
        if not lens_info.get('model') and not lens_info.get('lens_id'):
            # No electronic lens detected, check other clues
            
            # Check user comments and image description
            exif = metadata.get('exif', {})
            comments = []
            
            for field in ['UserComment', 'ImageDescription', 'Artist', 'Copyright']:
                value = exif.get(field, '')
                if value:
                    comments.append(str(value).lower())
                    
            # Check XMP
            xmp = metadata.get('xmp', {})
            for key, value in xmp.items():
                if any(term in key.lower() for term in ['description', 'title', 'subject']):
                    comments.append(str(value).lower())
                    
            # Search for lens patterns
            detected_lens = None
            for lens_type, patterns in self.manual_lens_patterns.items():
                for pattern in patterns:
                    for comment in comments:
                        if pattern in comment:
                            # Extract more details if possible
                            detected_lens = {
                                'type': lens_type,
                                'pattern_matched': pattern,
                                'source': comment,
                                'confidence': 0.8
                            }
                            
                            # Try to extract focal length and aperture
                            import re
                            
                            # Focal length pattern: 50mm, 85mm, etc
                            focal_match = re.search(r'(\d+)\s*mm', comment)
                            if focal_match:
                                detected_lens['focal_length'] = int(focal_match.group(1))
                                
                            # Aperture pattern: f/1.4, f1.8, etc
                            aperture_match = re.search(r'f/?(\d+\.?\d*)', comment)
                            if aperture_match:
                                detected_lens['max_aperture'] = float(aperture_match.group(1))
                                
                            return detected_lens
                            
        # Check for adapter signatures
        adapter_signatures = ['adapter', 'adapted', 'manual focus', 'mf lens']
        for sig in adapter_signatures:
            for field in ['UserComment', 'ImageDescription']:
                value = str(exif.get(field, '')).lower()
                if sig in value:
                    return {
                        'type': 'adapted',
                        'signature': sig,
                        'source': field,
                        'confidence': 0.6
                    }
                    
        return None
    
    def _parse_focal_length(self, value: Any) -> Optional[float]:
        """Parse focal length value"""
        if not value:
            return None
            
        try:
            if isinstance(value, (list, tuple)) and len(value) == 2:
                return float(value[0]) / float(value[1]) if value[1] != 0 else None
            elif isinstance(value, str):
                # Remove 'mm' and parse
                value = value.replace('mm', '').strip()
                return float(value)
            else:
                return float(value)
        except:
            return None
            
    def _parse_aperture(self, value: Any) -> Optional[float]:
        """Parse aperture value"""
        if not value:
            return None
            
        try:
            if isinstance(value, (list, tuple)) and len(value) == 2:
                # APEX value - convert to f-number
                apex = float(value[0]) / float(value[1]) if value[1] != 0 else 0
                return round(2 ** (apex / 2), 1)
            elif isinstance(value, str):
                # Remove 'f/' and parse
                value = value.replace('f/', '').replace('F/', '').strip()
                return float(value)
            else:
                return float(value)
        except:
            return None
            
    def _parse_shutter_speed(self, value: Any) -> Optional[str]:
        """Parse shutter speed value"""
        if not value:
            return None
            
        try:
            if isinstance(value, (list, tuple)) and len(value) == 2:
                if value[0] == 1:
                    return f"1/{value[1]}"
                else:
                    return f"{value[0]}/{value[1]}"
            else:
                return str(value)
        except:
            return None
            
    def _parse_iso(self, value: Any) -> Optional[int]:
        """Parse ISO value"""
        if not value:
            return None
            
        try:
            if isinstance(value, (list, tuple)):
                return int(value[0])
            else:
                return int(value)
        except:
            return None
    
    def get_lens_identifier(self, metadata: Dict[str, Any]) -> str:
        """Generate lens identifier from metadata"""
        lens_info = metadata.get('lens_info', {})
        
        # For electronic lenses
        if lens_info.get('model'):
            identifier = lens_info['model']
            if lens_info.get('serial_number'):
                identifier += f"_SN{lens_info['serial_number']}"
            return identifier
            
        # For manual lenses
        manual = lens_info.get('manual_lens_detected', {})
        if manual:
            parts = [manual.get('type', 'unknown')]
            if manual.get('focal_length'):
                parts.append(f"{manual['focal_length']}mm")
            if manual.get('max_aperture'):
                parts.append(f"f{manual['max_aperture']}")
            return '_'.join(parts)
            
        # Fallback
        return 'unknown_lens'
