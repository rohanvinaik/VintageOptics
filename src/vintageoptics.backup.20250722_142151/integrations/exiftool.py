# src/vintageoptics/integrations/exiftool.py
"""
ExifTool integration for advanced metadata extraction and manipulation.
Provides comprehensive access to metadata that standard libraries might miss.
"""

import subprocess
import json
import os
import tempfile
from typing import Dict, List, Optional, Any, Union
import logging
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


class ExifToolIntegration:
    """Integration with ExifTool for advanced metadata operations"""
    
    def __init__(self, exiftool_path: Optional[str] = None):
        self.exiftool_path = exiftool_path or self._find_exiftool()
        self.available = self._check_availability()
        
        if not self.available:
            logger.warning("ExifTool not found. Install from https://exiftool.org")
    
    def _find_exiftool(self) -> Optional[str]:
        """Find ExifTool executable in system"""
        
        # Check common locations
        possible_paths = [
            'exiftool',  # In PATH
            '/usr/local/bin/exiftool',
            '/usr/bin/exiftool',
            '/opt/homebrew/bin/exiftool',  # macOS with Homebrew
            'C:\\Windows\\exiftool.exe',
            'C:\\Program Files\\ExifTool\\exiftool.exe'
        ]
        
        for path in possible_paths:
            if shutil.which(path):
                return path
        
        return None
    
    def _check_availability(self) -> bool:
        """Check if ExifTool is available and working"""
        
        if not self.exiftool_path:
            return False
        
        try:
            result = subprocess.run(
                [self.exiftool_path, '-ver'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                version = result.stdout.strip()
                logger.info(f"ExifTool version {version} found")
                return True
                
        except Exception as e:
            logger.error(f"ExifTool check failed: {e}")
        
        return False
    
    def extract_all_metadata(self, image_path: str, 
                           include_binary: bool = False) -> Dict[str, Any]:
        """Extract all metadata from image using ExifTool"""
        
        if not self.available:
            return {}
        
        try:
            # Build command
            cmd = [
                self.exiftool_path,
                '-json',
                '-extractEmbedded',
                '-ee',  # Extract embedded
                '-a',   # Allow duplicates
                '-u',   # Show unknown tags
                '-g',   # Group output
                '-struct',  # Export structured data
                '-sort'  # Sort output
            ]
            
            if include_binary:
                cmd.append('-b')  # Include binary data
            
            cmd.append(image_path)
            
            # Execute
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Parse JSON output
                metadata_list = json.loads(result.stdout)
                if metadata_list:
                    return self._process_metadata(metadata_list[0])
            else:
                logger.error(f"ExifTool error: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
        
        return {}
    
    def _process_metadata(self, raw_metadata: Dict) -> Dict[str, Any]:
        """Process and organize raw ExifTool output"""
        
        processed = {
            'file': {},
            'exif': {},
            'xmp': {},
            'iptc': {},
            'maker_notes': {},
            'composite': {},
            'lens': {},
            'camera': {},
            'other': {}
        }
        
        # Process each metadata group
        for key, value in raw_metadata.items():
            if ':' in key:
                group, tag = key.split(':', 1)
                group_lower = group.lower()
                
                # Categorize by group
                if group_lower == 'file':
                    processed['file'][tag] = value
                elif group_lower == 'exif':
                    processed['exif'][tag] = value
                    
                    # Extract lens-specific EXIF tags
                    if any(lens_term in tag.lower() for lens_term in 
                          ['lens', 'focal', 'aperture']):
                        processed['lens'][tag] = value
                        
                elif group_lower == 'xmp':
                    processed['xmp'][tag] = value
                elif group_lower == 'iptc':
                    processed['iptc'][tag] = value
                elif 'makernote' in group_lower:
                    processed['maker_notes'][tag] = value
                elif group_lower == 'composite':
                    processed['composite'][tag] = value
                else:
                    processed['other'][f"{group}:{tag}"] = value
            else:
                # No group specified
                processed['other'][key] = value
        
        # Extract specific camera info
        for key in ['Make', 'Model', 'SerialNumber', 'InternalSerialNumber']:
            if key in processed['exif']:
                processed['camera'][key] = processed['exif'][key]
        
        return processed
    
    def extract_lens_info(self, image_path: str) -> Dict[str, Any]:
        """Extract comprehensive lens information"""
        
        if not self.available:
            return {}
        
        # Specific lens-related tags to extract
        lens_tags = [
            'LensMake',
            'LensModel', 
            'LensID',
            'LensSerialNumber',
            'LensInfo',
            'LensType',
            'LensSpec',
            'LensFocalRange',
            'LensMaxApertureRange',
            'LensMount',
            'FocalLength',
            'FocalLengthIn35mmFormat',
            'MaxApertureValue',
            'MinApertureValue',
            'FNumber',
            'FocalLength35efl',
            # Maker-specific lens tags
            'Canon:LensType',
            'Canon:LensModel',
            'Canon:LensSerialNumber',
            'Canon:InternalLensSerialNumber',
            'Nikon:LensID',
            'Nikon:LensFStops',
            'Nikon:LensType',
            'Nikon:LensSerialNumber',
            'Sony:LensType',
            'Sony:LensSpec',
            'Sony:LensMount',
            'Olympus:LensType',
            'Olympus:LensSerialNumber',
            'Olympus:LensModel',
            'Pentax:LensType',
            'Pentax:LensInfo',
            'Fujifilm:LensSerialNumber',
            'Fujifilm:LensModel',
            'Panasonic:LensType',
            'Panasonic:LensSerialNumber'
        ]
        
        try:
            # Build command to extract specific tags
            cmd = [
                self.exiftool_path,
                '-json',
                '-extractEmbedded'
            ]
            
            # Add each tag
            for tag in lens_tags:
                cmd.extend(['-' + tag])
            
            cmd.append(image_path)
            
            # Execute
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                metadata_list = json.loads(result.stdout)
                if metadata_list:
                    return self._process_lens_info(metadata_list[0])
                    
        except Exception as e:
            logger.error(f"Lens info extraction failed: {e}")
        
        return {}
    
    def _process_lens_info(self, raw_lens_data: Dict) -> Dict[str, Any]:
        """Process lens-specific metadata"""
        
        lens_info = {
            'make': None,
            'model': None,
            'id': None,
            'serial_number': None,
            'mount': None,
            'type': None,
            'focal_length': None,
            'focal_range': None,
            'aperture': None,
            'aperture_range': None,
            'other': {}
        }
        
        # Process each field
        for key, value in raw_lens_data.items():
            key_lower = key.lower()
            
            if 'lensmake' in key_lower:
                lens_info['make'] = value
            elif 'lensmodel' in key_lower:
                lens_info['model'] = value
            elif 'lensid' in key_lower:
                lens_info['id'] = value
            elif 'lensserialnumber' in key_lower:
                lens_info['serial_number'] = str(value)
            elif 'lensmount' in key_lower:
                lens_info['mount'] = value
            elif 'lenstype' in key_lower:
                lens_info['type'] = value
            elif 'focallength' in key_lower and '35' not in key_lower:
                # Parse focal length
                lens_info['focal_length'] = self._parse_focal_length(value)
            elif 'lensfocalrange' in key_lower:
                lens_info['focal_range'] = self._parse_focal_range(value)
            elif 'fnumber' in key_lower:
                lens_info['aperture'] = self._parse_aperture(value)
            elif 'aperturerange' in key_lower:
                lens_info['aperture_range'] = self._parse_aperture_range(value)
            else:
                lens_info['other'][key] = value
        
        return lens_info
    
    def write_metadata(self, image_path: str, metadata: Dict[str, Any],
                      create_backup: bool = True) -> bool:
        """Write metadata to image file"""
        
        if not self.available:
            return False
        
        try:
            # Build command
            cmd = [self.exiftool_path]
            
            if not create_backup:
                cmd.append('-overwrite_original')
            
            # Add metadata fields
            for key, value in metadata.items():
                if value is not None:
                    cmd.append(f'-{key}={value}')
            
            cmd.append(image_path)
            
            # Execute
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info(f"Metadata written successfully to {image_path}")
                return True
            else:
                logger.error(f"Metadata write failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Failed to write metadata: {e}")
        
        return False
    
    def add_lens_correction_metadata(self, image_path: str,
                                   correction_params: Dict[str, Any]) -> bool:
        """Add lens correction parameters to image metadata"""
        
        metadata = {}
        
        # Add to XMP namespace
        if 'distortion_k1' in correction_params:
            metadata['XMP:DistortionCorrectionK1'] = correction_params['distortion_k1']
        if 'distortion_k2' in correction_params:
            metadata['XMP:DistortionCorrectionK2'] = correction_params['distortion_k2']
        if 'chromatic_red' in correction_params:
            metadata['XMP:ChromaticAberrationRed'] = correction_params['chromatic_red']
        if 'chromatic_blue' in correction_params:
            metadata['XMP:ChromaticAberrationBlue'] = correction_params['chromatic_blue']
        
        # Add processing info
        metadata['XMP:ProcessingSoftware'] = 'VintageOptics'
        metadata['XMP:LensCorrectionApplied'] = 'True'
        
        return self.write_metadata(image_path, metadata)
    
    def extract_raw_metadata(self, image_path: str) -> Optional[bytes]:
        """Extract raw metadata block for analysis"""
        
        if not self.available:
            return None
        
        try:
            cmd = [
                self.exiftool_path,
                '-b',  # Binary output
                '-metadata',  # All metadata
                image_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return result.stdout
                
        except Exception as e:
            logger.error(f"Raw metadata extraction failed: {e}")
        
        return None
    
    def _parse_focal_length(self, value: str) -> Optional[float]:
        """Parse focal length value"""
        try:
            # Handle various formats: "50 mm", "50.0mm", "50"
            value = str(value).replace('mm', '').strip()
            return float(value)
        except:
            return None
    
    def _parse_focal_range(self, value: str) -> Optional[Tuple[float, float]]:
        """Parse focal range like '24-70mm'"""
        try:
            import re
            match = re.search(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)', str(value))
            if match:
                return (float(match.group(1)), float(match.group(2)))
        except:
            pass
        return None
    
    def _parse_aperture(self, value: str) -> Optional[float]:
        """Parse aperture value"""
        try:
            # Handle f/2.8, 2.8, etc
            value = str(value).replace('f/', '').replace('F/', '').strip()
            return float(value)
        except:
            return None
    
    def _parse_aperture_range(self, value: str) -> Optional[Tuple[float, float]]:
        """Parse aperture range"""
        try:
            import re
            # Handle formats like "f/2.8-f/5.6" or "2.8-5.6"
            numbers = re.findall(r'\d+\.?\d*', str(value))
            if len(numbers) >= 2:
                return (float(numbers[0]), float(numbers[1]))
        except:
            pass
        return None
    
    def create_xmp_sidecar(self, image_path: str, metadata: Dict[str, Any]) -> bool:
        """Create XMP sidecar file with metadata"""
        
        xmp_path = Path(image_path).with_suffix('.xmp')
        
        try:
            # Build XMP content
            xmp_content = self._build_xmp_content(metadata)
            
            # Write to file
            with open(xmp_path, 'w', encoding='utf-8') as f:
                f.write(xmp_content)
            
            logger.info(f"Created XMP sidecar: {xmp_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create XMP sidecar: {e}")
            return False
    
    def _build_xmp_content(self, metadata: Dict[str, Any]) -> str:
        """Build XMP XML content"""
        
        xmp = '''<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="VintageOptics">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
        xmlns:dc="http://purl.org/dc/elements/1.1/"
        xmlns:xmp="http://ns.adobe.com/xap/1.0/"
        xmlns:aux="http://ns.adobe.com/exif/1.0/aux/"
        xmlns:vintageoptics="http://vintageoptics.org/xmp/1.0/">
'''
        
        # Add metadata fields
        for key, value in metadata.items():
            if value is not None:
                # Escape XML special characters
                value_escaped = str(value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                xmp += f'      <vintageoptics:{key}>{value_escaped}</vintageoptics:{key}>\n'
        
        xmp += '''    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>'''
        
        return xmp
