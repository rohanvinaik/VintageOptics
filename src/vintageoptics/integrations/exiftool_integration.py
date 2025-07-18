# src/vintageoptics/integrations/exiftool_integration.py

import subprocess
import json
import tempfile
from typing import Dict, Optional

class ExifToolWrapper:
    """Wrapper for Phil Harvey's ExifTool - the most comprehensive metadata extractor"""
    
    def __init__(self, executable_path: str = "exiftool"):
        self.executable = executable_path
        self._check_installation()
    
    def _check_installation(self):
        """Verify ExifTool is installed"""
        try:
            subprocess.run([self.executable, "-ver"], capture_output=True, check=True)
        except:
            raise RuntimeError("ExifTool not found. Please install from https://exiftool.org/")
    
    def extract_all_metadata(self, image_path: str) -> Dict:
        """Extract ALL metadata including maker notes"""
        cmd = [
            self.executable,
            "-j",  # JSON output
            "-binary",  # Extract binary data
            "-extractEmbedded",  # Extract embedded images
            "-ee",  # Extract embedded data
            "-api", "largefilesupport=1",
            "-G",  # Show group names
            "-s",  # Short tag names
            "-struct",  # Extract structured data
            image_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            metadata = json.loads(result.stdout)[0]
            return self._parse_metadata_groups(metadata)
        else:
            raise RuntimeError(f"ExifTool error: {result.stderr}")
    
    def _parse_metadata_groups(self, metadata: Dict) -> Dict:
        """Organize metadata by groups"""
        organized = {
            'EXIF': {},
            'MakerNotes': {},
            'XMP': {},
            'IPTC': {},
            'Composite': {},
            'Other': {}
        }
        
        for key, value in metadata.items():
            if key.startswith('EXIF:'):
                organized['EXIF'][key[5:]] = value
            elif key.startswith('MakerNotes:'):
                organized['MakerNotes'][key[11:]] = value
            elif key.startswith('XMP:'):
                organized['XMP'][key[4:]] = value
            elif key.startswith('IPTC:'):
                organized['IPTC'][key[5:]] = value
            elif key.startswith('Composite:'):
                organized['Composite'][key[10:]] = value
            else:
                organized['Other'][key] = value
        
        return organized