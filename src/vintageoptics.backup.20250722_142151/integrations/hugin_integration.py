# src/vintageoptics/integrations/hugin_integration.py

import subprocess
import os
import numpy as np
from typing import List, Dict, Tuple

class HuginToolsIntegration:
    """Integration with Hugin's lens calibration tools"""
    
    def __init__(self):
        self.check_tools = ["calibrate_lens_gui", "tca_correct", "fulla", "nona"]
        self._verify_installation()
    
    def calibrate_lens_tca(self, image_pairs: List[Tuple[str, str]]) -> Dict:
        """Use tca_correct for precise chromatic aberration measurement"""
        # Create control point file
        pto_file = self._create_pto_for_tca(image_pairs)
        
        # Run tca_correct
        cmd = ["tca_correct", "-o", "tca_", pto_file]
        result = subprocess.run(cmd, capture_output=True)
        
        # Parse results
        return self._parse_tca_results(result.stdout.decode())
    
    def correct_with_fulla(self, image_path: str, params: Dict) -> str:
        """Use fulla for lens correction"""
        output_path = image_path.replace('.', '_corrected.')
        
        cmd = [
            "fulla",
            f"--red={params.get('tca_red', '1.0')}",
            f"--blue={params.get('tca_blue', '1.0')}",
            f"--vignetting={params.get('vignetting', '1.0:0:1.0:0')}",
            f"--distortion={params.get('distortion', '0:0:0:1.0')}",
            "-o", output_path,
            image_path
        ]
        
        subprocess.run(cmd, check=True)
        return output_path