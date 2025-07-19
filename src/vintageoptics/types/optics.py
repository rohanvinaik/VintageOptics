# src/vintageoptics/types/optics.py

from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class OpticalParameters:
    """Optical correction parameters"""
    distortion_k1: float = 0.0
    distortion_k2: float = 0.0
    distortion_k3: float = 0.0
    distortion_p1: float = 0.0
    distortion_p2: float = 0.0
    chromatic_red: float = 1.0
    chromatic_blue: float = 1.0
    vignetting_a1: float = 0.0
    vignetting_a2: float = 0.0
    vignetting_a3: float = 0.0

@dataclass
class LensProfile:
    """Complete lens profile information"""
    lens_id: str
    manufacturer: str
    model: str
    lens_type: str
    optical_params: OpticalParameters
    metadata: Dict
    confidence: float = 0.5
