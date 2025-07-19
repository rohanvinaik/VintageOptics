# src/vintageoptics/types/optics.py

from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple
from enum import Enum
import numpy as np

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

@dataclass
class ImageMetadata:
    """Image metadata and technical information"""
    width: int
    height: int
    channels: int = 3
    bit_depth: int = 8
    color_space: str = "sRGB"
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    lens_model: Optional[str] = None
    focal_length: Optional[float] = None
    aperture: Optional[float] = None
    iso: Optional[int] = None
    shutter_speed: Optional[float] = None
    timestamp: Optional[str] = None
    exif_data: Optional[Dict[str, Any]] = None

@dataclass
class LensParameters:
    """Physical lens parameters"""
    focal_length: float
    aperture: float
    min_aperture: float = 22.0
    max_aperture: float = 1.2
    focus_distance: Optional[float] = None
    image_stabilization: bool = False
    
@dataclass
class OpticalDefect:
    """Represents a detected optical defect"""
    type: str  # 'dust', 'scratch', 'fungus', 'haze', etc.
    severity: float  # 0.0 to 1.0
    location: Tuple[float, float]  # Normalized coordinates
    size: float  # Size in pixels or area
    confidence: float = 0.9

class LensType(Enum):
    """Lens type enumeration"""
    VINTAGE_MANUAL = "vintage_manual"
    MODERN_ELECTRONIC = "modern_electronic"
    ADAPTED_VINTAGE = "adapted_vintage"
    UNKNOWN = "unknown"
