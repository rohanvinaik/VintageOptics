# src/vintageoptics/detection/base_detector.py

from abc import ABC, abstractmethod
from typing import Dict, Optional

class BaseLensDetector(ABC):
    """Abstract base class for lens detectors"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
    
    @abstractmethod
    def detect(self, image_data) -> float:
        """Detect lens from image data and return probability score"""
        pass
