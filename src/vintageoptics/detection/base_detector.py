# src/vintageoptics/detection/base_detector.py

from abc import ABC, abstractmethod
from typing import Dict, Optional

class BaseLensDetector(ABC):
    """Abstract base class for lens detectors"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    @abstractmethod
    def detect(self, image_data) -> Optional[Dict]:
        """Detect lens from image data"""
        pass
