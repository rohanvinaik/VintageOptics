# src/vintageoptics/integrations/midas_integration.py

import torch
import cv2
import numpy as np
from typing import Optional, Dict

class MiDaSDepthEstimator:
    """Integration with Intel MiDaS for high-quality monocular depth estimation"""
    
    def __init__(self, model_type: str = "DPT_Large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.model = None
        self.transform = None
        self._load_model()
    
    def _load_model(self):
        """Load MiDaS model"""
        # Use torch hub for easy access
        if self.model_type == "DPT_Large":
            self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        elif self.model_type == "DPT_Hybrid":
            self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        else:
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.default_transform
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth map using MiDaS"""
        # Prepare image
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)
        
        # Predict depth
        with torch.no_grad():
            prediction = self.model(input_batch)
            
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Normalize to [0, 1]
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        
        return depth_map