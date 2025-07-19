#!/usr/bin/env python3

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing VintageOptics step by step...")

try:
    print("Step 1: Testing numpy...")
    import numpy as np
    print("✅ numpy OK")
    
    print("Step 2: Testing opencv...")
    import cv2
    print("✅ opencv OK")
    
    print("Step 3: Testing types module...")
    from vintageoptics.types.depth import DepthMap
    print("✅ types OK")
    
    print("Step 4: Testing detection base...")
    from vintageoptics.detection.base_detector import BaseLensDetector
    print("✅ detection base OK")
    
    print("Step 5: Testing metadata extractor...")
    from vintageoptics.detection.metadata_extractor import MetadataExtractor
    print("✅ metadata extractor OK")
    
    print("Step 6: Testing unified detector...")
    from vintageoptics.detection import UnifiedLensDetector
    print("✅ unified detector OK")
    
    print("Step 7: Testing pipeline...")
    from vintageoptics.core.pipeline import VintageOpticsPipeline
    print("✅ pipeline OK")
    
    print("\n🎉 All imports successful!")
    
except Exception as e:
    print(f"❌ Error at step: {e}")
    import traceback
    traceback.print_exc()
