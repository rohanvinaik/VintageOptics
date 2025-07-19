#!/usr/bin/env python3

import sys
sys.path.append('src')

try:
    print("Testing minimal import...")
    
    # Test the most basic import
    from vintageoptics.detection.base_detector import BaseLensDetector
    print("✅ Base detector imported successfully")
    
    # Test configuration
    from vintageoptics.core import ConfigManager
    print("✅ Config manager imported successfully")
    
    print("✅ Basic imports work!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
