#!/usr/bin/env python3

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("Testing basic VintageOptics imports...")
    
    # Test individual components first
    print("1. Testing types...")
    from vintageoptics.types import DepthMap, DepthLayer
    print("   ✅ Types OK")
    
    print("2. Testing detection...")
    from vintageoptics.detection import UnifiedLensDetector
    print("   ✅ Detection OK")
    
    print("3. Testing physics...")
    from vintageoptics.physics import OpticsEngine
    print("   ✅ Physics OK")
    
    print("4. Testing core config...")
    from vintageoptics.core import ConfigManager
    print("   ✅ Core config OK")
    
    print("5. Testing main pipeline...")
    from vintageoptics.core.pipeline import VintageOpticsPipeline
    print("   ✅ Pipeline import OK")
    
    print("6. Testing main package...")
    import vintageoptics
    print(f"   ✅ Main package OK - Version: {vintageoptics.__version__}")
    
    print("\n🎉 All basic imports successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
