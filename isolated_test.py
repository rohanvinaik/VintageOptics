#!/usr/bin/env python3

import sys
sys.path.append('src')

print("Testing individual modules...")

try:
    print("1. Testing config manager...")
    from vintageoptics.core.config_manager import ConfigManager
    print("✅ ConfigManager OK")
    
    print("2. Testing performance monitor...")
    from vintageoptics.core.performance_monitor import PerformanceMonitor
    print("✅ PerformanceMonitor OK")
    
    print("3. Testing core init...")
    from vintageoptics.core import ImageData, ProcessingResult
    print("✅ Core init OK")
    
    print("4. Testing types...")
    from vintageoptics.types.depth import DepthMap
    print("✅ Types OK")
    
    print("5. Testing detection base...")
    from vintageoptics.detection.base_detector import BaseLensDetector
    print("✅ Detection base OK")
    
    print("6. Testing metadata extractor...")
    from vintageoptics.detection.metadata_extractor import MetadataExtractor
    print("✅ Metadata extractor OK")
    
    print("7. Testing unified detector...")
    from vintageoptics.detection import UnifiedLensDetector
    print("✅ Unified detector OK")
    
    print("8. Testing pipeline...")
    from vintageoptics.core.pipeline import VintageOpticsPipeline
    print("✅ Pipeline OK")
    
    print("\n🎉 All individual modules imported successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
