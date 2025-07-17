#!/usr/bin/env python3

import sys
sys.path.append('src')

def test_step(step_name, import_func):
    try:
        import_func()
        print(f"✅ {step_name}")
        return True
    except Exception as e:
        print(f"❌ {step_name}: {e}")
        return False

print("Step-by-step import test:")

# Test 1: Basic Python modules
test_step("NumPy", lambda: __import__('numpy'))
test_step("OpenCV", lambda: __import__('cv2'))

# Test 2: Types modules  
test_step("Depth Types", lambda: __import__('vintageoptics.types.depth'))
test_step("Optics Types", lambda: __import__('vintageoptics.types.optics'))
test_step("IO Types", lambda: __import__('vintageoptics.types.io'))

# Test 3: Core utilities (non-circular)
test_step("Config Manager", lambda: __import__('vintageoptics.core.config_manager'))
test_step("Performance Monitor", lambda: __import__('vintageoptics.core.performance_monitor'))

# Test 4: Detection modules
test_step("Base Detector", lambda: __import__('vintageoptics.detection.base_detector'))
test_step("Metadata Extractor", lambda: __import__('vintageoptics.detection.metadata_extractor'))

# Test 5: Other modules
test_step("Physics Engine", lambda: __import__('vintageoptics.physics.optics_engine'))
test_step("Depth Analyzer", lambda: __import__('vintageoptics.depth.depth_analyzer'))

print("\nTesting potentially problematic imports:")
test_step("Detection Module", lambda: __import__('vintageoptics.detection'))
test_step("Core Module", lambda: __import__('vintageoptics.core'))
test_step("Pipeline", lambda: __import__('vintageoptics.core.pipeline'))
