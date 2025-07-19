#!/usr/bin/env python3
"""
Direct test of error_orthogonality module in isolation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Test imports one by one
print("Testing imports...")

try:
    import numpy as np
    print("✓ numpy imported")
except Exception as e:
    print(f"✗ numpy import failed: {e}")

try:
    import cv2
    print("✓ cv2 imported")
except Exception as e:
    print(f"✗ cv2 import failed: {e}")

try:
    from scipy import signal, stats
    print("✓ scipy imported")
except Exception as e:
    print(f"✗ scipy import failed: {e}")

try:
    from sklearn.decomposition import FastICA
    print("✓ sklearn imported")
except Exception as e:
    print(f"✗ sklearn import failed: {e}")

print("\nTesting VintageOptics imports...")

try:
    # Try importing just the error_orthogonality module directly
    import vintageoptics.analysis.error_orthogonality as eo
    print("✓ error_orthogonality module imported")
    
    # Check what's in the module
    print("\nAvailable classes and functions:")
    for item in dir(eo):
        if not item.startswith('_'):
            print(f"  - {item}")
    
    # Try creating instances
    print("\nTesting class instantiation...")
    
    analyzer = eo.OrthogonalErrorAnalyzer()
    print("✓ OrthogonalErrorAnalyzer created")
    
    corrector = eo.HybridErrorCorrector()
    print("✓ HybridErrorCorrector created")
    
    # Test basic functionality
    print("\nTesting basic functionality...")
    
    # Create simple test images
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Test error characterization
    profile = analyzer.characterize_analog_errors(test_img, {})
    print(f"✓ Analog error profile created: {profile.source_type}")
    
    profile = analyzer.characterize_digital_errors(test_img, {})
    print(f"✓ Digital error profile created: {profile.source_type}")
    
    print("\nAll tests passed! Module is working correctly.")
    
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("\nChecking import chain...")
    
    # Try to identify which import is failing
    try:
        import vintageoptics
        print("  ✓ vintageoptics package found")
    except:
        print("  ✗ vintageoptics package not found")
    
    try:
        import vintageoptics.analysis
        print("  ✓ vintageoptics.analysis package found")
    except Exception as e:
        print(f"  ✗ vintageoptics.analysis import failed: {e}")
    
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
