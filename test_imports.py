#!/usr/bin/env python3
"""
<<<<<<< HEAD
Test script to verify VintageOptics modules can be imported
=======
Test script for VintageOptics imports
>>>>>>> eb4b6e0da5af15d0abb016c55d48b27cdff961c3
"""

import sys
import os

<<<<<<< HEAD
# Add VintageOptics to path
vintage_optics_path = os.path.join(os.path.dirname(__file__), 'VintageOptics', 'src')
sys.path.insert(0, vintage_optics_path)

print("Testing VintageOptics imports...")
print(f"Python path includes: {vintage_optics_path}")
print()

# Test imports
modules_to_test = [
    ("Core Pipeline", "vintageoptics.core.pipeline", ["VintageOpticsPipeline", "PipelineConfig"]),
    ("Types", "vintageoptics.types.optics", ["ProcessingMode", "LensProfile"]),
    ("Hyperdimensional", "vintageoptics.hyperdimensional", ["HyperdimensionalLensAnalyzer"]),
    ("Physics Engine", "vintageoptics.physics.optics_engine", ["OpticsEngine"]),
    ("Detection", "vintageoptics.detection", ["UnifiedDetector"]),
    ("Analysis", "vintageoptics.analysis", ["LensCharacterizer"]),
]

successful = []
failed = []

for name, module, classes in modules_to_test:
    try:
        print(f"Testing {name}...", end=" ")
        exec(f"from {module} import {', '.join(classes)}")
        print("✓ SUCCESS")
        successful.append(name)
    except ImportError as e:
        print(f"✗ FAILED: {e}")
        failed.append((name, str(e)))
    except Exception as e:
        print(f"✗ ERROR: {e}")
        failed.append((name, str(e)))

print("\n" + "="*50)
print(f"Summary: {len(successful)}/{len(modules_to_test)} modules loaded successfully")

if failed:
    print("\nFailed modules:")
    for name, error in failed:
        print(f"  - {name}: {error}")
    print("\nTo fix these issues, you may need to:")
    print("  1. Install missing dependencies: pip install -r requirements/base.txt")
    print("  2. Ensure all __init__.py files are present")
    print("  3. Check for circular imports")
else:
    print("\nAll modules loaded successfully! ✨")
    print("The real VintageOptics pipeline should work in the GUI.")
=======
# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("Testing VintageOptics imports...")
    
    # Test core imports
    from vintageoptics.core.pipeline import VintageOpticsPipeline, ProcessingMode, ProcessingRequest
    print("✅ Core pipeline imports successful")
    
    # Test detection imports
    from vintageoptics.detection import UnifiedLensDetector
    print("✅ Detection imports successful")
    
    # Test physics imports
    from vintageoptics.physics import OpticsEngine
    print("✅ Physics imports successful")
    
    # Test depth imports
    from vintageoptics.depth import DepthAwareProcessor
    print("✅ Depth processing imports successful")
    
    # Test synthesis imports
    from vintageoptics.synthesis import LensSynthesizer
    print("✅ Synthesis imports successful")
    
    # Test statistical imports
    from vintageoptics.statistical import AdaptiveCleanup
    print("✅ Statistical cleanup imports successful")
    
    # Test analysis imports
    from vintageoptics.analysis import QualityAnalyzer
    print("✅ Analysis imports successful")
    
    # Test main package import
    import vintageoptics
    print(f"✅ Main package import successful - Version: {vintageoptics.__version__}")
    
    # Test pipeline initialization
    pipeline = VintageOpticsPipeline("config/default.yaml")
    print("✅ Pipeline initialization successful")
    
    print("\n🎉 All imports successful! VintageOptics is ready to use.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
>>>>>>> eb4b6e0da5af15d0abb016c55d48b27cdff961c3
