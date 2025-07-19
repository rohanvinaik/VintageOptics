#!/usr/bin/env python3
"""
Test script to verify VintageOptics modules can be imported
"""

import sys
import os

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
