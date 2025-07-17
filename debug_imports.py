#!/usr/bin/env python3
"""
VintageOptics Import Debug Script
This script helps identify and fix import issues step by step.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_import(module_name, description):
    """Test a single import and report result"""
    try:
        __import__(module_name)
        print(f"✅ {description}")
        return True
    except Exception as e:
        print(f"❌ {description}: {e}")
        return False

def main():
    print("🔍 VintageOptics Import Diagnostics")
    print("=" * 50)
    
    # Test basic Python packages
    print("\n📦 Testing Python Dependencies:")
    deps = [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("scipy", "SciPy"),
        ("sklearn", "Scikit-learn"),
        ("yaml", "PyYAML")
    ]
    
    all_deps_ok = True
    for dep, name in deps:
        if not test_import(dep, f"{name}"):
            all_deps_ok = False
    
    if not all_deps_ok:
        print("\n⚠️  Some dependencies are missing. Install with:")
        print("pip install numpy opencv-python scipy scikit-learn PyYAML")
        return False
    
    # Test VintageOptics modules
    print("\n🔧 Testing VintageOptics Modules:")
    
    # Test in dependency order
    modules = [
        ("vintageoptics.types.depth", "Depth Types"),
        ("vintageoptics.types.optics", "Optics Types"), 
        ("vintageoptics.types.io", "IO Types"),
        ("vintageoptics.detection.base_detector", "Base Detector"),
        ("vintageoptics.detection.metadata_extractor", "Metadata Extractor"),
        ("vintageoptics.detection", "Detection Module"),
        ("vintageoptics.physics", "Physics Module"),
        ("vintageoptics.depth.frequency_analyzer", "Frequency Analyzer"),
        ("vintageoptics.depth.depth_analyzer", "Depth Analyzer"),
        ("vintageoptics.depth", "Depth Module"),
        ("vintageoptics.synthesis", "Synthesis Module"),
        ("vintageoptics.statistical", "Statistical Module"),
        ("vintageoptics.analysis", "Analysis Module"),
        ("vintageoptics.core", "Core Module"),
        ("vintageoptics.core.pipeline", "Pipeline"),
        ("vintageoptics", "Main Package")
    ]
    
    success_count = 0
    for module, name in modules:
        if test_import(module, name):
            success_count += 1
    
    print(f"\n📊 Results: {success_count}/{len(modules)} modules imported successfully")
    
    if success_count == len(modules):
        print("\n🎉 All imports successful! VintageOptics is ready to use.")
        
        # Test pipeline creation
        try:
            from vintageoptics.core.pipeline import VintageOpticsPipeline
            pipeline = VintageOpticsPipeline("config/default.yaml")
            print("✅ Pipeline creation successful!")
        except Exception as e:
            print(f"⚠️  Pipeline creation failed: {e}")
        
        return True
    else:
        print("\n❌ Some imports failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
