#!/usr/bin/env python3
"""
VintageOptics Final Import Test
This is the definitive test to verify all imports work correctly.
"""

import sys
import os

# Ensure we're using the local source
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_import_safe(module_name, description):
    """Safely test an import and return result"""
    try:
        __import__(module_name)
        print(f"✅ {description}")
        return True
    except Exception as e:
        print(f"❌ {description}: {str(e)[:100]}...")
        return False

def main():
    print("🔍 VintageOptics Final Import Test")
    print("=" * 50)
    
    success_count = 0
    total_tests = 0
    
    # Core dependencies
    tests = [
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('scipy', 'SciPy'), 
        ('sklearn', 'Scikit-learn'),
        ('yaml', 'PyYAML'),
    ]
    
    print("\n📦 Testing Dependencies:")
    for module, desc in tests:
        total_tests += 1
        if test_import_safe(module, desc):
            success_count += 1
    
    # VintageOptics modules (in dependency order)
    vo_tests = [
        ('vintageoptics.core.config_manager', 'Config Manager'),
        ('vintageoptics.core.performance_monitor', 'Performance Monitor'),
        ('vintageoptics.types.depth', 'Depth Types'),
        ('vintageoptics.types.optics', 'Optics Types'),
        ('vintageoptics.types.io', 'IO Types'),
        ('vintageoptics.detection.base_detector', 'Base Detector'),
        ('vintageoptics.detection.metadata_extractor', 'Metadata Extractor'),
        ('vintageoptics.physics.optics_engine', 'Physics Engine'),
        ('vintageoptics.depth.frequency_analyzer', 'Frequency Analyzer'),
        ('vintageoptics.synthesis', 'Synthesis Module'),
        ('vintageoptics.statistical', 'Statistical Module'),
        ('vintageoptics.analysis', 'Analysis Module'),
        ('vintageoptics.core', 'Core Module'),
        ('vintageoptics.detection', 'Detection Module'),
        ('vintageoptics.depth', 'Depth Module'),
        ('vintageoptics.core.pipeline', 'Pipeline'),
        ('vintageoptics', 'Main Package'),
    ]
    
    print("\n🔧 Testing VintageOptics Modules:")
    for module, desc in vo_tests:
        total_tests += 1
        if test_import_safe(module, desc):
            success_count += 1
    
    # Final test: Create pipeline
    print("\n🚀 Testing Pipeline Creation:")
    total_tests += 1
    try:
        from vintageoptics.core.pipeline import VintageOpticsPipeline
        pipeline = VintageOpticsPipeline("config/default.yaml")
        print("✅ Pipeline Creation")
        success_count += 1
    except Exception as e:
        print(f"❌ Pipeline Creation: {str(e)[:100]}...")
    
    # Results
    print(f"\n📊 Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("\n🎉 ALL TESTS PASSED! VintageOptics is ready to use!")
        return True
    else:
        print(f"\n⚠️  {total_tests - success_count} tests failed.")
        print("Check the errors above for specific issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
