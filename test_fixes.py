#!/usr/bin/env python3
"""Test script to verify the fixes work"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all imports work correctly"""
    print("Testing imports...")
    
    try:
        # Test pipeline import
        from vintageoptics.core.pipeline import VintageOpticsPipeline
        print("✓ Pipeline import successful")
        
        # Test detector imports
        from vintageoptics.detection.electronic_detector import ElectronicLensDetector
        from vintageoptics.detection.vintage_detector import VintageLensDetector
        print("✓ Detector imports successful")
        
        # Test depth module
        from vintageoptics.depth import DepthAwareProcessor
        print("✓ Depth module import successful")
        
        # Test types
        from vintageoptics.types.io import ImageData, ProcessingResult, BatchResult
        print("✓ Types import successful")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_detector_implementation():
    """Test that detectors are no longer stubs"""
    print("\nTesting detector implementations...")
    
    from vintageoptics.detection.electronic_detector import ElectronicLensDetector
    from vintageoptics.detection.vintage_detector import VintageLensDetector
    
    # Create mock config
    config = {}
    
    # Test electronic detector
    e_detector = ElectronicLensDetector(config)
    
    # Create mock image data
    class MockImageData:
        def __init__(self):
            self.metadata = {
                'LensModel': 'Canon EF 50mm f/1.8',
                'FocalLength': 50,
                'FNumber': 1.8
            }
            self.image = None
    
    result = e_detector.detect(MockImageData())
    if result and result.get('lens_type') == 'electronic':
        print("✓ Electronic detector returns valid result")
    else:
        print("✗ Electronic detector still stub")
        
    # Test vintage detector
    v_detector = VintageLensDetector(config)
    
    # Test with vintage lens metadata
    mock_vintage = MockImageData()
    mock_vintage.metadata = {'LensModel': 'Helios 44-2 58mm f/2'}
    
    result = v_detector.detect(mock_vintage)
    if result and result.get('lens_type') == 'vintage':
        print("✓ Vintage detector returns valid result")
    else:
        print("✗ Vintage detector still stub")

def test_pipeline_methods():
    """Test that pipeline methods are implemented"""
    print("\nTesting pipeline methods...")
    
    from vintageoptics.core.pipeline import VintageOpticsPipeline
    
    # Check if methods exist and aren't just pass statements
    methods_to_check = [
        '_setup_plugins',
        '_cache_lens_profile',
        '_detect_applied_corrections',
        '_calculate_correction_params'
    ]
    
    # Create a minimal config file
    import tempfile
    import yaml
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'depth': {'enabled': False}}, f)
        config_path = f.name
    
    try:
        pipeline = VintageOpticsPipeline(config_path)
        
        for method_name in methods_to_check:
            if hasattr(pipeline, method_name):
                method = getattr(pipeline, method_name)
                # Check if method has real implementation (more than just pass)
                import inspect
                source = inspect.getsource(method)
                if 'pass' not in source or len(source.strip().split('\n')) > 3:
                    print(f"✓ {method_name} is implemented")
                else:
                    print(f"✗ {method_name} is still a stub")
            else:
                print(f"✗ {method_name} not found")
                
    finally:
        os.unlink(config_path)

def test_depth_processor():
    """Test depth processor implementation"""
    print("\nTesting depth processor...")
    
    from vintageoptics.depth import DepthAwareProcessor
    import numpy as np
    
    config = {'depth': {'num_layers': 3}}
    processor = DepthAwareProcessor(config)
    
    # Test with mock data
    image = np.random.rand(100, 100, 3).astype(np.float32)
    depth_map = np.random.rand(100, 100).astype(np.float32)
    params = {'distortion_k1': 0.1}
    
    try:
        result = processor.process_with_layers(image, depth_map, params)
        if result.shape == image.shape:
            print("✓ Depth processor returns valid output")
        else:
            print("✗ Depth processor output shape mismatch")
    except Exception as e:
        print(f"✗ Depth processor error: {e}")

def main():
    """Run all tests"""
    print("VintageOptics Fix Verification\n" + "="*30)
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_detector_implementation() 
    test_pipeline_methods()
    test_depth_processor()
    
    print("\n" + "="*30)
    if all_passed:
        print("✓ All critical fixes verified!")
    else:
        print("✗ Some issues remain")

if __name__ == "__main__":
    main()
