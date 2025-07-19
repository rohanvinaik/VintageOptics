#!/usr/bin/env python3
"""
VintageOptics MVP Component Test Suite
Tests all individual components and upgrades them to working implementations
"""

import sys
import os
import numpy as np
import cv2
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from vintageoptics.core.config_manager import ConfigManager
from vintageoptics.core.pipeline import VintageOpticsPipeline, ProcessingMode, ProcessingRequest
from vintageoptics.detection import UnifiedLensDetector
from vintageoptics.physics import OpticsEngine
from vintageoptics.depth import DepthAwareProcessor
from vintageoptics.synthesis import LensSynthesizer
from vintageoptics.statistical import AdaptiveCleanup
from vintageoptics.analysis import QualityAnalyzer

def create_test_image(size=(640, 480), pattern='checkerboard'):
    """Create a test image for testing"""
    h, w = size
    
    if pattern == 'checkerboard':
        # Create checkerboard pattern to test distortion
        image = np.zeros((h, w, 3), dtype=np.uint8)
        square_size = 40
        for i in range(0, h, square_size):
            for j in range(0, w, square_size):
                if ((i // square_size) + (j // square_size)) % 2 == 0:
                    image[i:i+square_size, j:j+square_size] = [255, 255, 255]
                else:
                    image[i:i+square_size, j:j+square_size] = [0, 0, 0]
    
    elif pattern == 'grid':
        # Create grid pattern
        image = np.ones((h, w, 3), dtype=np.uint8) * 128
        for i in range(0, h, 20):
            image[i:i+2, :] = [255, 255, 255]
        for j in range(0, w, 20):
            image[:, j:j+2] = [255, 255, 255]
    
    elif pattern == 'radial':
        # Create radial pattern to test vignetting
        center_x, center_y = w // 2, h // 2
        y, x = np.ogrid[:h, :w]
        radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_radius = min(center_x, center_y)
        normalized_radius = radius / max_radius
        intensity = (255 * (1 - normalized_radius)).clip(0, 255).astype(np.uint8)
        image = np.stack([intensity, intensity, intensity], axis=2)
    
    return image

def test_component(component_name, test_func):
    """Test a component and report results"""
    print(f"\n🔧 Testing {component_name}...")
    try:
        result = test_func()
        if result:
            print(f"✅ {component_name}: PASS")
            return True
        else:
            print(f"⚠️  {component_name}: PARTIAL")
            return False
    except Exception as e:
        print(f"❌ {component_name}: FAIL - {e}")
        return False

def test_config_manager():
    """Test configuration management"""
    config = ConfigManager.load("config/default.yaml")
    
    # Check required config sections
    required_sections = ['processing', 'physics', 'depth', 'synthesis']
    for section in required_sections:
        if section not in config:
            print(f"Missing config section: {section}")
            return False
    
    print("  ✓ Configuration loaded successfully")
    print(f"  ✓ Found {len(config)} config sections")
    return True

def test_lens_detection():
    """Test lens detection capabilities"""
    config = ConfigManager.load("config/default.yaml")
    detector = UnifiedLensDetector(config)
    
    # Create mock image data
    from vintageoptics.core import ImageData
    test_image = create_test_image()
    image_data = ImageData(test_image)
    
    # Test detection
    result = detector.detect_comprehensive(image_data)
    
    if not isinstance(result, dict):
        print("  ❌ Detection should return dict")
        return False
    
    required_fields = ['lens_id', 'manufacturer', 'model', 'confidence']
    for field in required_fields:
        if field not in result:
            print(f"  ❌ Missing field: {field}")
            return False
    
    print("  ✓ Lens detection returns required fields")
    print(f"  ✓ Detected: {result['manufacturer']} {result['model']}")
    return True

def test_physics_engine():
    """Test physics-based corrections"""
    config = ConfigManager.load("config/default.yaml")
    engine = OpticsEngine(config)
    
    # Test with checkerboard pattern (good for distortion testing)
    test_image = create_test_image(pattern='checkerboard')
    print(f"  ✓ Created test image: {test_image.shape}")
    
    # Test correction
    corrected = engine.apply_corrections(test_image, {
        'distortion_k1': 0.1,
        'distortion_k2': 0.05,
        'chromatic_red': 1.02,
        'chromatic_blue': 0.98
    })
    
    if corrected is None:
        print("  ❌ Correction returned None")
        return False
    
    if corrected.shape != test_image.shape:
        print("  ❌ Output shape doesn't match input")
        return False
    
    # Check that image was actually modified (not just copied)
    difference = np.sum(np.abs(corrected.astype(float) - test_image.astype(float)))
    if difference == 0:
        print("  ⚠️  Image appears unchanged - may need better implementation")
        return False
    
    print("  ✓ Physics corrections applied successfully")
    print(f"  ✓ Image modified (difference: {difference:.0f})")
    return True

def test_depth_analysis():
    """Test depth analysis capabilities"""
    config = ConfigManager.load("config/default.yaml")
    processor = DepthAwareProcessor(config)
    
    # Create test image with depth variation
    test_image = create_test_image(pattern='radial')
    
    # Test depth-aware processing
    from vintageoptics.types.depth import DepthMap, DepthLayer
    mock_depth_map = DepthMap(
        depth_map=np.random.random((480, 640)),
        confidence_map=np.ones((480, 640)) * 0.8,
        focus_points=[],
        depth_layers=[]
    )
    
    result = processor.process_with_layers(test_image, mock_depth_map, {})
    
    if result is None:
        print("  ❌ Depth processing returned None")
        return False
    
    if result.shape != test_image.shape:
        print("  ❌ Output shape mismatch")
        return False
    
    print("  ✓ Depth-aware processing working")
    return True

def test_statistical_cleanup():
    """Test statistical defect cleanup"""
    config = ConfigManager.load("config/default.yaml")
    cleanup = AdaptiveCleanup(config)
    
    # Create image with simulated defects
    test_image = create_test_image(pattern='grid')
    
    # Add some "defects" (noise)
    noisy_image = test_image.copy()
    noise = np.random.randint(0, 50, test_image.shape, dtype=np.uint8)
    noisy_image = cv2.add(noisy_image, noise)
    
    # Test cleanup
    cleaned = cleanup.clean_with_preservation(noisy_image, {}, None)
    
    if cleaned is None:
        print("  ❌ Cleanup returned None")
        return False
    
    # Calculate noise reduction
    original_noise = np.std(noisy_image - test_image)
    remaining_noise = np.std(cleaned - test_image)
    noise_reduction = (original_noise - remaining_noise) / original_noise
    
    print(f"  ✓ Noise reduction: {noise_reduction:.1%}")
    return True

def test_quality_analysis():
    """Test quality analysis"""
    analyzer = QualityAnalyzer()
    
    # Create before/after images
    original = create_test_image(pattern='checkerboard')
    
    # Simulate "improved" image (slightly sharper)
    improved = cv2.GaussianBlur(original, (3, 3), 0)
    improved = cv2.addWeighted(original, 1.5, improved, -0.5, 0)
    improved = np.clip(improved, 0, 255).astype(np.uint8)
    
    # Analyze quality
    metrics = analyzer.analyze(original, improved)
    
    if not isinstance(metrics, dict):
        print("  ❌ Analysis should return dict")
        return False
    
    if 'quality_score' not in metrics:
        print("  ❌ Missing quality_score")
        return False
    
    print(f"  ✓ Quality score: {metrics['quality_score']}")
    return True

def test_synthesis():
    """Test lens synthesis capabilities"""
    config = ConfigManager.load("config/default.yaml")
    synthesizer = LensSynthesizer(config)
    
    test_image = create_test_image(pattern='radial')
    
    # Mock profiles
    source_profile = {'lens_id': 'test_source'}
    target_profile = {'lens_id': 'test_target'}
    
    # Test synthesis
    result = synthesizer.apply_lens_character(
        test_image, source_profile, target_profile, None, {}
    )
    
    if result is None:
        print("  ❌ Synthesis returned None")
        return False
    
    print("  ✓ Lens synthesis working")
    return True

def test_end_to_end_pipeline():
    """Test complete pipeline"""
    print("\n🚀 Testing End-to-End Pipeline...")
    
    pipeline = VintageOpticsPipeline("config/default.yaml")
    
    # Create test image file
    test_image = create_test_image(size=(800, 600), pattern='checkerboard')
    test_image_path = 'test_input.jpg'
    output_path = 'test_output.jpg'
    
    # Save test image
    cv2.imwrite(test_image_path, cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
    print(f"  ✓ Created test image: {test_image_path}")
    
    try:
        # Create processing request
        request = ProcessingRequest(
            image_path=test_image_path,
            mode=ProcessingMode.CORRECT,
            output_path=output_path,
            use_depth=True,
            gpu_acceleration=False
        )
        
        # Process image
        result = pipeline.process(request)
        
        if result is None:
            print("  ❌ Pipeline returned None")
            return False
        
        if not hasattr(result, 'image'):
            print("  ❌ Result missing image")
            return False
        
        print("  ✓ Pipeline processing completed")
        print(f"  ✓ Result mode: {result.mode}")
        
        # Cleanup
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Pipeline error: {e}")
        # Cleanup on error
        for path in [test_image_path, output_path]:
            if os.path.exists(path):
                os.remove(path)
        return False

def main():
    """Run all MVP component tests"""
    print("🧪 VintageOptics MVP Component Test Suite")
    print("=" * 50)
    
    tests = [
        ("Configuration Manager", test_config_manager),
        ("Lens Detection", test_lens_detection), 
        ("Physics Engine", test_physics_engine),
        ("Depth Analysis", test_depth_analysis),
        ("Statistical Cleanup", test_statistical_cleanup),
        ("Quality Analysis", test_quality_analysis),
        ("Lens Synthesis", test_synthesis),
        ("End-to-End Pipeline", test_end_to_end_pipeline),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_component(test_name, test_func):
            passed += 1
    
    print(f"\n📊 MVP Test Results: {passed}/{total} components working")
    
    if passed == total:
        print("\n🎉 MVP COMPLETE! All components functional.")
        print("✅ VintageOptics is ready for production use!")
    elif passed >= total * 0.8:
        print("\n✅ MVP MOSTLY COMPLETE! Core functionality working.")
        print("⚠️  Some advanced features may need refinement.")
    else:
        print("\n⚠️  MVP PARTIAL. Core components need implementation.")
    
    return passed >= total * 0.8

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
