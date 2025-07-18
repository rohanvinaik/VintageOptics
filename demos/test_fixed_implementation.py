#!/usr/bin/env python3
"""
Test script to verify the fixed implementation
Tests the core functionality after implementing the recommended fixes
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import cv2
from vintageoptics.core.pipeline import VintageOpticsPipeline, ProcessingMode, ProcessingRequest
from vintageoptics.core.config_manager import ConfigManager
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_config():
    """Create a test configuration"""
    return {
        'database': {
            'path': 'test_db.sqlite',
            'enabled': False  # Disable DB for simple test
        },
        'vintageml': {
            'patch_size': 16,
            'pca_components': 8,
            'som_size': (10, 10),
            'use_adaptive_reduction': True,
            'use_quantization': False
        },
        'hybrid': {
            'max_iterations': 5,
            'min_iterations': 2,
            'convergence_threshold': 0.01,
            'early_stop_patience': 2,
            'use_modern_fallback': False
        },
        'pac': {
            'entropy_threshold': 2.0,
            'confidence_delta': 0.05
        },
        'depth': {
            'enabled': False,  # Disable depth for simple test
            'num_layers': 3
        },
        'parallel_processing': False,
        'plugins': {
            'enabled': False
        }
    }


def create_synthetic_image():
    """Create a synthetic test image with lens distortions"""
    # Create a checkerboard pattern
    size = 512
    square_size = 32
    image = np.zeros((size, size, 3), dtype=np.uint8)
    
    for i in range(0, size, square_size):
        for j in range(0, size, square_size):
            if (i // square_size + j // square_size) % 2 == 0:
                image[i:i+square_size, j:j+square_size] = [255, 255, 255]
    
    # Add synthetic barrel distortion
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    
    # Create distortion maps
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    
    for y in range(h):
        for x in range(w):
            # Normalized coordinates
            nx = (x - cx) / cx
            ny = (y - cy) / cy
            r2 = nx * nx + ny * ny
            
            # Barrel distortion
            k1 = 0.1
            k2 = 0.05
            radial = 1 + k1 * r2 + k2 * r2 * r2
            
            # Map back to pixel coordinates
            map_x[y, x] = cx + nx * cx * radial
            map_y[y, x] = cy + ny * cy * radial
    
    # Apply distortion
    distorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    
    # Add some synthetic defects
    # Dust spots
    for _ in range(10):
        x = np.random.randint(50, w-50)
        y = np.random.randint(50, h-50)
        radius = np.random.randint(2, 5)
        cv2.circle(distorted, (x, y), radius, (100, 100, 100), -1)
    
    # Vignetting
    vignette = np.ones_like(distorted, dtype=np.float32)
    for y in range(h):
        for x in range(w):
            dist = np.sqrt((x - cx)**2 + (y - cy)**2) / np.sqrt(cx**2 + cy**2)
            vignette[y, x] = 1.0 - 0.5 * dist**2
    
    distorted = (distorted * vignette).astype(np.uint8)
    
    return distorted


def test_unified_lens_detector():
    """Test the UnifiedLensDetector import fix"""
    logger.info("Testing UnifiedLensDetector import...")
    
    try:
        from vintageoptics.detection import UnifiedLensDetector
        logger.info("✓ UnifiedLensDetector imports correctly")
        
        # Test instantiation
        config = create_test_config()
        detector = UnifiedLensDetector(config)
        logger.info("✓ UnifiedLensDetector instantiates correctly")
        
        # Test detection on synthetic image
        image = create_synthetic_image()
        image_data = {'image': image, 'path': None}
        result = detector.detect_comprehensive(image_data)
        logger.info(f"✓ Detection completed: {result.model} (confidence: {result.confidence:.2f})")
        
    except Exception as e:
        logger.error(f"✗ UnifiedLensDetector test failed: {e}")
        return False
    
    return True


def test_vintage_ml_consensus():
    """Test the vintage ML consensus mechanism"""
    logger.info("\nTesting Vintage ML consensus mechanism...")
    
    try:
        from vintageoptics.vintageml.detector import VintageMLDefectDetector, VintageDefectResult
        
        config = create_test_config()
        detector = VintageMLDefectDetector(config)
        logger.info("✓ VintageMLDefectDetector instantiates correctly")
        
        # Create test results for consensus
        h, w = 512, 512
        
        # Simulate multiple detection results
        results = [
            VintageDefectResult(
                defect_mask=np.random.randint(0, 2, (h, w), dtype=np.uint8) * 255,
                defect_type="dust",
                confidence=0.8,
                method_used="perceptron"
            ),
            VintageDefectResult(
                defect_mask=np.random.randint(0, 2, (h, w), dtype=np.uint8) * 255,
                defect_type="dust",
                confidence=0.7,
                method_used="knn"
            ),
            VintageDefectResult(
                defect_mask=np.random.randint(0, 2, (h, w), dtype=np.uint8) * 255,
                defect_type="scratch",
                confidence=0.6,
                method_used="som"
            )
        ]
        
        # Test consensus combination
        consensus_results = detector._combine_results(results)
        logger.info(f"✓ Consensus mechanism works: {len(consensus_results)} consensus results from {len(results)} inputs")
        
        for result in consensus_results:
            logger.info(f"  - {result.defect_type}: confidence={result.confidence:.2f}, method={result.method_used}")
        
    except Exception as e:
        logger.error(f"✗ Vintage ML consensus test failed: {e}")
        return False
    
    return True


def test_hybrid_pipeline_convergence():
    """Test the hybrid pipeline convergence logic"""
    logger.info("\nTesting Hybrid Pipeline convergence...")
    
    try:
        from vintageoptics.core.hybrid_pipeline import HybridPhysicsMLPipeline
        
        config = create_test_config()
        pipeline = HybridPhysicsMLPipeline(config)
        logger.info("✓ HybridPhysicsMLPipeline instantiates correctly")
        
        # Test with synthetic image
        image = create_synthetic_image()
        metadata = {'LensModel': 'Synthetic Test Lens'}
        
        # Process with convergence tracking
        result = pipeline.process(image, metadata)
        
        logger.info(f"✓ Hybrid processing completed:")
        logger.info(f"  - Iterations: {result.iterations}")
        logger.info(f"  - Final error: {result.residual_error:.4f}")
        logger.info(f"  - ML confidence: {result.ml_confidence:.2f}")
        logger.info(f"  - Defect masks: {list(result.defect_masks.keys())}")
        
        # Check convergence behavior
        if result.iterations < config['hybrid']['max_iterations']:
            logger.info("✓ Early convergence achieved")
        else:
            logger.info("⚠ Maximum iterations reached")
        
    except Exception as e:
        logger.error(f"✗ Hybrid pipeline convergence test failed: {e}")
        return False
    
    return True


def test_full_pipeline():
    """Test the full processing pipeline"""
    logger.info("\nTesting full processing pipeline...")
    
    try:
        # Create test config
        config = create_test_config()
        
        # Save config temporarily
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        
        # Initialize pipeline
        pipeline = VintageOpticsPipeline(config_path)
        logger.info("✓ VintageOpticsPipeline instantiates correctly")
        
        # Create test image
        image = create_synthetic_image()
        
        # Save test image temporarily
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            image_path = f.name
            cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Test correction mode
        request = ProcessingRequest(
            image_path=image_path,
            mode=ProcessingMode.CORRECT,
            output_path=None,
            settings={'correct_distortion': True, 'correct_vignetting': True}
        )
        
        result = pipeline.process(request)
        logger.info("✓ Correction processing completed")
        
        if hasattr(result, 'quality_metrics') and result.quality_metrics:
            logger.info(f"  - Quality score: {result.quality_metrics.get('quality_score', 0):.2f}")
        
        # Cleanup temp files
        os.unlink(config_path)
        os.unlink(image_path)
        
    except Exception as e:
        logger.error(f"✗ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all tests"""
    logger.info("=== VintageOptics Fixed Implementation Test Suite ===\n")
    
    tests = [
        ("UnifiedLensDetector Import", test_unified_lens_detector),
        ("Vintage ML Consensus", test_vintage_ml_consensus),
        ("Hybrid Pipeline Convergence", test_hybrid_pipeline_convergence),
        ("Full Pipeline", test_full_pipeline)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running: {test_name} ---")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n=== Test Summary ===")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
