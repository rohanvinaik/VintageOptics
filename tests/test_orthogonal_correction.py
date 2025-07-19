#!/usr/bin/env python3
"""
Test script for the Orthogonal Error Correction pipeline

Demonstrates the new error correction capabilities that leverage
the complementary nature of analog and digital error sources.
"""

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vintageoptics.analysis import HybridErrorCorrector
from vintageoptics.orchestration import ModularPipeline
from vintageoptics.utils.logger import setup_logger

logger = setup_logger(__name__)


def generate_test_images():
    """Generate synthetic test images with characteristic errors"""
    height, width = 512, 512
    
    # Create base image
    y, x = np.ogrid[:height, :width]
    center = (width//2, height//2)
    radius = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # Base pattern
    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[radius < 200] = [200, 150, 100]  # Circle pattern
    
    # Add vintage-style errors (smooth, physics-based)
    vintage = base.copy().astype(float)
    
    # Vignetting
    vignette = 1 - (radius / radius.max())**2
    vintage *= vignette[:, :, np.newaxis]
    
    # Chromatic aberration
    vintage[:, :, 0] = np.roll(vintage[:, :, 0], 2, axis=1)  # Red shift
    vintage[:, :, 2] = np.roll(vintage[:, :, 2], -2, axis=1)  # Blue shift
    
    # Add digital-style errors (noisy, discrete)
    digital = base.copy().astype(float)
    
    # Shot noise
    noise = np.random.normal(0, 20, digital.shape)
    digital += noise
    
    # Fixed pattern noise (hot pixels)
    hot_pixels = np.random.random((height, width)) > 0.999
    digital[hot_pixels] = 255
    
    # Clip and convert
    vintage = np.clip(vintage, 0, 255).astype(np.uint8)
    digital = np.clip(digital, 0, 255).astype(np.uint8)
    
    return vintage, digital


def test_error_correction():
    """Test the orthogonal error correction"""
    logger.info("Testing Orthogonal Error Correction")
    
    # Generate test images
    vintage_img, digital_img = generate_test_images()
    
    # Initialize corrector
    corrector = HybridErrorCorrector()
    
    # Process
    corrected, report = corrector.process(vintage_img, digital_img)
    
    # Display results
    confidence = report['correction_confidence']
    logger.info(f"Correction confidence: {confidence:.2%}")
    
    # Analyze error characteristics
    vintage_chars = report['vintage_characteristics']
    digital_chars = report['digital_characteristics']
    
    logger.info(f"Vintage error temporal stability: {vintage_chars.temporal_stability:.2f}")
    logger.info(f"Digital error temporal stability: {digital_chars.temporal_stability:.2f}")
    
    return corrected, report


def test_pipeline():
    """Test the full modular pipeline"""
    logger.info("Testing Modular Pipeline")
    
    # Create test configuration
    from vintageoptics.orchestration import PipelineConfig, PipelineStage
    
    config = PipelineConfig(
        stages=[
            PipelineStage(
                name="orthogonal_correction",
                module="vintageoptics.analysis.error_orthogonality",
                params={"confidence_threshold": 0.7}
            )
        ],
        parallel_processing=False
    )
    
    # Initialize pipeline
    pipeline = ModularPipeline()
    pipeline.config = config
    
    # Generate and save test images
    vintage_img, digital_img = generate_test_images()
    
    # Create temp files
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as vf:
        vintage_path = Path(vf.name)
        import cv2
        cv2.imwrite(str(vintage_path), vintage_img)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as df:
        digital_path = Path(df.name)
        cv2.imwrite(str(digital_path), digital_img)
    
    # Process
    results = pipeline.process_image_pair(vintage_path, digital_path)
    
    # Clean up
    vintage_path.unlink()
    digital_path.unlink()
    
    return results


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("VintageOptics Orthogonal Error Correction Test Suite")
    print("="*60 + "\n")
    
    # Test 1: Error Correction
    print("Test 1: Basic Error Correction")
    print("-"*40)
    corrected, report = test_error_correction()
    print(f"✓ Correction completed with {report['correction_confidence']:.1%} confidence\n")
    
    # Test 2: Pipeline Integration
    print("Test 2: Pipeline Integration")
    print("-"*40)
    results = test_pipeline()
    print("✓ Pipeline execution completed\n")
    
    print("="*60)
    print("All tests completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
