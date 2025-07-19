#!/usr/bin/env python3
"""
Example script demonstrating hyperdimensional computing features in VintageOptics.

This shows how to use HD computing for advanced error correction and lens analysis.
"""

import numpy as np
import cv2
from pathlib import Path

# Import the HD features
from vintageoptics.hyperdimensional import (
    HyperdimensionalLensAnalyzer,
    quick_hd_correction,
    analyze_lens_defects,
    separate_vintage_digital_errors
)


def demonstrate_hd_correction():
    """Demonstrate basic HD correction on an image."""
    print("=== Hyperdimensional Correction Demo ===\n")
    
    # Load an example image (replace with your image path)
    # For demo, we'll create a synthetic image with defects
    image = create_synthetic_defective_image()
    
    # Quick correction with default settings
    print("1. Applying quick HD correction...")
    corrected = quick_hd_correction(image, strength=0.8)
    
    # Save results
    cv2.imwrite('original_with_defects.jpg', image)
    cv2.imwrite('hd_corrected.jpg', corrected)
    print("   ✓ Saved: original_with_defects.jpg, hd_corrected.jpg\n")
    
    # Detailed analysis
    print("2. Analyzing defects...")
    defects = analyze_lens_defects(image)
    print(f"   Found {defects['dust_spots']} dust spots")
    print(f"   Found {defects['scratches']} scratches")
    print(f"   Found {defects['regions']} affected regions")
    print(f"   Total defects: {defects['total_defects']}\n")
    
    # Separate vintage vs digital errors
    print("3. Separating vintage and digital errors...")
    separation = separate_vintage_digital_errors(image)
    
    cv2.imwrite('vintage_errors.jpg', separation['vintage_errors'])
    cv2.imwrite('digital_errors.jpg', separation['digital_errors'])
    cv2.imwrite('clean_separated.jpg', separation['clean'])
    
    print(f"   Vintage error confidence: {separation['vintage_confidence']:.2%}")
    print(f"   Digital error confidence: {separation['digital_confidence']:.2%}")
    print("   ✓ Saved: vintage_errors.jpg, digital_errors.jpg, clean_separated.jpg\n")


def demonstrate_lens_matching():
    """Demonstrate lens signature matching."""
    print("=== Lens Signature Matching Demo ===\n")
    
    analyzer = HyperdimensionalLensAnalyzer()
    
    # Create signatures for different lenses
    print("1. Creating lens signatures...")
    
    # Simulate different lens characteristics
    lens_samples = {
        'Canon FD 50mm f/1.4': [create_canon_fd_style_image() for _ in range(3)],
        'Helios 44-2 58mm f/2': [create_helios_style_image() for _ in range(3)],
        'Pentax SMC 50mm f/1.7': [create_pentax_style_image() for _ in range(3)]
    }
    
    for lens_name, samples in lens_samples.items():
        signature = analyzer.create_lens_signature(lens_name, samples)
        print(f"   ✓ Created signature for {lens_name}")
    
    # Test matching
    print("\n2. Testing lens matching...")
    test_image = create_helios_style_image()  # Should match Helios
    
    matched = analyzer.match_lens(test_image, threshold=0.7)
    if matched:
        print(f"   ✓ Matched lens: {matched}")
    else:
        print("   ✗ No match found")


def demonstrate_iterative_enhancement():
    """Demonstrate iterative quality enhancement."""
    print("=== Iterative Enhancement Demo ===\n")
    
    analyzer = HyperdimensionalLensAnalyzer()
    
    # Create a heavily degraded image
    image = create_heavily_degraded_image()
    cv2.imwrite('heavily_degraded.jpg', image)
    
    print("1. Starting iterative enhancement...")
    result = analyzer.iterative_enhancement(
        image,
        target_quality=0.75,
        max_iterations=5
    )
    
    cv2.imwrite('iteratively_enhanced.jpg', result['enhanced'])
    
    print(f"   Final quality score: {result['final_quality']:.2%}")
    print(f"   Iterations used: {result['iterations']}")
    print("\n   Quality progression:")
    for h in result['history']:
        print(f"   - Iteration {h['iteration']}: {h['quality_score']:.2%} (strength: {h['strength']:.1f})")
    print("   ✓ Saved: heavily_degraded.jpg, iteratively_enhanced.jpg\n")


def demonstrate_advanced_features():
    """Demonstrate advanced HD computing features."""
    print("=== Advanced HD Features Demo ===\n")
    
    analyzer = HyperdimensionalLensAnalyzer(dimension=20000)  # Higher dimension for accuracy
    
    # Load or create test image
    image = create_complex_defect_image()
    
    # Full analysis with all features
    print("1. Performing comprehensive analysis...")
    results = analyzer.analyze_and_correct(image, mode='auto', strength=0.9)
    
    print(f"   Detected mode: {results.get('detected_mode', 'hybrid')}")
    print(f"   Quality score: {results['quality_score']:.2%}")
    
    # Topological analysis details
    topo = results['topology']
    print(f"\n2. Topological analysis:")
    print(f"   - Dust features: {len(topo['dust_features'])}")
    for i, feature in enumerate(topo['dust_features'][:3]):  # Show first 3
        print(f"     • Feature {i+1}: persistence={feature.persistence:.3f}, "
              f"location=({feature.location[0]:.2f}, {feature.location[1]:.2f})")
    
    # HD vector analysis
    print(f"\n3. Hyperdimensional encoding:")
    defect_hv = results['defect_hypervector']
    print(f"   - Vector dimension: {len(defect_hv)}")
    print(f"   - Vector norm: {np.linalg.norm(defect_hv):.3f}")
    print(f"   - Sparsity: {np.sum(np.abs(defect_hv) < 0.01) / len(defect_hv):.2%}")
    
    cv2.imwrite('comprehensive_corrected.jpg', results['corrected'])
    print("\n   ✓ Saved: comprehensive_corrected.jpg")


# Helper functions to create synthetic test images

def create_synthetic_defective_image():
    """Create a synthetic image with various defects."""
    # Base image - gradient with some structure
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Add gradient background
    for i in range(600):
        image[i, :] = [100 + i//4, 120 + i//5, 140 + i//6]
    
    # Add some structure (circles)
    cv2.circle(image, (200, 200), 80, (200, 200, 200), -1)
    cv2.circle(image, (600, 400), 120, (180, 180, 180), -1)
    
    # Add vintage defects
    # Dust spots
    for _ in range(20):
        x, y = np.random.randint(0, 800), np.random.randint(0, 600)
        radius = np.random.randint(3, 15)
        cv2.circle(image, (x, y), radius, (50, 50, 50), -1)
    
    # Scratches
    for _ in range(5):
        pt1 = (np.random.randint(0, 800), np.random.randint(0, 600))
        pt2 = (np.random.randint(0, 800), np.random.randint(0, 600))
        cv2.line(image, pt1, pt2, (60, 60, 60), 2)
    
    # Add digital defects
    # Hot pixels
    hot_pixels = np.random.rand(600, 800) > 0.998
    image[hot_pixels] = [255, 255, 255]
    
    # Banding
    for i in range(0, 600, 50):
        image[i:i+2, :] = image[i:i+2, :] * 0.8
    
    return image


def create_canon_fd_style_image():
    """Create image with Canon FD lens characteristics."""
    image = create_synthetic_defective_image()
    
    # Add slight vignetting
    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]
    center = (h/2, w/2)
    dist = np.sqrt((Y - center[0])**2 + (X - center[1])**2)
    vignette = 1 - (dist / np.max(dist)) * 0.3
    
    for c in range(3):
        image[:, :, c] = (image[:, :, c] * vignette).astype(np.uint8)
    
    return image


def create_helios_style_image():
    """Create image with Helios swirly bokeh characteristics."""
    image = create_synthetic_defective_image()
    
    # Add swirly distortion pattern
    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]
    center = (h/2, w/2)
    
    # Create swirl effect
    angle = np.arctan2(Y - center[0], X - center[1])
    dist = np.sqrt((Y - center[0])**2 + (X - center[1])**2)
    
    swirl_strength = 0.02
    angle_offset = swirl_strength * dist / np.max(dist)
    
    # Apply subtle swirl (simplified version)
    # In real implementation, would use proper remapping
    noise = np.sin(angle + angle_offset) * 10
    
    for c in range(3):
        image[:, :, c] = np.clip(image[:, :, c] + noise, 0, 255).astype(np.uint8)
    
    return image


def create_pentax_style_image():
    """Create image with Pentax SMC coating characteristics."""
    image = create_synthetic_defective_image()
    
    # Add characteristic flare pattern
    cv2.circle(image, (400, 300), 150, (220, 210, 200), 2)
    cv2.circle(image, (400, 300), 180, (210, 200, 190), 1)
    
    return image


def create_heavily_degraded_image():
    """Create heavily degraded image for iterative enhancement demo."""
    image = create_synthetic_defective_image()
    
    # Add heavy noise
    noise = np.random.normal(0, 30, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    # Add haze
    haze = np.ones_like(image) * 50
    image = cv2.addWeighted(image, 0.7, haze, 0.3, 0)
    
    return image


def create_complex_defect_image():
    """Create image with complex mixed defects."""
    image = create_synthetic_defective_image()
    
    # Add various complex defects
    # Fungus pattern
    center = (400, 300)
    for r in range(50, 150, 20):
        angle = np.random.rand() * 2 * np.pi
        for a in np.linspace(0, 2*np.pi, 8):
            x = int(center[0] + r * np.cos(a + angle))
            y = int(center[1] + r * np.sin(a + angle))
            cv2.circle(image, (x, y), 5, (70, 70, 70), -1)
    
    # Oil spots
    for _ in range(3):
        x, y = np.random.randint(100, 700), np.random.randint(100, 500)
        cv2.ellipse(image, (x, y), (40, 25), np.random.rand()*180, 0, 360, (80, 85, 90), -1)
    
    return image


if __name__ == "__main__":
    print("\n" + "="*60)
    print("VintageOptics Hyperdimensional Computing Demo")
    print("="*60 + "\n")
    
    # Run all demonstrations
    demonstrate_hd_correction()
    print("\n" + "-"*60 + "\n")
    
    demonstrate_lens_matching()
    print("\n" + "-"*60 + "\n")
    
    demonstrate_iterative_enhancement()
    print("\n" + "-"*60 + "\n")
    
    demonstrate_advanced_features()
    
    print("\n✨ Demo complete! Check the generated images to see the results.")
