#!/usr/bin/env python3
"""
Demonstration of VintageOptics with Vintage ML integration
Shows the hybrid physics-ML pipeline in action
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

# Import VintageOptics components
from src.vintageoptics.core import (
    VintageOpticsPipeline, 
    ProcessingMode, 
    ProcessingRequest
)
from src.vintageoptics.vintageml import VintageMLDefectDetector, VintageMLTrainer

def create_synthetic_vintage_image():
    """Create a synthetic image with vintage lens characteristics"""
    # Create base image with gradient
    size = 512
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    
    # Base pattern
    base = np.sin(5 * x) * np.cos(5 * y) * 0.3 + 0.5
    
    # Add radial vignetting
    r = np.sqrt(x**2 + y**2)
    vignetting = 1 - 0.5 * r**2
    
    # Create RGB image
    image = np.zeros((size, size, 3))
    image[:, :, 0] = base * vignetting * 255  # Red channel
    image[:, :, 1] = base * vignetting * 0.9 * 255  # Green channel
    image[:, :, 2] = base * vignetting * 0.8 * 255  # Blue channel
    
    # Add some dust spots
    n_dust = 20
    for _ in range(n_dust):
        cx = np.random.randint(50, size-50)
        cy = np.random.randint(50, size-50)
        radius = np.random.randint(2, 8)
        brightness = np.random.uniform(0.3, 0.7)
        cv2.circle(image, (cx, cy), radius, (brightness*255,)*3, -1)
    
    # Add scratches
    for _ in range(3):
        pt1 = (np.random.randint(0, size), np.random.randint(0, size))
        pt2 = (np.random.randint(0, size), np.random.randint(0, size))
        cv2.line(image, pt1, pt2, (200, 200, 200), 1)
    
    # Add chromatic aberration at edges
    shift = 3
    image[shift:, shift:, 0] = image[:-shift, :-shift, 0]  # Shift red channel
    image[:-shift, :-shift, 2] = image[shift:, shift:, 2]  # Shift blue channel
    
    return image.astype(np.uint8)

def demonstrate_vintage_ml():
    """Demonstrate vintage ML defect detection"""
    print("=== Vintage ML Defect Detection Demo ===\n")
    
    # Create test image
    print("1. Creating synthetic vintage lens image...")
    image = create_synthetic_vintage_image()
    
    # Initialize detector with config
    config = {
        'vintageml': {
            'patch_size': 16,
            'pca_components': 8,
            'som_size': (10, 10)
        },
        'cleanup': {
            'dust_sensitivity': 0.8,
            'preserve_character': True
        }
    }
    
    detector = VintageMLDefectDetector(config)
    
    # Detect defects (unsupervised mode)
    print("2. Running vintage ML detection (unsupervised mode)...")
    results = detector.detect_defects(image)
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Feature extraction visualization
    features, positions = detector.extract_features(image)
    print(f"   - Extracted {len(features)} patches with {features.shape[1]} features each")
    
    # Show detected defects
    for i, result in enumerate(results[:4]):
        ax = axes[i+1]
        
        # Overlay defect mask
        overlay = image.copy()
        mask_colored = np.zeros_like(image)
        mask_colored[:, :, 0] = result.defect_mask  # Red for defects
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        ax.imshow(overlay)
        ax.set_title(f"{result.method_used}: {result.defect_type}\nConfidence: {result.confidence:.2f}")
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(len(results)+1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('vintage_ml_detection_demo.png', dpi=150, bbox_inches='tight')
    print("   - Saved visualization to vintage_ml_detection_demo.png")
    
    return image, results

def demonstrate_hybrid_pipeline():
    """Demonstrate the full hybrid physics-ML pipeline"""
    print("\n=== Hybrid Physics-ML Pipeline Demo ===\n")
    
    # Create test image
    image = create_synthetic_vintage_image()
    cv2.imwrite('test_vintage_image.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Initialize pipeline
    print("1. Initializing VintageOptics pipeline...")
    
    # Create config file
    config = {
        'hybrid': {
            'max_iterations': 3,
            'convergence_threshold': 0.01,
            'use_modern_fallback': False
        },
        'vintageml': {
            'patch_size': 16,
            'pca_components': 8
        },
        'pac': {
            'entropy_threshold': 2.0,
            'confidence_delta': 0.05
        },
        'physics': {
            'distortion_model': 'brown_conrady',
            'chromatic_model': 'linear_shift'
        },
        'cleanup': {
            'preserve_character': True,
            'dust_sensitivity': 0.8
        },
        'depth': {
            'enabled': False  # Disable for demo
        }
    }
    
    import yaml
    with open('demo_config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    pipeline = VintageOpticsPipeline('demo_config.yaml')
    
    # Process with hybrid mode
    print("2. Processing image with hybrid physics-ML approach...")
    request = ProcessingRequest(
        image_path='test_vintage_image.png',
        mode=ProcessingMode.HYBRID,
        output_path='corrected_hybrid.png'
    )
    
    result = pipeline.process(request)
    
    # Show iteration details
    if hasattr(result, 'iterations'):
        print(f"   - Converged in {result.iterations} iterations")
        print(f"   - Final residual error: {result.residual_error:.4f}")
        print(f"   - ML confidence: {result.ml_confidence:.2f}")
    
    # Visualize the iterative process
    print("\n3. Visualizing the hybrid processing steps...")
    
    # This would show:
    # - Original image
    # - Initial physics correction
    # - ML-detected residuals
    # - Refined physics correction
    # - Final ML cleanup
    # - Final result
    
    print("\nDemo complete! Check the output files:")
    print("  - vintage_ml_detection_demo.png: Vintage ML detection visualization")
    print("  - corrected_hybrid.png: Final corrected image")
    print("  - demo_config.yaml: Configuration used")

def demonstrate_training():
    """Demonstrate training the vintage ML models"""
    print("\n=== Vintage ML Training Demo ===\n")
    
    # Create training data
    print("1. Generating synthetic training data...")
    training_data = []
    
    for i in range(10):
        # Create image with known defects
        image = create_synthetic_vintage_image()
        
        # Create ground truth mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Add known dust locations
        for _ in range(10):
            x = np.random.randint(10, image.shape[1]-10)
            y = np.random.randint(10, image.shape[0]-10)
            cv2.circle(mask, (x, y), 5, 255, -1)
        
        training_data.append((image, mask))
    
    print(f"   - Created {len(training_data)} training samples")
    
    # Train detector
    print("2. Training vintage ML models...")
    config = {
        'vintageml': {'patch_size': 16, 'pca_components': 8}
    }
    
    detector = VintageMLDefectDetector(config)
    detector.train(training_data)
    
    # Show training visualization
    print("3. Visualizing learned components...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # PCA components
    if hasattr(detector.pca, 'components_'):
        ax = axes[0, 0]
        # Show first principal component
        pc1 = detector.pca.components_[0].reshape(16, 16)
        ax.imshow(pc1, cmap='RdBu')
        ax.set_title("First Principal Component")
        ax.axis('off')
        
        # Explained variance
        ax = axes[0, 1]
        ax.bar(range(len(detector.pca.explained_variance_ratio_)), 
               detector.pca.explained_variance_ratio_)
        ax.set_xlabel("Component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title("PCA Explained Variance")
    
    # SOM visualization
    if hasattr(detector.som, 'weights'):
        ax = axes[0, 2]
        # U-matrix
        u_matrix = detector.som.get_u_matrix()
        im = ax.imshow(u_matrix, cmap='hot')
        ax.set_title("SOM U-Matrix")
        ax.axis('off')
        plt.colorbar(im, ax=ax)
    
    # Perceptron decision boundary (for 2D projection)
    if detector.perceptron and hasattr(detector.perceptron, 'weights'):
        ax = axes[1, 0]
        ax.text(0.5, 0.5, f"Perceptron trained\n{len(detector.perceptron.errors_per_epoch)} epochs", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Perceptron Training")
        ax.axis('off')
    
    # Training convergence
    ax = axes[1, 1]
    if hasattr(detector.perceptron, 'errors_per_epoch'):
        ax.plot(detector.perceptron.errors_per_epoch)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Errors")
        ax.set_title("Perceptron Convergence")
    
    # Hide unused
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('vintage_ml_training_demo.png', dpi=150, bbox_inches='tight')
    print("   - Saved training visualization to vintage_ml_training_demo.png")
    
    # Save trained models
    detector.save_models('vintage_ml_models_demo.pkl')
    print("   - Saved trained models to vintage_ml_models_demo.pkl")

def main():
    """Run all demonstrations"""
    print("VintageOptics - Vintage ML Integration Demo")
    print("=" * 50 + "\n")
    
    # Run demos
    demonstrate_vintage_ml()
    demonstrate_hybrid_pipeline()
    demonstrate_training()
    
    print("\n" + "=" * 50)
    print("All demos complete! The vintage ML approach provides:")
    print("  - Transparent, interpretable defect detection")
    print("  - Lightweight computation suitable for real-time use")
    print("  - Educational value showing AI evolution")
    print("  - Effective first-pass detection before modern ML")

if __name__ == '__main__':
    main()
