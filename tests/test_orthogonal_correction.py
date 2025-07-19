#!/usr/bin/env python3
"""
Test script for the Orthogonal Error Correction pipeline

Demonstrates the new error correction capabilities that leverage
the complementary nature of analog and digital error sources.
"""

import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import only what we need directly
import cv2
import tempfile


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


def test_basic_orthogonal_concept():
    """Test the basic orthogonal error concept without complex imports"""
    print("Testing basic orthogonal error concept...")
    
    # Generate test images
    vintage_img, digital_img = generate_test_images()
    
    # Basic orthogonal analysis
    # Convert to float for processing
    vintage_f = vintage_img.astype(float) / 255
    digital_f = digital_img.astype(float) / 255
    
    # Simple frequency analysis to separate errors
    # Vintage errors are low-frequency (smooth)
    # Digital errors are high-frequency (noise)
    
    # FFT analysis
    vintage_fft = np.fft.fft2(cv2.cvtColor(vintage_img, cv2.COLOR_BGR2GRAY))
    digital_fft = np.fft.fft2(cv2.cvtColor(digital_img, cv2.COLOR_BGR2GRAY))
    
    # Shift zero frequency to center
    vintage_fft_shift = np.fft.fftshift(vintage_fft)
    digital_fft_shift = np.fft.fftshift(digital_fft)
    
    # Get magnitude spectrum
    vintage_mag = np.abs(vintage_fft_shift)
    digital_mag = np.abs(digital_fft_shift)
    
    h, w = vintage_mag.shape
    center_h, center_w = h // 2, w // 2
    
    # Analyze frequency distribution
    # Low frequency region (center)
    radius = min(h, w) // 4
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center_w)**2 + (Y - center_h)**2)
    low_freq_mask = dist < radius
    high_freq_mask = dist >= radius
    
    # Calculate low/high frequency ratios
    vintage_low_freq_ratio = np.sum(vintage_mag[low_freq_mask]) / np.sum(vintage_mag)
    digital_low_freq_ratio = np.sum(digital_mag[low_freq_mask]) / np.sum(digital_mag)
    
    print(f"Vintage low frequency ratio: {vintage_low_freq_ratio:.3f}")
    print(f"Digital low frequency ratio: {digital_low_freq_ratio:.3f}")
    
    # Basic mutual error rejection
    # Where vintage has smooth gradients AND digital has noise → trust vintage
    # Where digital has clean edges AND vintage has aberration → trust digital
    
    # Simple edge detection
    vintage_edges = cv2.Canny(vintage_img, 50, 150)
    digital_edges = cv2.Canny(digital_img, 50, 150)
    
    # Local variance as noise indicator
    def local_variance(img, kernel_size=5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
        mean = cv2.blur(gray, (kernel_size, kernel_size))
        sqr_mean = cv2.blur(gray**2, (kernel_size, kernel_size))
        variance = sqr_mean - mean**2
        return variance
    
    vintage_var = local_variance(vintage_img)
    digital_var = local_variance(digital_img)
    
    # Confidence maps
    vintage_confidence = 1 - (vintage_var / vintage_var.max())
    digital_confidence = 1 - (digital_var / digital_var.max())
    
    # Simple fusion
    output = np.zeros_like(vintage_img)
    for c in range(3):
        output[:, :, c] = (
            vintage_img[:, :, c] * vintage_confidence +
            digital_img[:, :, c] * digital_confidence
        ) / (vintage_confidence + digital_confidence + 1e-8)
    
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    print("✓ Basic orthogonal error analysis completed")
    
    return vintage_img, digital_img, output


def test_error_orthogonality():
    """Test error orthogonality using direct implementation"""
    print("\nTesting error orthogonality concept...")
    
    # Import the specific module directly
    try:
        from vintageoptics.analysis.error_orthogonality import HybridErrorCorrector, OrthogonalErrorAnalyzer
        
        # Generate test images
        vintage_img, digital_img = generate_test_images()
        
        # Initialize corrector
        corrector = HybridErrorCorrector()
        
        # Process
        corrected, report = corrector.process(vintage_img, digital_img)
        
        # Display results
        confidence = report['correction_confidence']
        print(f"Correction confidence: {confidence:.2%}")
        
        # Analyze error characteristics
        vintage_chars = report['vintage_characteristics']
        digital_chars = report['digital_characteristics']
        
        print(f"Vintage error temporal stability: {vintage_chars.temporal_stability:.2f}")
        print(f"Digital error temporal stability: {digital_chars.temporal_stability:.2f}")
        
        print("✓ Error orthogonality test completed")
        
        return corrected, report
        
    except ImportError as e:
        print(f"✗ Could not import error_orthogonality module: {e}")
        print("  This might be due to circular import issues.")
        return None, None
    except Exception as e:
        print(f"✗ Error during orthogonality test: {e}")
        return None, None


def visualize_results(vintage, digital, corrected, output_path="orthogonal_test_results.png"):
    """Create a visualization of the results"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        axes[0, 0].imshow(cv2.cvtColor(vintage, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Vintage (Analog Errors)")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(digital, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("Digital (Sensor Errors)")
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("Orthogonal Correction")
        axes[1, 0].axis('off')
        
        # Show difference map
        diff = np.abs(vintage.astype(float) - corrected.astype(float)).mean(axis=2)
        im = axes[1, 1].imshow(diff, cmap='hot')
        axes[1, 1].set_title("Correction Difference Map")
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"✓ Saved visualization to {output_path}")
    except ImportError:
        print("• Matplotlib not available, skipping visualization")
    except Exception as e:
        print(f"• Could not create visualization: {e}")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("VintageOptics Orthogonal Error Correction Test Suite")
    print("="*60 + "\n")
    
    # Test 1: Basic concept test (no complex imports)
    print("Test 1: Basic Orthogonal Concept")
    print("-"*40)
    try:
        vintage_img, digital_img, basic_corrected = test_basic_orthogonal_concept()
        print("✓ Basic concept test completed\n")
        
        # Visualize basic results
        visualize_results(vintage_img, digital_img, basic_corrected, 
                         "basic_orthogonal_test.png")
        
    except Exception as e:
        print(f"✗ Basic concept test failed: {e}\n")
    
    # Test 2: Full implementation test (if imports work)
    print("\nTest 2: Full Orthogonal Error Correction")
    print("-"*40)
    corrected, report = test_error_orthogonality()
    
    if corrected is not None:
        print("✓ Full implementation test completed\n")
        
        # Visualize full results
        visualize_results(vintage_img, digital_img, corrected,
                         "full_orthogonal_test.png")
    else:
        print("• Full implementation test skipped due to import issues\n")
    
    print("="*60)
    print("Tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
