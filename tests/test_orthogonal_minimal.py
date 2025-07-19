#!/usr/bin/env python3
"""
Minimal test to verify orthogonal error correction concept
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def test_orthogonal_concept():
    """Test the orthogonal error correction concept with minimal dependencies"""
    
    # Create test image
    height, width = 400, 400
    y, x = np.ogrid[:height, :width]
    center = (width//2, height//2)
    radius = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # Base pattern
    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[radius < 150] = [180, 140, 100]
    
    # Vintage errors (smooth, continuous)
    vintage = base.copy().astype(float)
    # Add vignetting
    vignette = 1 - (radius / radius.max())**2
    vintage *= vignette[:, :, np.newaxis]
    # Add chromatic shift
    vintage[:, :, 0] = np.roll(vintage[:, :, 0], 3, axis=1)
    vintage[:, :, 2] = np.roll(vintage[:, :, 2], -3, axis=1)
    vintage = np.clip(vintage, 0, 255).astype(np.uint8)
    
    # Digital errors (discrete, noisy)
    digital = base.copy().astype(float)
    # Add Gaussian noise
    noise = np.random.normal(0, 25, digital.shape)
    digital += noise
    # Add salt & pepper noise
    salt_pepper = np.random.random((height, width)) 
    digital[salt_pepper > 0.995] = 255
    digital[salt_pepper < 0.005] = 0
    digital = np.clip(digital, 0, 255).astype(np.uint8)
    
    # Orthogonal error separation concept:
    # Vintage errors are low-frequency, digital errors are high-frequency
    
    # Convert to grayscale for frequency analysis
    vintage_gray = cv2.cvtColor(vintage, cv2.COLOR_BGR2GRAY)
    digital_gray = cv2.cvtColor(digital, cv2.COLOR_BGR2GRAY)
    
    # FFT to separate frequency components
    def analyze_frequency_content(img):
        fft = np.fft.fft2(img)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        h, w = img.shape
        center_h, center_w = h // 2, w // 2
        
        # Create frequency masks
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - center_w)**2 + (Y - center_h)**2)
        
        # Low frequency (center 25%)
        low_freq_mask = dist < min(h, w) * 0.25
        # High frequency (outer 75%)
        high_freq_mask = ~low_freq_mask
        
        low_freq_energy = np.sum(magnitude[low_freq_mask])
        high_freq_energy = np.sum(magnitude[high_freq_mask])
        total_energy = np.sum(magnitude)
        
        return {
            'low_freq_ratio': low_freq_energy / total_energy,
            'high_freq_ratio': high_freq_energy / total_energy,
            'fft_magnitude': magnitude
        }
    
    vintage_freq = analyze_frequency_content(vintage_gray)
    digital_freq = analyze_frequency_content(digital_gray)
    
    print(f"Vintage image - Low freq ratio: {vintage_freq['low_freq_ratio']:.3f}")
    print(f"Digital image - Low freq ratio: {digital_freq['low_freq_ratio']:.3f}")
    
    # Mutual error rejection based on orthogonality
    # Simple approach: blend based on local characteristics
    
    # Compute local variance as noise indicator
    kernel_size = 7
    vintage_var = cv2.Laplacian(vintage_gray, cv2.CV_64F).var()
    digital_var = cv2.Laplacian(digital_gray, cv2.CV_64F).var()
    
    # Vintage has lower variance (smoother)
    # Digital has higher variance (noisier)
    print(f"\nVintage variance: {vintage_var:.1f}")
    print(f"Digital variance: {digital_var:.1f}")
    
    # Simple confidence-based blending
    # Trust vintage more in smooth areas, digital more at edges
    edges = cv2.Canny(base, 50, 150)
    edge_mask = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=2)
    
    # Normalize to 0-1
    edge_weight = edge_mask.astype(float) / 255
    smooth_weight = 1 - edge_weight
    
    # Blend images
    corrected = np.zeros_like(vintage)
    for c in range(3):
        corrected[:, :, c] = (
            vintage[:, :, c] * smooth_weight +
            digital[:, :, c] * edge_weight
        )
    
    corrected = corrected.astype(np.uint8)
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Images
    axes[0, 0].imshow(cv2.cvtColor(vintage, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Vintage (Analog Errors)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(digital, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Digital (Sensor Errors)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Orthogonal Correction')
    axes[0, 2].axis('off')
    
    # Frequency analysis
    axes[1, 0].imshow(np.log(vintage_freq['fft_magnitude'] + 1), cmap='hot')
    axes[1, 0].set_title('Vintage Frequency Spectrum')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(np.log(digital_freq['fft_magnitude'] + 1), cmap='hot')
    axes[1, 1].set_title('Digital Frequency Spectrum')
    axes[1, 1].axis('off')
    
    # Weight map
    axes[1, 2].imshow(edge_weight, cmap='gray')
    axes[1, 2].set_title('Edge Weight Map')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('orthogonal_concept_demo.png', dpi=150)
    plt.show()
    
    print("\nOrthogonal error correction concept demonstrated!")
    print("Results saved to 'orthogonal_concept_demo.png'")

if __name__ == "__main__":
    test_orthogonal_concept()
