#!/usr/bin/env python3
"""
VintageOptics Real-World Demo
Creates a test image and runs it through the complete pipeline
"""

import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_realistic_test_image():
    """Create a realistic test image with distortion and defects"""
    # Create a base image with geometric patterns
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Add checkerboard pattern
    square_size = 40
    for i in range(0, 600, square_size):
        for j in range(0, 800, square_size):
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                img[i:i+square_size, j:j+square_size] = [200, 200, 200]
    
    # Add some radial lines (good for testing distortion)
    center = (400, 300)
    for angle in range(0, 360, 30):
        x2 = int(center[0] + 300 * np.cos(np.radians(angle)))
        y2 = int(center[1] + 300 * np.sin(np.radians(angle)))
        cv2.line(img, center, (x2, y2), (100, 100, 100), 2)
    
    # Add circles (good for testing vignetting)
    for radius in [50, 100, 150, 200]:
        cv2.circle(img, center, radius, (150, 150, 150), 2)
    
    # Simulate barrel distortion
    h, w = img.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Center coordinates
    cx, cy = w / 2, h / 2
    
    # Normalize coordinates
    x_norm = (map_x - cx) / cx
    y_norm = (map_y - cy) / cy
    
    # Apply barrel distortion
    r = np.sqrt(x_norm**2 + y_norm**2)
    k1 = 0.15  # Distortion coefficient
    distortion_factor = 1 + k1 * r**2
    
    x_dist = x_norm * distortion_factor
    y_dist = y_norm * distortion_factor
    
    # Convert back to pixel coordinates
    map_x_dist = (x_dist * cx + cx).astype(np.float32)
    map_y_dist = (y_dist * cy + cy).astype(np.float32)
    
    # Apply distortion
    distorted = cv2.remap(img, map_x_dist, map_y_dist, cv2.INTER_LINEAR)
    
    # Add chromatic aberration simulation
    red_scale = 1.02
    blue_scale = 0.98
    
    # Scale red channel
    map_x_red = (x_norm * red_scale * cx + cx).astype(np.float32)
    map_y_red = (y_norm * red_scale * cy + cy).astype(np.float32)
    distorted[:, :, 0] = cv2.remap(img[:, :, 0], map_x_red, map_y_red, cv2.INTER_LINEAR)
    
    # Scale blue channel
    map_x_blue = (x_norm * blue_scale * cx + cx).astype(np.float32)
    map_y_blue = (y_norm * blue_scale * cy + cy).astype(np.float32)
    distorted[:, :, 2] = cv2.remap(img[:, :, 2], map_x_blue, map_y_blue, cv2.INTER_LINEAR)
    
    # Add vignetting
    y_coords, x_coords = np.ogrid[:h, :w]
    x_centered = x_coords - cx
    y_centered = y_coords - cy
    radius_norm = np.sqrt(x_centered**2 + y_centered**2) / min(cx, cy)
    
    vignette = 1.0 - 0.3 * radius_norm**2
    vignette = np.clip(vignette, 0.3, 1.0)
    distorted = (distorted * vignette[:, :, np.newaxis]).astype(np.uint8)
    
    # Add some "dust spots"
    for _ in range(10):
        x = np.random.randint(50, w-50)
        y = np.random.randint(50, h-50)
        radius = np.random.randint(2, 8)
        cv2.circle(distorted, (x, y), radius, (0, 0, 0), -1)
    
    return distorted

def main():
    print("🎬 VintageOptics Real-World Demo")
    print("=" * 40)
    
    try:
        # Create test image
        print("📷 Creating realistic test image...")
        test_image = create_realistic_test_image()
        test_path = 'demo_input.jpg'
        output_path = 'demo_output.jpg'
        
        # Save test image
        cv2.imwrite(test_path, cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
        print(f"✅ Created test image: {test_path}")
        
        # Initialize pipeline
        print("\n🔧 Initializing VintageOptics pipeline...")
        from vintageoptics.core.pipeline import VintageOpticsPipeline, ProcessingMode, ProcessingRequest
        
        pipeline = VintageOpticsPipeline("config/default.yaml")
        print("✅ Pipeline initialized")
        
        # Create processing request
        request = ProcessingRequest(
            image_path=test_path,
            mode=ProcessingMode.CORRECT,
            output_path=output_path,
            settings={
                'distortion_k1': -0.15,  # Correct the barrel distortion we added
                'chromatic_red': 1/1.02,   # Correct the red scaling
                'chromatic_blue': 1/0.98,  # Correct the blue scaling
                'vignetting_a1': 0.3       # Correct the vignetting
            }
        )
        
        print(f"\n⚙️  Processing image with corrections...")
        print(f"   - Distortion correction: k1 = {request.settings['distortion_k1']}")
        print(f"   - Chromatic aberration: R = {request.settings['chromatic_red']:.3f}")
        print(f"   - Vignetting correction: a1 = {request.settings['vignetting_a1']}")
        
        # Process image
        result = pipeline.process(request)
        
        print(f"✅ Processing completed!")
        print(f"   - Mode: {result.mode}")
        print(f"   - Output saved: {output_path}")
        
        # Analyze quality improvement
        if hasattr(result, 'quality_metrics'):
            metrics = result.quality_metrics
            print(f"   - Quality score: {metrics.get('quality_score', 'N/A')}")
        
        # Show file sizes
        if os.path.exists(output_path):
            input_size = os.path.getsize(test_path)
            output_size = os.path.getsize(output_path)
            print(f"   - Input size: {input_size:,} bytes")
            print(f"   - Output size: {output_size:,} bytes")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"📁 Files created:")
        print(f"   - Input: {test_path}")
        print(f"   - Output: {output_path}")
        print(f"   - Compare the images to see the corrections!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup option
        cleanup = input("\n🧹 Delete demo files? (y/N): ").lower().strip()
        if cleanup == 'y':
            for path in [test_path, output_path]:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"🗑️  Deleted: {path}")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
