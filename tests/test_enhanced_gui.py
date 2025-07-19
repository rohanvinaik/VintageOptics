#!/usr/bin/env python3
"""
Test script for VintageOptics Enhanced GUI
Tests metadata extraction and equipment context features
"""

import requests
import json
import sys
import os
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000"
TEST_IMAGE = None  # Will be set based on available images

def find_test_image():
    """Find a suitable test image"""
    # Common test image locations
    possible_paths = [
        "~/Pictures",
        "~/Desktop", 
        "~/Downloads",
        "./examples",
        "./test_images",
        "./data/samples"
    ]
    
    extensions = ['.jpg', '.jpeg', '.png', '.tiff']
    
    for path_str in possible_paths:
        path = Path(path_str).expanduser()
        if path.exists():
            for ext in extensions:
                images = list(path.glob(f"*{ext}"))
                if images:
                    return str(images[0])
    
    return None

def test_api_status():
    """Test if API is running and check components"""
    print("1. Testing API Status...")
    try:
        response = requests.get(f"{API_URL}/api/status")
        if response.status_code == 200:
            status = response.json()
            print(f"   ✓ API Status: {status['status']}")
            print(f"   ✓ Mode: {status['mode']}")
            print(f"   ✓ VintageOptics Available: {status['vintageoptics_available']}")
            print("   Components:")
            for component, available in status['components'].items():
                symbol = "✓" if available else "✗"
                print(f"     {symbol} {component}: {available}")
            return True
        else:
            print(f"   ✗ API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("   ✗ Could not connect to API. Is it running?")
        print("   Run: python frontend_api_enhanced.py")
        return False

def test_metadata_extraction(image_path):
    """Test metadata extraction endpoint"""
    print("\n2. Testing Metadata Extraction...")
    
    if not os.path.exists(image_path):
        print(f"   ✗ Test image not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_URL}/api/extract-metadata", files=files)
        
        if response.status_code == 200:
            metadata = response.json()
            print("   ✓ Metadata extracted successfully")
            print(f"   Camera: {metadata['camera']['make']} {metadata['camera']['model']}")
            print(f"   Lens: {metadata['lens']['model']}")
            print(f"   Settings: ISO {metadata['settings']['iso']}, "
                  f"{metadata['settings']['shutter_speed']}s, "
                  f"{metadata['settings']['aperture']}")
            return True
        else:
            print(f"   ✗ Extraction failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_enhanced_processing(image_path):
    """Test enhanced processing with equipment context"""
    print("\n3. Testing Enhanced Processing...")
    
    if not os.path.exists(image_path):
        print(f"   ✗ Test image not found: {image_path}")
        return False
    
    try:
        # Prepare request
        with open(image_path, 'rb') as f:
            files = {'file': f}
            params = {
                'lens_profile': 'auto',
                'correction_mode': 'hybrid',
                'camera_model': 'Canon EOS 5D Mark IV',
                'lens_model': 'Canon EF 50mm f/1.4 USM',
                'focal_length': '50',
                'aperture': '2.8',
                'defect_dust': 'false',
                'defect_fungus': 'false'
            }
            
            print("   Processing with equipment context...")
            response = requests.post(
                f"{API_URL}/api/process/enhanced",
                files=files,
                params=params
            )
        
        if response.status_code == 200:
            print("   ✓ Processing completed successfully")
            print(f"   Processing Time: {response.headers.get('x-processing-time', 'Unknown')}")
            print(f"   Quality Score: {response.headers.get('x-quality-score', 'Unknown')}%")
            print(f"   Camera Detected: {response.headers.get('x-camera-detected', 'Unknown')}")
            print(f"   Lens Detected: {response.headers.get('x-lens-detected', 'Unknown')}")
            
            # Save result
            output_path = "test_enhanced_result.jpg"
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"   ✓ Result saved to: {output_path}")
            return True
        else:
            print(f"   ✗ Processing failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_lens_profiles():
    """Test lens profiles endpoint"""
    print("\n4. Testing Lens Profiles...")
    try:
        response = requests.get(f"{API_URL}/api/lens-profiles")
        if response.status_code == 200:
            profiles = response.json()
            print(f"   ✓ Found {len(profiles)} lens profiles:")
            for profile in profiles:
                symbol = "✓" if profile['available'] else "✗"
                print(f"     {symbol} {profile['name']} (id: {profile['id']})")
            return True
        else:
            print(f"   ✗ Failed to get profiles: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("VintageOptics Enhanced GUI Test Suite")
    print("=====================================\n")
    
    # Check if API is running
    if not test_api_status():
        print("\nPlease start the API first:")
        print("  python frontend_api_enhanced.py")
        sys.exit(1)
    
    # Find test image
    test_image = find_test_image()
    if not test_image:
        print("\nNo test image found. Please specify an image path.")
        print("Usage: python test_enhanced_gui.py [image_path]")
        sys.exit(1)
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    
    print(f"\nUsing test image: {test_image}")
    
    # Run tests
    results = []
    results.append(test_metadata_extraction(test_image))
    results.append(test_enhanced_processing(test_image))
    results.append(test_lens_profiles())
    
    # Summary
    print("\n=====================================")
    print("Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ {total - passed} tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
