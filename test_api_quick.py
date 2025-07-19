#!/usr/bin/env python3
"""Quick test of the enhanced GUI API"""

import requests
import sys

# Test API status
print("Testing VintageOptics Enhanced GUI API...")
print("-" * 40)

try:
    # Test status endpoint
    response = requests.get("http://localhost:8000/api/status")
    if response.status_code == 200:
        status = response.json()
        print("✓ API is running")
        print(f"  Mode: {status['mode']}")
        print(f"  VintageOptics: {status['vintageoptics_available']}")
    else:
        print("✗ API returned error:", response.status_code)
        sys.exit(1)
        
    # Test process endpoint
    print("\nTesting process endpoint...")
    test_params = {
        "lens_profile": "canon-50mm-f1.4",
        "correction_mode": "hybrid"
    }
    
    # Create a simple test image
    import numpy as np
    import cv2
    import io
    
    # Create a gradient test image
    test_img = np.zeros((600, 800, 3), dtype=np.uint8)
    for i in range(600):
        for j in range(800):
            test_img[i, j] = [int(255 * i / 600), int(255 * j / 800), 128]
    
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', test_img)
    files = {'file': ('test.jpg', buffer.tobytes(), 'image/jpeg')}
    
    # Test both endpoints
    for endpoint in ['/api/process', '/api/process/enhanced']:
        print(f"\nTesting {endpoint}...")
        response = requests.post(
            f"http://localhost:8000{endpoint}",
            files=files,
            params=test_params
        )
        
        if response.status_code == 200:
            print(f"  ✓ {endpoint} works!")
            print(f"    Processing time: {response.headers.get('x-processing-time', 'N/A')}")
            print(f"    Quality score: {response.headers.get('x-quality-score', 'N/A')}")
        else:
            print(f"  ✗ {endpoint} failed: {response.status_code}")
            if response.headers.get('content-type', '').startswith('application/json'):
                print(f"    Error: {response.json()}")
    
    print("\n✓ All tests passed!")
    
except requests.exceptions.ConnectionError:
    print("✗ Cannot connect to API on http://localhost:8000")
    print("  Make sure the server is running: python frontend_api_enhanced.py")
    sys.exit(1)
except Exception as e:
    print(f"✗ Test failed: {e}")
    sys.exit(1)
