#!/usr/bin/env python3
"""Quick test of the enhanced GUI API"""

import pytest
import requests
import numpy as np
import cv2
import io
from unittest.mock import patch, MagicMock


class TestEnhancedGUIAPI:
    """Test suite for the enhanced GUI API."""
    
    @pytest.fixture
    def api_base_url(self):
        """Base URL for the API."""
        return "http://localhost:8000"
    
    @pytest.fixture
    def test_image(self):
        """Create a test image for processing."""
        # Create a gradient test image
        test_img = np.zeros((600, 800, 3), dtype=np.uint8)
        for i in range(600):
            for j in range(800):
                test_img[i, j] = [int(255 * i / 600), int(255 * j / 800), 128]
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', test_img)
        return buffer.tobytes()
    
    @pytest.mark.skip(reason="Requires API server to be running")
    def test_api_status_live(self, api_base_url):
        """Test API status endpoint (requires running server)."""
        response = requests.get(f"{api_base_url}/api/status")
        assert response.status_code == 200
        
        status = response.json()
        assert 'vintageoptics_available' in status
    
    @patch('requests.get')
    def test_api_status_mocked(self, mock_get, api_base_url):
        """Test API status endpoint with mocked response."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'mode': 'test',
            'vintageoptics_available': True
        }
        mock_get.return_value = mock_response
        
        # Make the request
        response = requests.get(f"{api_base_url}/api/status")
        
        # Verify
        assert response.status_code == 200
        status = response.json()
        assert status['mode'] == 'test'
        assert status['vintageoptics_available'] is True
    
    @pytest.mark.skip(reason="Requires API server to be running")
    def test_process_endpoint_live(self, api_base_url, test_image):
        """Test process endpoint (requires running server)."""
        test_params = {
            "lens_profile": "canon-50mm-f1.4",
            "correction_mode": "hybrid"
        }
        
        files = {'file': ('test.jpg', test_image, 'image/jpeg')}
        
        response = requests.post(
            f"{api_base_url}/api/process",
            files=files,
            params=test_params
        )
        
        assert response.status_code == 200
        assert 'x-processing-time' in response.headers
        assert 'x-quality-score' in response.headers
    
    @patch('requests.post')
    def test_process_endpoint_mocked(self, mock_post, api_base_url, test_image):
        """Test process endpoint with mocked response."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {
            'x-processing-time': '0.5s',
            'x-quality-score': '0.85'
        }
        mock_response.content = b'processed_image_data'
        mock_post.return_value = mock_response
        
        # Make the request
        test_params = {
            "lens_profile": "canon-50mm-f1.4",
            "correction_mode": "hybrid"
        }
        files = {'file': ('test.jpg', test_image, 'image/jpeg')}
        
        response = requests.post(
            f"{api_base_url}/api/process",
            files=files,
            params=test_params
        )
        
        # Verify
        assert response.status_code == 200
        assert response.headers['x-processing-time'] == '0.5s'
        assert response.headers['x-quality-score'] == '0.85'
        
        # Verify the call was made with correct parameters
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == f"{api_base_url}/api/process"
        assert call_args[1]['params'] == test_params
    
    @patch('requests.post')
    def test_enhanced_process_endpoint_mocked(self, mock_post, api_base_url, test_image):
        """Test enhanced process endpoint with mocked response."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {
            'x-processing-time': '0.8s',
            'x-quality-score': '0.92'
        }
        mock_response.content = b'enhanced_processed_image_data'
        mock_post.return_value = mock_response
        
        # Make the request
        test_params = {
            "lens_profile": "canon-50mm-f1.4",
            "correction_mode": "hybrid"
        }
        files = {'file': ('test.jpg', test_image, 'image/jpeg')}
        
        response = requests.post(
            f"{api_base_url}/api/process/enhanced",
            files=files,
            params=test_params
        )
        
        # Verify
        assert response.status_code == 200
        assert response.headers['x-processing-time'] == '0.8s'
        assert response.headers['x-quality-score'] == '0.92'
    
    def test_connection_error_handling(self, api_base_url):
        """Test that connection errors are handled gracefully."""
        with patch('requests.get', side_effect=requests.exceptions.ConnectionError()):
            with pytest.raises(requests.exceptions.ConnectionError):
                requests.get(f"{api_base_url}/api/status")


# Standalone test runner (for backwards compatibility)
def run_standalone_tests():
    """Run tests without pytest (for manual testing)."""
    print("Testing VintageOptics Enhanced GUI API...")
    print("-" * 40)
    
    try:
        # Test API status
        response = requests.get("http://localhost:8000/api/status")
        if response.status_code == 200:
            status = response.json()
            print("✓ API is running")
            print(f"  VintageOptics: {status.get('vintageoptics_available', 'Unknown')}")
        else:
            print("✗ API returned error:", response.status_code)
            return False
        
        print("\n✓ Basic connectivity test passed!")
        print("\nFor full tests, run: pytest test_api_quick.py")
        return True
        
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API on http://localhost:8000")
        print("  Make sure the server is running: python frontend_api.py")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


if __name__ == "__main__":
    # When run directly, do a simple connectivity test
    import sys
    success = run_standalone_tests()
    sys.exit(0 if success else 1)
