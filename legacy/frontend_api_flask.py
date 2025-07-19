"""
Simple Flask API for VintageOptics GUI
This version uses Flask which is simpler to install and run.
"""

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import io
import time
from PIL import Image
import logging
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, expose_headers=["x-processing-time", "x-quality-score", "x-defects-detected", "x-correction-applied"])

# Lens profiles
LENS_PROFILES = {
    "canon-50mm-f1.4": {
        "name": "Canon FD 50mm f/1.4",
        "config": {
            "focal_length": 50,
            "max_aperture": 1.4,
            "vignetting": 0.3,
            "distortion": -0.02,
            "chromatic_aberration": 0.015
        }
    },
    "helios-44-2": {
        "name": "Helios 44-2 58mm f/2",
        "config": {
            "focal_length": 58,
            "max_aperture": 2.0,
            "vignetting": 0.4,
            "distortion": 0.01,
            "chromatic_aberration": 0.02,
            "swirly_bokeh": True
        }
    },
    "takumar-55mm": {
        "name": "Super Takumar 55mm f/1.8",
        "config": {
            "focal_length": 55,
            "max_aperture": 1.8,
            "vignetting": 0.25,
            "distortion": -0.015,
            "chromatic_aberration": 0.01
        }
    },
    "nikkor-105mm": {
        "name": "Nikkor 105mm f/2.5",
        "config": {
            "focal_length": 105,
            "max_aperture": 2.5,
            "vignetting": 0.2,
            "distortion": -0.005,
            "chromatic_aberration": 0.008
        }
    },
    "zeiss-planar": {
        "name": "Zeiss Planar 50mm f/1.4",
        "config": {
            "focal_length": 50,
            "max_aperture": 1.4,
            "vignetting": 0.15,
            "distortion": -0.01,
            "chromatic_aberration": 0.005
        }
    }
}

@app.route('/api/process', methods=['POST'])
def process_image():
    """Process an image with vintage lens effects and corrections."""
    
    start_time = time.time()
    
    try:
        # Get file from request
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file provided"}), 400
        
        # Get parameters
        lens_profile = request.args.get('lens_profile', 'canon-50mm-f1.4')
        correction_mode = request.args.get('correction_mode', 'hybrid')
        
        # Get defects
        defects = {
            "dust": request.args.get('defect_dust') == 'true',
            "fungus": request.args.get('defect_fungus') == 'true',
            "scratches": request.args.get('defect_scratches') == 'true',
            "haze": request.args.get('defect_haze') == 'true',
            "separation": request.args.get('defect_separation') == 'true',
            "coating": request.args.get('defect_coating') == 'true'
        }
        
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image file"}), 400
        
        # Get lens configuration
        lens_config = LENS_PROFILES.get(lens_profile, LENS_PROFILES["canon-50mm-f1.4"])
        
        # Apply vintage effects
        if any(defects.values()):
            image = apply_vintage_defects(image, defects)
        
        # Apply lens characteristics
        image = apply_lens_characteristics(image, lens_config["config"])
        
        # Apply correction
        if correction_mode != "none":
            corrected_image, stats = apply_correction(image, correction_mode)
        else:
            corrected_image = image
            stats = {"quality_score": 0, "defects_detected": 0, "correction_applied": 0}
        
        # Convert to PIL Image
        corrected_image_rgb = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(corrected_image_rgb)
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        
        processing_time = time.time() - start_time
        
        # Create response
        response = send_file(
            img_byte_arr,
            mimetype='image/jpeg',
            as_attachment=False
        )
        
        # Add headers
        response.headers['x-processing-time'] = f"{processing_time:.1f}s"
        response.headers['x-quality-score'] = str(stats["quality_score"])
        response.headers['x-defects-detected'] = str(stats["defects_detected"])
        response.headers['x-correction-applied'] = str(stats["correction_applied"])
        
        return response
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return jsonify({"error": str(e)}), 500


def apply_vintage_defects(image: np.ndarray, defects: dict) -> np.ndarray:
    """Apply selected vintage defects."""
    result = image.copy()
    
    if defects["dust"]:
        # Add dust particles
        h, w = image.shape[:2]
        num_spots = np.random.randint(20, 50)
        for _ in range(num_spots):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            radius = np.random.randint(1, 4)
            cv2.circle(result, (x, y), radius, (0, 0, 0), -1)
    
    if defects["haze"]:
        # Add hazy effect
        blurred = cv2.GaussianBlur(image, (21, 21), 0)
        result = cv2.addWeighted(result, 0.7, blurred, 0.3, 0)
    
    if defects["scratches"]:
        # Add scratches
        h, w = image.shape[:2]
        for _ in range(np.random.randint(2, 5)):
            pt1 = (np.random.randint(0, w), np.random.randint(0, h))
            pt2 = (np.random.randint(0, w), np.random.randint(0, h))
            cv2.line(result, pt1, pt2, (200, 200, 200), np.random.randint(1, 3))
    
    return result


def apply_lens_characteristics(image: np.ndarray, config: dict) -> np.ndarray:
    """Apply lens-specific characteristics."""
    result = image.copy()
    
    # Apply vignetting
    if "vignetting" in config:
        h, w = image.shape[:2]
        Y, X = np.ogrid[:h, :w]
        center_x, center_y = w/2, h/2
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        vignette = 1 - (dist / max_dist) * config["vignetting"]
        vignette = np.clip(vignette, 0, 1)
        for i in range(3):
            result[:, :, i] = (result[:, :, i] * vignette).astype(np.uint8)
    
    # Simple chromatic aberration
    if "chromatic_aberration" in config and config["chromatic_aberration"] > 0:
        b, g, r = cv2.split(result)
        strength = config["chromatic_aberration"]
        scale_r = 1 + strength / 100
        scale_b = 1 - strength / 100
        
        h, w = image.shape[:2]
        M_r = cv2.getRotationMatrix2D((w/2, h/2), 0, scale_r)
        M_b = cv2.getRotationMatrix2D((w/2, h/2), 0, scale_b)
        
        r_scaled = cv2.warpAffine(r, M_r, (w, h))
        b_scaled = cv2.warpAffine(b, M_b, (w, h))
        
        result = cv2.merge([b_scaled, g, r_scaled])
    
    return result


def apply_correction(image: np.ndarray, mode: str) -> tuple:
    """Apply correction based on mode."""
    # Simplified correction simulation
    result = image.copy()
    
    # Basic sharpening
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(result, -1, kernel)
    
    # Blend based on mode
    if mode == "physical":
        result = cv2.addWeighted(result, 0.3, sharpened, 0.7, 0)
        stats = {"quality_score": 88, "defects_detected": 5, "correction_applied": 75}
    elif mode == "ml":
        result = cv2.addWeighted(result, 0.2, sharpened, 0.8, 0)
        stats = {"quality_score": 90, "defects_detected": 7, "correction_applied": 80}
    else:  # hybrid
        result = cv2.addWeighted(result, 0.25, sharpened, 0.75, 0)
        stats = {"quality_score": 92, "defects_detected": 8, "correction_applied": 85}
    
    return result, stats


@app.route('/api/lens-profiles', methods=['GET'])
def get_lens_profiles():
    """Get available lens profiles."""
    profiles = []
    for key, value in LENS_PROFILES.items():
        profiles.append({
            "id": key,
            "name": value["name"],
            "config": value["config"]
        })
    return jsonify(profiles)


@app.route('/api/synthesis-presets', methods=['GET'])
def get_synthesis_presets():
    """Get synthesis presets."""
    return jsonify([
        {"id": "dreamy", "name": "Dreamy Portrait", "description": "Soft, ethereal look"},
        {"id": "vintage", "name": "Vintage Film", "description": "Classic film look"},
        {"id": "artistic", "name": "Artistic Bokeh", "description": "Creative bokeh patterns"}
    ])


@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({
        "name": "VintageOptics GUI API",
        "version": "1.0.0",
        "status": "running"
    })


if __name__ == "__main__":
    print("Starting VintageOptics GUI API on http://localhost:8000")
    app.run(host="0.0.0.0", port=8000, debug=False)
