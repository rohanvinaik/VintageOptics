"""
VintageOptics GUI API - Full Integration
This version properly integrates with the actual VintageOptics processing pipeline.
"""

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import io
import sys
import os
import time
from PIL import Image
import logging

# Add the VintageOptics source to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'VintageOptics', 'src'))

# Now import the actual VintageOptics modules
try:
    from vintageoptics.core.pipeline import ProcessingPipeline
    from vintageoptics.physics.optics_engine import OpticsEngine
    from vintageoptics.synthesis.lens_synthesizer import LensSynthesizer
    from vintageoptics.detection.vintage_detector import VintageDetector
    from vintageoptics.types.optics import (
        LensProfile, ProcessingMode, DefectType, QualityMetrics
    )
    VINTAGE_OPTICS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import VintageOptics modules: {e}")
    print("Running in demo mode with basic effects only.")
    VINTAGE_OPTICS_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, expose_headers=["x-processing-time", "x-quality-score", "x-defects-detected", "x-correction-applied"])

# Initialize VintageOptics components if available
if VINTAGE_OPTICS_AVAILABLE:
    pipeline = ProcessingPipeline()
    optics_engine = OpticsEngine()
    lens_synthesizer = LensSynthesizer()
    vintage_detector = VintageDetector()
else:
    pipeline = None
    optics_engine = None
    lens_synthesizer = None
    vintage_detector = None

# Lens profiles with full VintageOptics parameters
LENS_PROFILES = {
    "canon-50mm-f1.4": {
        "name": "Canon FD 50mm f/1.4",
        "profile": LensProfile(
            name="Canon FD 50mm f/1.4",
            focal_length=50.0,
            max_aperture=1.4,
            min_aperture=16.0,
            aperture_blades=8,
            coating_type="single-coated",
            era="1970s",
            manufacturer="Canon",
            # Distortion coefficients
            k1=-0.02,
            k2=0.005,
            p1=0.0001,
            p2=-0.0001,
            # Vignetting
            vignetting_amount=0.3,
            vignetting_falloff=2.5,
            # Chromatic aberration
            chromatic_aberration=0.015,
            lateral_chromatic_scale=1.002,
            # Bokeh characteristics
            bokeh_quality=0.85,
            coma_amount=0.1,
            spherical_aberration=0.05
        ) if VINTAGE_OPTICS_AVAILABLE else None
    },
    "helios-44-2": {
        "name": "Helios 44-2 58mm f/2",
        "profile": LensProfile(
            name="Helios 44-2 58mm f/2",
            focal_length=58.0,
            max_aperture=2.0,
            min_aperture=16.0,
            aperture_blades=8,
            coating_type="uncoated",
            era="1960s",
            manufacturer="KMZ",
            # Distortion - positive for pincushion
            k1=0.01,
            k2=0.003,
            # Strong swirly bokeh characteristics
            bokeh_quality=0.95,
            coma_amount=0.25,  # High coma for swirly effect
            spherical_aberration=0.15,
            # Vignetting
            vignetting_amount=0.4,
            vignetting_falloff=2.0,
            # Chromatic aberration
            chromatic_aberration=0.02,
            lateral_chromatic_scale=1.003
        ) if VINTAGE_OPTICS_AVAILABLE else None
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
        lens_profile_id = request.args.get('lens_profile', 'canon-50mm-f1.4')
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
        
        # Process with actual VintageOptics if available
        if VINTAGE_OPTICS_AVAILABLE and pipeline is not None:
            result, stats = process_with_vintageoptics(
                image, lens_profile_id, correction_mode, defects
            )
        else:
            # Fallback to demo processing
            result, stats = process_demo(
                image, lens_profile_id, correction_mode, defects
            )
        
        # Convert to PIL Image
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(result_rgb)
        
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
        response.headers['x-quality-score'] = str(stats.get("quality_score", 0))
        response.headers['x-defects-detected'] = str(stats.get("defects_detected", 0))
        response.headers['x-correction-applied'] = str(stats.get("correction_applied", 0))
        
        return response
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return jsonify({"error": str(e)}), 500


def process_with_vintageoptics(image, lens_profile_id, correction_mode, defects):
    """Process using actual VintageOptics pipeline."""
    
    # Get lens profile
    lens_data = LENS_PROFILES.get(lens_profile_id)
    if not lens_data or not lens_data['profile']:
        raise ValueError(f"Unknown lens profile: {lens_profile_id}")
    
    lens_profile = lens_data['profile']
    
    # Configure pipeline
    pipeline.configure(
        mode=ProcessingMode[correction_mode.upper()],
        lens_profile=lens_profile,
        enable_defects=any(defects.values())
    )
    
    # Apply defects if selected
    if any(defects.values()):
        defect_types = []
        if defects['dust']:
            defect_types.append(DefectType.DUST)
        if defects['fungus']:
            defect_types.append(DefectType.FUNGUS)
        if defects['scratches']:
            defect_types.append(DefectType.SCRATCHES)
        if defects['haze']:
            defect_types.append(DefectType.HAZE)
        
        # Apply vintage defects
        image = vintage_detector.add_defects(image, defect_types)
    
    # Apply lens characteristics
    image_with_lens = optics_engine.apply_lens_model(image, lens_profile)
    
    # Apply correction if needed
    if correction_mode != 'none':
        result = pipeline.process(image_with_lens)
        
        # Get quality metrics
        quality_metrics = result.quality_metrics
        stats = {
            "quality_score": int(quality_metrics.overall_quality * 100),
            "defects_detected": len(result.detected_defects),
            "correction_applied": int(result.correction_strength * 100)
        }
    else:
        result = image_with_lens
        stats = {
            "quality_score": 0,
            "defects_detected": 0,
            "correction_applied": 0
        }
    
    return result, stats


def process_demo(image, lens_profile_id, correction_mode, defects):
    """Demo processing when VintageOptics is not available."""
    result = image.copy()
    
    # Apply basic vignetting
    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]
    center_x, center_y = w/2, h/2
    dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    vignette = 1 - (dist / max_dist) * 0.3
    vignette = np.clip(vignette, 0, 1)
    
    for i in range(3):
        result[:, :, i] = (result[:, :, i] * vignette).astype(np.uint8)
    
    # Basic sharpening for "correction"
    if correction_mode != 'none':
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]]) / 9.0
        result = cv2.filter2D(result, -1, kernel)
    
    stats = {
        "quality_score": 85 if correction_mode != 'none' else 70,
        "defects_detected": sum(defects.values()),
        "correction_applied": 50 if correction_mode != 'none' else 0
    }
    
    return result, stats


@app.route('/api/lens-profiles', methods=['GET'])
def get_lens_profiles():
    """Get available lens profiles."""
    profiles = []
    for key, value in LENS_PROFILES.items():
        profiles.append({
            "id": key,
            "name": value["name"],
            "available": value["profile"] is not None
        })
    return jsonify(profiles)


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get API status."""
    return jsonify({
        "status": "running",
        "vintageoptics_available": VINTAGE_OPTICS_AVAILABLE,
        "mode": "full" if VINTAGE_OPTICS_AVAILABLE else "demo",
        "components": {
            "pipeline": pipeline is not None,
            "optics_engine": optics_engine is not None,
            "lens_synthesizer": lens_synthesizer is not None,
            "vintage_detector": vintage_detector is not None
        }
    })


@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({
        "name": "VintageOptics GUI API",
        "version": "2.0.0",
        "status": "running",
        "mode": "full" if VINTAGE_OPTICS_AVAILABLE else "demo"
    })


if __name__ == "__main__":
    print(f"Starting VintageOptics GUI API on http://localhost:8000")
    print(f"Mode: {'Full VintageOptics' if VINTAGE_OPTICS_AVAILABLE else 'Demo mode'}")
    if not VINTAGE_OPTICS_AVAILABLE:
        print("To enable full VintageOptics processing, ensure the modules are properly installed.")
    app.run(host="0.0.0.0", port=8000, debug=False)
