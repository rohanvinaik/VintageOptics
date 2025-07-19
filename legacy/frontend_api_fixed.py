"""
VintageOptics GUI API - Fixed Enhanced Version
This version works with the actual VintageOptics structure
"""

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import io
import sys
import os
import time
import tempfile
import re
import logging
from PIL import Image
from typing import Dict, Any, Optional, Tuple

# Add the VintageOptics source to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'VintageOptics', 'src'))

# Import with compatibility layer
try:
    # Try direct import first
    from vintageoptics.core.pipeline import ProcessingPipeline
    from vintageoptics.physics.optics_engine import OpticsEngine
    from vintageoptics.synthesis.lens_synthesizer import LensSynthesizer
    from vintageoptics.detection.vintage_detector import VintageDetector
    from vintageoptics.types.optics import (
        LensProfile, ProcessingMode, DefectType, QualityMetrics
    )
    VINTAGE_OPTICS_AVAILABLE = True
    print("Direct VintageOptics import successful")
except ImportError:
    try:
        # Use compatibility layer
        from vintageoptics_compat import (
            ProcessingPipeline, OpticsEngine, LensSynthesizer, 
            VintageDetector, LensProfile, ProcessingMode, 
            DefectType, QualityMetrics
        )
        VINTAGE_OPTICS_AVAILABLE = True
        print("Using VintageOptics compatibility layer")
    except ImportError as e:
        print(f"Warning: Could not import VintageOptics modules: {e}")
        print("Running in demo mode with basic effects only.")
        VINTAGE_OPTICS_AVAILABLE = False

# Try to import metadata extraction
try:
    from simple_metadata import SimpleMetadataExtractor
    METADATA_AVAILABLE = True
except:
    METADATA_AVAILABLE = False
    print("Metadata extraction not available")

# Import enhanced pipeline
try:
    from simple_enhanced_pipeline import get_simple_enhanced_pipeline
    ENHANCED_PIPELINE_AVAILABLE = True
    print("Simple enhanced pipeline available")
except ImportError as e:
    ENHANCED_PIPELINE_AVAILABLE = False
    print(f"Enhanced pipeline not available: {e}")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, expose_headers=[
    "x-processing-time", "x-quality-score", "x-defects-detected", 
    "x-correction-applied", "x-lens-detected", "x-camera-detected"
])

# Initialize VintageOptics components if available
if VINTAGE_OPTICS_AVAILABLE:
    try:
        pipeline = ProcessingPipeline()
        optics_engine = OpticsEngine()
        lens_synthesizer = LensSynthesizer()
        vintage_detector = VintageDetector()
        print("VintageOptics components initialized successfully!")
    except Exception as e:
        print(f"Error initializing VintageOptics components: {e}")
        pipeline = None
        optics_engine = None
        lens_synthesizer = None
        vintage_detector = None
        VINTAGE_OPTICS_AVAILABLE = False
else:
    pipeline = None
    optics_engine = None
    lens_synthesizer = None
    vintage_detector = None

# Lens profiles
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
            k1=-0.02,
            k2=0.005,
            p1=0.0001,
            p2=-0.0001,
            vignetting_amount=0.3,
            vignetting_falloff=2.5,
            chromatic_aberration=0.015,
            lateral_chromatic_scale=1.002,
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
            k1=0.01,
            k2=0.003,
            bokeh_quality=0.95,
            coma_amount=0.25,
            spherical_aberration=0.15,
            vignetting_amount=0.4,
            vignetting_falloff=2.0,
            chromatic_aberration=0.02,
            lateral_chromatic_scale=1.003
        ) if VINTAGE_OPTICS_AVAILABLE else None
    }
}


@app.route('/api/process/enhanced', methods=['POST'])
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
        
        # Get equipment context (for enhanced endpoint)
        camera_model = request.args.get('camera_model', '')
        lens_model = request.args.get('lens_model', '')
        
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
        
        # Extract metadata if requested
        metadata = {}
        if camera_model or lens_model:
            metadata = {
                'camera': {'model': camera_model},
                'lens': {'model': lens_model}
            }
        
        # Process with actual VintageOptics if available
        if VINTAGE_OPTICS_AVAILABLE and pipeline is not None:
            try:
                result, stats = process_with_vintageoptics(
                    image, lens_profile_id, correction_mode, defects
                )
            except Exception as e:
                logger.error(f"VintageOptics processing failed: {e}")
                result, stats = process_demo(
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
        response.headers['x-lens-detected'] = metadata.get('lens', {}).get('model', 'Unknown')
        response.headers['x-camera-detected'] = metadata.get('camera', {}).get('model', 'Unknown')
        
        return response
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return jsonify({"error": str(e)}), 500


def process_with_vintageoptics(image, lens_profile_id, correction_mode, defects):
    """Process using actual VintageOptics pipeline."""
    
    # Get lens profile
    lens_data = LENS_PROFILES.get(lens_profile_id)
    if not lens_data or not lens_data['profile']:
        # Create a default profile
        lens_profile = LensProfile(
            name="Default",
            focal_length=50.0,
            max_aperture=2.8
        ) if VINTAGE_OPTICS_AVAILABLE else None
    else:
        lens_profile = lens_data['profile']
    
    # Try enhanced pipeline first
    if ENHANCED_PIPELINE_AVAILABLE:
        try:
            logger.info("Using simple enhanced pipeline...")
            enhanced_pipeline = get_simple_enhanced_pipeline()
            
            # Log available components
            logger.info(f"Components loaded: {enhanced_pipeline.components_loaded}")
            
            # Process with enhanced pipeline
            enhanced_result, metrics = enhanced_pipeline.process(
                image, 
                lens_profile, 
                correction_mode
            )
            
            # Build stats from real metrics
            stats = {
                "quality_score": int(metrics.get('overall_quality', 0.85) * 100),
                "defects_detected": 0,  # Not implemented yet
                "correction_applied": int(metrics.get('correction_applied', 0.8) * 100),
                "sharpness": int(metrics.get('sharpness', 0.8) * 100),
                "contrast": int(metrics.get('contrast', 0.75) * 100),
                "processing_time": metrics.get('processing_time', 0),
                "effects_applied": metrics.get('effects_applied', 0)
            }
            
            logger.info(f"Enhanced processing complete in {metrics.get('processing_time', 0):.3f}s")
            logger.info(f"Real quality metrics: {stats}")
            logger.info(f"Effects applied: {metrics.get('effects_list', [])}")
            
            return enhanced_result, stats
            
        except Exception as e:
            logger.error(f"Enhanced pipeline failed: {e}")
            # Fall through to standard pipeline
    
    # Original pipeline code continues below...
    
    # Configure pipeline
    pipeline.configure(
        mode=correction_mode.upper(),
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
        
        # Get result (handle both old and new result formats)
        if hasattr(result, 'corrected_image'):
            processed_image = result.corrected_image
        elif hasattr(result, 'image'):
            processed_image = result.image
        else:
            processed_image = result
        
        # Get quality metrics
        if hasattr(result, 'quality_metrics') and result.quality_metrics:
            quality = result.quality_metrics
            stats = {
                "quality_score": int(getattr(quality, 'overall_quality', 0.85) * 100),
                "defects_detected": len(getattr(result, 'detected_defects', [])),
                "correction_applied": int(getattr(result, 'correction_strength', 0.8) * 100)
            }
        else:
            stats = {
                "quality_score": 85,
                "defects_detected": 0,
                "correction_applied": 80
            }
    else:
        processed_image = image_with_lens
        stats = {
            "quality_score": 70,
            "defects_detected": 0,
            "correction_applied": 0
        }
    
    return processed_image, stats


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


@app.route('/api/extract-metadata', methods=['POST'])
def extract_metadata_endpoint():
    """Extract metadata from uploaded image."""
    
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file provided"}), 400
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            file.save(tmp_file.name)
            temp_path = tmp_file.name
        
        metadata = {}
        
        # Try to extract metadata
        if METADATA_AVAILABLE:
            try:
                extractor = SimpleMetadataExtractor()
                metadata = extractor.extract_metadata(temp_path)
            except:
                pass
        
        # Clean up
        os.unlink(temp_path)
        
        # Format response with defaults
        response = {
            "camera": metadata.get('camera', {
                "make": "Unknown",
                "model": "Unknown"
            }),
            "lens": metadata.get('lens', {
                "model": "Unknown"
            }),
            "settings": metadata.get('settings', {
                "iso": "Unknown",
                "shutter_speed": "Unknown",
                "aperture": "Unknown",
                "focal_length": "Unknown"
            })
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Metadata extraction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


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
        "metadata_available": METADATA_AVAILABLE,
        "mode": "full" if VINTAGE_OPTICS_AVAILABLE else "demo",
        "components": {
            "pipeline": pipeline is not None,
            "optics_engine": optics_engine is not None,
            "lens_synthesizer": lens_synthesizer is not None,
            "vintage_detector": vintage_detector is not None,
            "metadata": METADATA_AVAILABLE
        }
    })


@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({
        "name": "VintageOptics GUI API - Fixed",
        "version": "2.2.0",
        "status": "running",
        "mode": "full" if VINTAGE_OPTICS_AVAILABLE else "demo",
        "features": [
            "image_processing",
            "metadata_extraction" if METADATA_AVAILABLE else "metadata_extraction_demo",
            "equipment_context",
            "lens_profiles"
        ]
    })


if __name__ == "__main__":
    print(f"Starting VintageOptics GUI API on http://localhost:8000")
    print(f"Mode: {'Full VintageOptics' if VINTAGE_OPTICS_AVAILABLE else 'Demo mode'}")
    print(f"Metadata: {'Available' if METADATA_AVAILABLE else 'Not available'}")
    if not VINTAGE_OPTICS_AVAILABLE:
        print("To enable full VintageOptics processing, ensure the modules are properly installed.")
    app.run(host="0.0.0.0", port=8000, debug=False)
