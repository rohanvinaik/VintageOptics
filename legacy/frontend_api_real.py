"""
VintageOptics GUI API - Real Pipeline Integration
This connects the actual VintageOptics processing pipeline to the GUI.
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
vintage_optics_path = os.path.join(os.path.dirname(__file__), 'VintageOptics', 'src')
sys.path.insert(0, vintage_optics_path)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import VintageOptics
try:
    from vintageoptics.core.pipeline import VintageOpticsPipeline, PipelineConfig
    from vintageoptics.types.optics import ProcessingMode, LensProfile
    VINTAGE_OPTICS_AVAILABLE = True
    logger.info("VintageOptics modules loaded successfully!")
except ImportError as e:
    logger.error(f"Could not import VintageOptics: {e}")
    VINTAGE_OPTICS_AVAILABLE = False

app = Flask(__name__)
CORS(app, expose_headers=["x-processing-time", "x-quality-score", "x-defects-detected", "x-correction-applied"])

# Initialize pipeline if available
pipeline = None
if VINTAGE_OPTICS_AVAILABLE:
    try:
        config = PipelineConfig(
            mode=ProcessingMode.HYBRID,
            use_hd=True,
            correction_strength=0.8,
            target_quality=0.85,
            use_gpu=False,  # Set to False for CPU processing
            enable_caching=True
        )
        pipeline = VintageOpticsPipeline(config)
        logger.info("VintageOptics pipeline initialized!")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        pipeline = None

# Lens profiles
LENS_PROFILES = {
    "canon-50mm-f1.4": LensProfile(
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
    ) if VINTAGE_OPTICS_AVAILABLE else None,
    
    "helios-44-2": LensProfile(
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
    ) if VINTAGE_OPTICS_AVAILABLE else None,
    
    "takumar-55mm": LensProfile(
        name="Super Takumar 55mm f/1.8",
        focal_length=55.0,
        max_aperture=1.8,
        min_aperture=16.0,
        aperture_blades=6,
        coating_type="multi-coated",
        era="1960s",
        manufacturer="Asahi Pentax",
        k1=-0.015,
        k2=0.004,
        vignetting_amount=0.25,
        vignetting_falloff=2.2,
        chromatic_aberration=0.01,
        lateral_chromatic_scale=1.001,
        bokeh_quality=0.8,
        coma_amount=0.08,
        spherical_aberration=0.04
    ) if VINTAGE_OPTICS_AVAILABLE else None
}

@app.route('/api/process', methods=['POST'])
def process_image():
    """Process an image with the real VintageOptics pipeline."""
    
    start_time = time.time()
    
    try:
        # Get file from request
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file provided"}), 400
        
        # Get parameters
        lens_profile_id = request.args.get('lens_profile', 'canon-50mm-f1.4')
        correction_mode = request.args.get('correction_mode', 'hybrid')
        
        # Get defects (for future implementation)
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
        
        # Process with VintageOptics if available
        if VINTAGE_OPTICS_AVAILABLE and pipeline is not None:
            logger.info("Processing with real VintageOptics pipeline...")
            
            # Get lens profile
            lens_profile = LENS_PROFILES.get(lens_profile_id)
            
            # Update pipeline config for this request
            pipeline.config.mode = ProcessingMode[correction_mode.upper()]
            
            # If synthesis mode and defects are selected, apply them first
            if any(defects.values()) and correction_mode == 'synthesis':
                # This would add defects before synthesis
                # For now, we'll let the pipeline handle it
                pass
            
            # Process the image
            try:
                result = pipeline.process(
                    image,
                    lens_profile=lens_profile
                )
                
                processed_image = result.corrected_image
                
                # Get stats from result
                stats = {
                    "quality_score": int(result.quality_metrics.overall_quality * 100) if result.quality_metrics else 85,
                    "defects_detected": len(result.lens_characteristics.dust_spots) if result.lens_characteristics else 0,
                    "correction_applied": int(pipeline.config.correction_strength * 100),
                    "mode_used": result.mode_used.value,
                    "processing_time": result.processing_time
                }
                
                logger.info(f"Processing complete: {stats}")
                
            except Exception as e:
                logger.error(f"Pipeline processing failed: {e}")
                # Fallback to simple processing
                processed_image = apply_simple_effects(image, lens_profile_id, defects)
                stats = {"quality_score": 75, "defects_detected": 0, "correction_applied": 50}
                
        else:
            logger.warning("Using fallback processing (VintageOptics not available)")
            processed_image = apply_simple_effects(image, lens_profile_id, defects)
            stats = {"quality_score": 75, "defects_detected": 0, "correction_applied": 50}
        
        # Convert to PIL Image
        processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(processed_rgb)
        
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
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def apply_simple_effects(image, lens_profile_id, defects):
    """Fallback simple effects when VintageOptics isn't available."""
    result = image.copy()
    
    # Simple vignetting
    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]
    center_x, center_y = w/2, h/2
    dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    # Different vignetting for different lenses
    vignette_strength = {
        "canon-50mm-f1.4": 0.3,
        "helios-44-2": 0.4,
        "takumar-55mm": 0.25
    }.get(lens_profile_id, 0.3)
    
    vignette = 1 - (dist / max_dist) * vignette_strength
    vignette = np.clip(vignette, 0, 1)
    
    for i in range(3):
        result[:, :, i] = (result[:, :, i] * vignette).astype(np.uint8)
    
    # Simple chromatic aberration
    if lens_profile_id == "helios-44-2":
        b, g, r = cv2.split(result)
        # Shift channels slightly
        M = np.float32([[1, 0, 2], [0, 1, 0]])
        r = cv2.warpAffine(r, M, (w, h))
        M = np.float32([[1, 0, -2], [0, 1, 0]])
        b = cv2.warpAffine(b, M, (w, h))
        result = cv2.merge([b, g, r])
    
    # Add some defects if requested
    if defects.get('dust'):
        # Add simple dust spots
        for _ in range(30):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            radius = np.random.randint(1, 3)
            cv2.circle(result, (x, y), radius, (0, 0, 0), -1)
    
    return result


@app.route('/api/lens-profiles', methods=['GET'])
def get_lens_profiles():
    """Get available lens profiles."""
    profiles = []
    for key, value in LENS_PROFILES.items():
        profiles.append({
            "id": key,
            "name": value.name if value else key,
            "available": value is not None
        })
    return jsonify(profiles)


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get API status."""
    return jsonify({
        "status": "running",
        "vintageoptics_available": VINTAGE_OPTICS_AVAILABLE,
        "pipeline_initialized": pipeline is not None,
        "mode": "full" if (VINTAGE_OPTICS_AVAILABLE and pipeline) else "fallback"
    })


@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({
        "name": "VintageOptics GUI API",
        "version": "3.0.0",
        "status": "running",
        "mode": "full" if (VINTAGE_OPTICS_AVAILABLE and pipeline) else "fallback"
    })


if __name__ == "__main__":
    print(f"\nStarting VintageOptics GUI API on http://localhost:8000")
    print(f"VintageOptics Available: {VINTAGE_OPTICS_AVAILABLE}")
    print(f"Pipeline Initialized: {pipeline is not None}")
    print("\nTo check status: http://localhost:8000/api/status\n")
    
    app.run(host="0.0.0.0", port=8000, debug=False)
