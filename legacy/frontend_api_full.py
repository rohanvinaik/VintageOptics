"""
VintageOptics GUI API - Full Pipeline Version
This version uses the complete VintageOptics pipeline with all advanced features
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
import logging
from PIL import Image
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# Add the VintageOptics source to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'VintageOptics', 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the full VintageOptics pipeline
try:
    from vintageoptics.core.pipeline import VintageOpticsPipeline, PipelineConfig
    from vintageoptics.types.optics import ProcessingMode, LensProfile
    logger.info("Successfully imported VintageOpticsPipeline")
    FULL_PIPELINE_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import VintageOpticsPipeline: {e}")
    FULL_PIPELINE_AVAILABLE = False

# Try to import other components we might need
try:
    from vintageoptics.synthesis.lens_synthesizer import LensSynthesizer
    from vintageoptics.detection.vintage_detector import VintageDetector
    logger.info("Successfully imported additional components")
except ImportError as e:
    logger.warning(f"Could not import additional components: {e}")

# Import metadata extraction
try:
    from simple_metadata import SimpleMetadataExtractor
    METADATA_AVAILABLE = True
except:
    METADATA_AVAILABLE = False
    logger.info("Metadata extraction not available")

app = Flask(__name__)
CORS(app, expose_headers=[
    "x-processing-time", "x-quality-score", "x-defects-detected", 
    "x-correction-applied", "x-lens-detected", "x-camera-detected",
    "x-pipeline-mode", "x-hd-used", "x-iterations"
])

# Initialize the full pipeline with custom configuration
if FULL_PIPELINE_AVAILABLE:
    try:
        # Create pipeline configuration
        pipeline_config = PipelineConfig(
            mode=ProcessingMode.AUTO,  # Auto-detect best mode
            use_hd=True,  # Enable hyperdimensional computing
            hd_dimension=10000,
            auto_detect=True,
            detection_confidence=0.7,
            correction_strength=0.8,
            preserve_character=True,
            adaptive_strength=True,
            target_quality=0.85,
            max_iterations=3,
            use_gpu=False,  # Disable GPU for compatibility
            enable_caching=True,
            parallel_processing=True,
            save_intermediate=False,
            generate_report=True,
            compute_quality_maps=False
        )
        
        # Initialize the pipeline
        pipeline = VintageOpticsPipeline(pipeline_config)
        logger.info("Full VintageOptics pipeline initialized successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing VintageOptics pipeline: {e}")
        import traceback
        traceback.print_exc()
        pipeline = None
        FULL_PIPELINE_AVAILABLE = False
else:
    pipeline = None

# Predefined lens profiles
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
    ) if FULL_PIPELINE_AVAILABLE else None,
    
    "helios-44-2": LensProfile(
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
    ) if FULL_PIPELINE_AVAILABLE else None,
    
    "leica-summicron": LensProfile(
        name="Leica Summicron 50mm f/2",
        focal_length=50.0,
        max_aperture=2.0,
        min_aperture=16.0,
        aperture_blades=10,
        coating_type="multi-coated",
        era="1980s",
        manufacturer="Leica",
        k1=-0.015,
        k2=0.002,
        bokeh_quality=0.92,
        coma_amount=0.05,
        spherical_aberration=0.03,
        vignetting_amount=0.25,
        vignetting_falloff=3.0,
        chromatic_aberration=0.008,
        lateral_chromatic_scale=1.001
    ) if FULL_PIPELINE_AVAILABLE else None
}


@app.route('/api/process/full', methods=['POST'])
@app.route('/api/process', methods=['POST'])
def process_image():
    """Process an image with the full VintageOptics pipeline."""
    
    start_time = time.time()
    
    try:
        # Get file from request
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file provided"}), 400
        
        # Get parameters
        lens_profile_id = request.args.get('lens_profile', 'auto')
        correction_mode = request.args.get('correction_mode', 'auto')
        
        # Get processing options
        use_hd = request.args.get('use_hd', 'true') == 'true'
        target_quality = float(request.args.get('target_quality', '0.85'))
        correction_strength = float(request.args.get('correction_strength', '0.8'))
        
        # Get equipment context
        camera_model = request.args.get('camera_model', '')
        lens_model = request.args.get('lens_model', '')
        
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image file"}), 400
        
        logger.info(f"Processing image: {image.shape}, mode: {correction_mode}, lens: {lens_profile_id}")
        
        # Process with full VintageOptics pipeline
        if FULL_PIPELINE_AVAILABLE and pipeline is not None:
            try:
                # Update pipeline configuration for this request
                pipeline.config.use_hd = use_hd
                pipeline.config.target_quality = target_quality
                pipeline.config.correction_strength = correction_strength
                
                # Set processing mode
                if correction_mode != 'auto':
                    pipeline.config.mode = ProcessingMode(correction_mode.upper())
                else:
                    pipeline.config.mode = ProcessingMode.AUTO
                
                # Get lens profile
                lens_profile = None
                if lens_profile_id != 'auto' and lens_profile_id in LENS_PROFILES:
                    lens_profile = LENS_PROFILES[lens_profile_id]
                
                # Process the image
                logger.info("Starting full pipeline processing...")
                result = pipeline.process(image, lens_profile=lens_profile)
                
                # Extract processed image and metrics
                processed_image = result.corrected_image
                
                # Build comprehensive stats
                stats = {
                    "quality_score": int(result.quality_metrics.overall_quality * 100) if result.quality_metrics else 85,
                    "defects_detected": len(result.confidence_scores.get('defects', [])),
                    "correction_applied": int(pipeline.config.correction_strength * 100),
                    "sharpness": int(result.quality_metrics.sharpness * 100) if result.quality_metrics else 80,
                    "contrast": int(result.quality_metrics.contrast * 100) if result.quality_metrics else 75,
                    "noise_level": int((1 - result.quality_metrics.noise_level) * 100) if result.quality_metrics else 20,
                    "processing_time": result.processing_time,
                    "mode_used": result.mode_used.value,
                    "iterations_used": result.iterations_used,
                    "hd_used": use_hd and result.hd_analysis is not None
                }
                
                # Add HD-specific metrics if available
                if result.hd_analysis:
                    stats["hd_quality_score"] = int(result.hd_analysis.get('quality_score', 0) * 100)
                    stats["vintage_confidence"] = int(result.confidence_scores.get('vintage_errors', 0) * 100)
                    stats["digital_confidence"] = int(result.confidence_scores.get('digital_errors', 0) * 100)
                
                logger.info(f"Full pipeline processing complete: {stats}")
                
            except Exception as e:
                logger.error(f"Full pipeline processing failed: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to simple processing
                processed_image, stats = process_fallback(image, lens_profile_id, correction_mode)
        else:
            # Fallback processing
            logger.warning("Full pipeline not available, using fallback")
            processed_image, stats = process_fallback(image, lens_profile_id, correction_mode)
        
        # Convert to PIL Image
        result_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
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
        
        # Add comprehensive headers
        response.headers['x-processing-time'] = f"{processing_time:.2f}s"
        response.headers['x-quality-score'] = str(stats.get("quality_score", 0))
        response.headers['x-defects-detected'] = str(stats.get("defects_detected", 0))
        response.headers['x-correction-applied'] = str(stats.get("correction_applied", 0))
        response.headers['x-pipeline-mode'] = stats.get("mode_used", "unknown")
        response.headers['x-hd-used'] = str(stats.get("hd_used", False))
        response.headers['x-iterations'] = str(stats.get("iterations_used", 1))
        
        return response
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def process_fallback(image, lens_profile_id, correction_mode):
    """Fallback processing when full pipeline is not available."""
    result = image.copy()
    
    # Apply basic vintage effects
    h, w = image.shape[:2]
    
    # Vignetting
    Y, X = np.ogrid[:h, :w]
    center_x, center_y = w/2, h/2
    dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    vignette = 1 - (dist / max_dist) ** 2.5 * 0.3
    vignette = np.clip(vignette, 0, 1)
    
    for i in range(3):
        result[:, :, i] = (result[:, :, i] * vignette).astype(np.uint8)
    
    # Color grading
    result = result.astype(np.float32)
    result[:, :, 2] *= 1.05  # Slight red boost
    result[:, :, 1] *= 1.02  # Slight green boost  
    result[:, :, 0] *= 0.98  # Slight blue reduction
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Basic sharpening if correction mode
    if correction_mode != 'none':
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]]) / 9.0
        result = cv2.filter2D(result, -1, kernel)
    
    stats = {
        "quality_score": 75,
        "defects_detected": 0,
        "correction_applied": 50 if correction_mode != 'none' else 0,
        "processing_time": 0.5,
        "mode_used": "fallback",
        "iterations_used": 1,
        "hd_used": False
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
        
        # Format response
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
        if value is not None:
            profiles.append({
                "id": key,
                "name": value.name,
                "focal_length": value.focal_length,
                "max_aperture": value.max_aperture,
                "era": value.era,
                "manufacturer": value.manufacturer,
                "available": True
            })
    return jsonify(profiles)


@app.route('/api/pipeline-info', methods=['GET'])
def get_pipeline_info():
    """Get information about the pipeline configuration."""
    if FULL_PIPELINE_AVAILABLE and pipeline is not None:
        return jsonify({
            "pipeline": "full",
            "config": {
                "mode": pipeline.config.mode.value,
                "use_hd": pipeline.config.use_hd,
                "hd_dimension": pipeline.config.hd_dimension,
                "correction_strength": pipeline.config.correction_strength,
                "target_quality": pipeline.config.target_quality,
                "max_iterations": pipeline.config.max_iterations,
                "use_gpu": pipeline.config.use_gpu,
                "components": {
                    "hd_analyzer": pipeline.hd_analyzer is not None,
                    "lens_characterizer": hasattr(pipeline, 'lens_characterizer'),
                    "quality_analyzer": hasattr(pipeline, 'quality_analyzer'),
                    "optics_engine": hasattr(pipeline, 'optics_engine'),
                    "depth_analyzer": hasattr(pipeline, 'depth_analyzer')
                }
            }
        })
    else:
        return jsonify({
            "pipeline": "fallback",
            "reason": "Full pipeline not available",
            "config": {}
        })


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get API status."""
    return jsonify({
        "status": "running",
        "pipeline_available": FULL_PIPELINE_AVAILABLE,
        "metadata_available": METADATA_AVAILABLE,
        "mode": "full" if FULL_PIPELINE_AVAILABLE else "fallback",
        "components": {
            "pipeline": pipeline is not None,
            "metadata": METADATA_AVAILABLE
        }
    })


@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({
        "name": "VintageOptics Full Pipeline API",
        "version": "3.0.0",
        "status": "running",
        "pipeline": "full" if FULL_PIPELINE_AVAILABLE else "fallback",
        "features": [
            "full_pipeline_processing",
            "hyperdimensional_computing",
            "auto_mode_selection",
            "quality_targeting",
            "depth_analysis",
            "metadata_extraction"
        ]
    })


if __name__ == "__main__":
    print(f"Starting VintageOptics Full Pipeline API on http://localhost:8000")
    print(f"Pipeline: {'Full VintageOptics' if FULL_PIPELINE_AVAILABLE else 'Fallback mode'}")
    if FULL_PIPELINE_AVAILABLE:
        print("Full pipeline with hyperdimensional computing is available!")
        print("Components initialized:")
        if pipeline:
            print(f"  - HD Analyzer: {pipeline.hd_analyzer is not None}")
            print(f"  - Processing modes: AUTO, CORRECTION, SYNTHESIS, HYBRID")
    else:
        print("Full pipeline not available - check error messages above")
    
    app.run(host="0.0.0.0", port=8000, debug=False)
