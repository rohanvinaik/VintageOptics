"""
VintageOptics GUI API - Full Pipeline Version with Fixed Imports
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

# Try importing the full pipeline with proper error handling
FULL_PIPELINE_AVAILABLE = False
pipeline = None

try:
    # First, let's try a minimal import to see what works
    from vintageoptics.types.optics import ProcessingMode, LensProfile
    logger.info("Successfully imported types")
    
    # Try importing the simplified pipeline wrapper if it exists
    try:
        from vintageoptics.core.pipeline import ProcessingPipeline
        from vintageoptics.physics.optics_engine import OpticsEngine
        from vintageoptics.synthesis.lens_synthesizer import LensSynthesizer
        FULL_PIPELINE_AVAILABLE = True
        logger.info("Using simplified ProcessingPipeline")
    except ImportError:
        logger.warning("ProcessingPipeline not available, trying VintageOpticsPipeline")
        
        # Try the full pipeline
        from vintageoptics.core.pipeline import VintageOpticsPipeline, PipelineConfig
        
        # Create simplified config without hyperdimensional features
        pipeline_config = PipelineConfig(
            mode=ProcessingMode.AUTO,
            use_hd=False,  # Disable HD to avoid import issues
            auto_detect=True,
            correction_strength=0.8,
            preserve_character=True,
            target_quality=0.85,
            max_iterations=3,
            use_gpu=False,
            enable_caching=True,
            generate_report=False
        )
        
        pipeline = VintageOpticsPipeline(pipeline_config)
        FULL_PIPELINE_AVAILABLE = True
        logger.info("Full VintageOpticsPipeline initialized")
        
except ImportError as e:
    logger.error(f"Failed to import VintageOptics: {e}")
    import traceback
    traceback.print_exc()
    
    # Try a fallback approach - import individual components
    try:
        logger.info("Attempting modular import approach...")
        
        # Import types
        from vintageoptics.types.optics import ProcessingMode, LensProfile
        
        # Import individual components
        components = {}
        
        # Try physics engine
        try:
            from vintageoptics.physics.optics_engine import OpticsEngine
            components['optics_engine'] = OpticsEngine()
            logger.info("✓ OpticsEngine loaded")
        except Exception as e:
            logger.warning(f"✗ OpticsEngine failed: {e}")
            
        # Try synthesis
        try:
            from vintageoptics.synthesis.lens_synthesizer import LensSynthesizer
            components['lens_synthesizer'] = LensSynthesizer()
            logger.info("✓ LensSynthesizer loaded")
        except Exception as e:
            logger.warning(f"✗ LensSynthesizer failed: {e}")
            
        # Try analysis
        try:
            from vintageoptics.analysis import QualityAnalyzer
            components['quality_analyzer'] = QualityAnalyzer()
            logger.info("✓ QualityAnalyzer loaded")
        except Exception as e:
            logger.warning(f"✗ QualityAnalyzer failed: {e}")
            
        # Create a minimal pipeline wrapper
        class MinimalPipeline:
            def __init__(self, components):
                self.components = components
                self.config = type('Config', (), {
                    'correction_strength': 0.8,
                    'mode': ProcessingMode.AUTO
                })()
                
            def process(self, image, lens_profile=None):
                """Minimal processing using available components"""
                result = type('Result', (), {})()
                result.corrected_image = image.copy()
                result.processing_time = 0
                result.mode_used = ProcessingMode.HYBRID
                result.iterations_used = 1
                result.confidence_scores = {}
                
                # Apply optics engine if available
                if 'optics_engine' in self.components and lens_profile:
                    try:
                        result.corrected_image = self.components['optics_engine'].apply_lens_model(
                            result.corrected_image, lens_profile
                        )
                        logger.info("Applied optics engine")
                    except Exception as e:
                        logger.warning(f"Optics engine failed: {e}")
                
                # Apply synthesis if available
                if 'lens_synthesizer' in self.components and lens_profile:
                    try:
                        result.corrected_image = self.components['lens_synthesizer'].apply(
                            result.corrected_image, lens_profile, strength=0.8
                        )
                        logger.info("Applied lens synthesis")
                    except Exception as e:
                        logger.warning(f"Lens synthesis failed: {e}")
                
                # Analyze quality if available
                if 'quality_analyzer' in self.components:
                    try:
                        result.quality_metrics = self.components['quality_analyzer'].analyze(
                            result.corrected_image
                        )
                        logger.info("Analyzed quality")
                    except Exception as e:
                        logger.warning(f"Quality analysis failed: {e}")
                        result.quality_metrics = None
                
                return result
        
        if components:
            pipeline = MinimalPipeline(components)
            FULL_PIPELINE_AVAILABLE = True
            logger.info(f"Created minimal pipeline with {len(components)} components")
        
    except Exception as e:
        logger.error(f"Modular import also failed: {e}")
        traceback.print_exc()

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
    "x-pipeline-mode", "x-components-used"
])

# Predefined lens profiles
LENS_PROFILES = {}

if FULL_PIPELINE_AVAILABLE:
    try:
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
            ),
            
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
            )
        }
    except Exception as e:
        logger.error(f"Failed to create lens profiles: {e}")


@app.route('/api/process/full', methods=['POST'])
@app.route('/api/process', methods=['POST'])
def process_image():
    """Process an image with available VintageOptics components."""
    
    start_time = time.time()
    
    try:
        # Get file from request
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file provided"}), 400
        
        # Get parameters
        lens_profile_id = request.args.get('lens_profile', 'auto')
        correction_mode = request.args.get('correction_mode', 'auto')
        correction_strength = float(request.args.get('correction_strength', '0.8'))
        
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image file"}), 400
        
        logger.info(f"Processing image: {image.shape}, mode: {correction_mode}, lens: {lens_profile_id}")
        
        # Process with available pipeline
        if FULL_PIPELINE_AVAILABLE and pipeline is not None:
            try:
                # Update config if possible
                if hasattr(pipeline, 'config'):
                    pipeline.config.correction_strength = correction_strength
                    if correction_mode != 'auto' and hasattr(ProcessingMode, correction_mode.upper()):
                        pipeline.config.mode = ProcessingMode(correction_mode.upper())
                
                # Get lens profile
                lens_profile = None
                if lens_profile_id != 'auto' and lens_profile_id in LENS_PROFILES:
                    lens_profile = LENS_PROFILES[lens_profile_id]
                
                # Process the image
                logger.info("Processing with available pipeline...")
                result = pipeline.process(image, lens_profile=lens_profile)
                
                # Extract processed image and metrics
                processed_image = result.corrected_image
                
                # Build stats based on what's available
                stats = {
                    "quality_score": 85,
                    "defects_detected": 0,
                    "correction_applied": int(correction_strength * 100),
                    "processing_time": getattr(result, 'processing_time', time.time() - start_time),
                    "mode_used": getattr(result, 'mode_used', ProcessingMode.HYBRID).value,
                    "iterations_used": getattr(result, 'iterations_used', 1),
                    "components_used": []
                }
                
                # Add quality metrics if available
                if hasattr(result, 'quality_metrics') and result.quality_metrics:
                    qm = result.quality_metrics
                    stats["quality_score"] = int(getattr(qm, 'overall_quality', 0.85) * 100)
                    stats["sharpness"] = int(getattr(qm, 'sharpness', 0.8) * 100)
                    stats["contrast"] = int(getattr(qm, 'contrast', 0.75) * 100)
                    stats["noise_level"] = int(getattr(qm, 'noise_level', 0.2) * 100)
                
                # Track which components were used
                if hasattr(pipeline, 'components'):
                    stats["components_used"] = list(pipeline.components.keys())
                
                logger.info(f"Processing complete: {stats}")
                
            except Exception as e:
                logger.error(f"Pipeline processing failed: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to basic processing
                processed_image, stats = process_basic(image, lens_profile_id, correction_mode)
        else:
            # Basic processing
            logger.warning("No pipeline available, using basic processing")
            processed_image, stats = process_basic(image, lens_profile_id, correction_mode)
        
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
        
        # Add headers
        response.headers['x-processing-time'] = f"{processing_time:.2f}s"
        response.headers['x-quality-score'] = str(stats.get("quality_score", 0))
        response.headers['x-defects-detected'] = str(stats.get("defects_detected", 0))
        response.headers['x-correction-applied'] = str(stats.get("correction_applied", 0))
        response.headers['x-pipeline-mode'] = stats.get("mode_used", "basic")
        response.headers['x-components-used'] = ','.join(stats.get("components_used", []))
        
        return response
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def process_basic(image, lens_profile_id, correction_mode):
    """Basic processing with vintage effects."""
    result = image.copy()
    
    # Apply vintage effects based on lens profile
    h, w = image.shape[:2]
    
    # Vignetting
    Y, X = np.ogrid[:h, :w]
    center_x, center_y = w/2, h/2
    dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    # Different vignetting for different lenses
    if lens_profile_id == "helios-44-2":
        vignette_amount = 0.4
        falloff = 2.0
    elif lens_profile_id == "canon-50mm-f1.4":
        vignette_amount = 0.3
        falloff = 2.5
    else:
        vignette_amount = 0.25
        falloff = 2.2
    
    vignette = 1 - (dist / max_dist) ** falloff * vignette_amount
    vignette = np.clip(vignette, 0, 1)
    
    # Apply vignetting
    result = result.astype(np.float32)
    for i in range(3):
        result[:, :, i] *= vignette
    
    # Color grading based on lens era
    if "helios" in lens_profile_id.lower():
        # Warm, swirly bokeh character
        result[:, :, 2] *= 1.08  # Red boost
        result[:, :, 1] *= 1.05  # Green boost
        result[:, :, 0] *= 0.95  # Blue reduction
    elif "canon" in lens_profile_id.lower():
        # Slightly warm, natural
        result[:, :, 2] *= 1.04
        result[:, :, 1] *= 1.02
        result[:, :, 0] *= 0.98
    
    # Simple lens distortion
    if correction_mode == "synthesis":
        # Apply barrel distortion for vintage look
        k1 = 0.01 if "helios" in lens_profile_id.lower() else -0.02
        camera_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.array([k1, 0, 0, 0, 0], dtype=np.float32)
        result = cv2.undistort(result, camera_matrix, dist_coeffs)
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Basic sharpening if correction mode
    if correction_mode in ["correction", "hybrid"]:
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]]) / 9.0
        result = cv2.filter2D(result, -1, kernel)
    
    stats = {
        "quality_score": 75,
        "defects_detected": 0,
        "correction_applied": 50 if correction_mode != "none" else 0,
        "processing_time": 0.5,
        "mode_used": "basic",
        "iterations_used": 1,
        "components_used": ["basic_vignetting", "basic_color", "basic_distortion"]
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
                "era": getattr(value, 'era', 'Unknown'),
                "manufacturer": getattr(value, 'manufacturer', 'Unknown'),
                "available": True
            })
    
    # Add basic profiles even if full pipeline isn't available
    if not profiles:
        profiles = [
            {
                "id": "canon-50mm-f1.4",
                "name": "Canon FD 50mm f/1.4",
                "focal_length": 50.0,
                "max_aperture": 1.4,
                "era": "1970s",
                "manufacturer": "Canon",
                "available": True
            },
            {
                "id": "helios-44-2",
                "name": "Helios 44-2 58mm f/2",
                "focal_length": 58.0,
                "max_aperture": 2.0,
                "era": "1960s",
                "manufacturer": "KMZ",
                "available": True
            }
        ]
    
    return jsonify(profiles)


@app.route('/api/pipeline-info', methods=['GET'])
def get_pipeline_info():
    """Get information about the pipeline configuration."""
    info = {
        "pipeline": "none",
        "components": {},
        "available_modes": ["correction", "synthesis", "hybrid", "auto"]
    }
    
    if FULL_PIPELINE_AVAILABLE and pipeline is not None:
        info["pipeline"] = type(pipeline).__name__
        
        # Check which components are available
        if hasattr(pipeline, 'components'):
            info["components"] = {name: True for name in pipeline.components.keys()}
        else:
            # Check standard components
            components_to_check = [
                'optics_engine', 'lens_synthesizer', 'quality_analyzer',
                'lens_characterizer', 'depth_analyzer', 'hd_analyzer'
            ]
            for comp in components_to_check:
                if hasattr(pipeline, comp):
                    info["components"][comp] = getattr(pipeline, comp) is not None
        
        # Get config info if available
        if hasattr(pipeline, 'config'):
            config = pipeline.config
            info["config"] = {
                "mode": getattr(config, 'mode', ProcessingMode.AUTO).value,
                "correction_strength": getattr(config, 'correction_strength', 0.8),
                "use_hd": getattr(config, 'use_hd', False)
            }
    
    return jsonify(info)


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get API status."""
    return jsonify({
        "status": "running",
        "pipeline_available": FULL_PIPELINE_AVAILABLE,
        "metadata_available": METADATA_AVAILABLE,
        "mode": "full" if FULL_PIPELINE_AVAILABLE else "basic",
        "lens_profiles": len(LENS_PROFILES)
    })


@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({
        "name": "VintageOptics API",
        "version": "3.1.0",
        "status": "running",
        "pipeline": "available" if FULL_PIPELINE_AVAILABLE else "basic",
        "features": [
            "lens_correction",
            "lens_synthesis",
            "vintage_effects",
            "metadata_extraction" if METADATA_AVAILABLE else "metadata_demo"
        ]
    })


if __name__ == "__main__":
    print(f"Starting VintageOptics API on http://localhost:8000")
    print(f"Pipeline: {'Available' if FULL_PIPELINE_AVAILABLE else 'Basic mode'}")
    
    if FULL_PIPELINE_AVAILABLE and pipeline:
        print(f"Pipeline type: {type(pipeline).__name__}")
        if hasattr(pipeline, 'components'):
            print(f"Components: {list(pipeline.components.keys())}")
    
    print(f"Lens profiles: {len(LENS_PROFILES)}")
    
    app.run(host="0.0.0.0", port=8000, debug=False)
