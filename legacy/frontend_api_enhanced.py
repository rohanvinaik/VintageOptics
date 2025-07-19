"""
VintageOptics GUI API - Enhanced with Equipment Context
This version includes automatic metadata extraction and equipment context handling.
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
from simple_metadata import SimpleMetadataExtractor

# Add the VintageOptics source to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'VintageOptics', 'src'))

# Now import the actual VintageOptics modules
try:
    # Import the correct pipeline class
    from vintageoptics.core.pipeline import VintageOpticsPipeline, PipelineConfig
    
    from vintageoptics.physics.optics_engine import OpticsEngine
    from vintageoptics.synthesis.lens_synthesizer import LensSynthesizer
    from vintageoptics.detection.vintage_detector import VintageDetector
    from vintageoptics.integrations.exiftool import ExifToolIntegration
    from vintageoptics.types.optics import (
        LensProfile, ProcessingMode, DefectType, QualityMetrics
    )
    VINTAGE_OPTICS_AVAILABLE = True
    print("Successfully imported VintageOptics modules!")
except ImportError as e:
    print(f"Warning: Could not import VintageOptics modules: {e}")
    print("Running in demo mode with basic effects only.")
    print(f"Python path includes: {sys.path}")
    VINTAGE_OPTICS_AVAILABLE = False

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
        pipeline = VintageOpticsPipeline()
        optics_engine = OpticsEngine()
        lens_synthesizer = LensSynthesizer()
        vintage_detector = VintageDetector()
        exiftool = ExifToolIntegration()
        print("VintageOptics components initialized successfully!")
    except Exception as e:
        print(f"Error initializing VintageOptics components: {e}")
        pipeline = None
        optics_engine = None
        lens_synthesizer = None
        vintage_detector = None
        exiftool = None
        VINTAGE_OPTICS_AVAILABLE = False
else:
    pipeline = None
    optics_engine = None
    lens_synthesizer = None
    vintage_detector = None
    exiftool = None

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


@app.route('/api/process/enhanced', methods=['POST'])
def process_image_enhanced():
    """Enhanced image processing with automatic metadata extraction and equipment context."""
    
    start_time = time.time()
    
    try:
        # Get file from request
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file provided"}), 400
        
        # Get parameters
        lens_profile_id = request.args.get('lens_profile', 'auto')
        correction_mode = request.args.get('correction_mode', 'hybrid')
        
        # Get equipment context from request
        camera_model = request.args.get('camera_model', '')
        lens_model = request.args.get('lens_model', '')
        
        # Get manual override parameters
        manual_focal_length = request.args.get('focal_length', type=float)
        manual_aperture = request.args.get('aperture', type=float)
        manual_k1 = request.args.get('distortion_k1', type=float)
        manual_k2 = request.args.get('distortion_k2', type=float)
        
        # Get defects
        defects = {
            "dust": request.args.get('defect_dust') == 'true',
            "fungus": request.args.get('defect_fungus') == 'true',
            "scratches": request.args.get('defect_scratches') == 'true',
            "haze": request.args.get('defect_haze') == 'true',
            "separation": request.args.get('defect_separation') == 'true',
            "coating": request.args.get('defect_coating') == 'true'
        }
        
        # Save uploaded file temporarily for metadata extraction
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            file.save(tmp_file.name)
            temp_path = tmp_file.name
        
        # Read image for processing
        file.seek(0)
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image file"}), 400
        
        # Extract metadata from image
        metadata = {}
        lens_info = {}
        
        if VINTAGE_OPTICS_AVAILABLE and exiftool and exiftool.available:
            try:
                # Extract all metadata
                metadata = exiftool.extract_all_metadata(temp_path)
                # Extract lens-specific information
                lens_info = exiftool.extract_lens_info(temp_path)
                
                logger.info(f"Extracted metadata: {metadata.get('camera', {})}")
                logger.info(f"Extracted lens info: {lens_info}")
            except Exception as e:
                logger.warning(f"Metadata extraction failed: {e}")
        
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass
        
        # Determine lens profile to use
        lens_profile = None
        
        if lens_profile_id == 'auto':
            # Try to match lens from metadata
            detected_lens = None
            
            if lens_info.get('model'):
                # Try to match against known profiles
                lens_model_lower = lens_info['model'].lower()
                for profile_id, profile_data in LENS_PROFILES.items():
                    if any(term in lens_model_lower for term in 
                          profile_data['name'].lower().split()):
                        detected_lens = profile_id
                        logger.info(f"Auto-detected lens profile: {profile_id}")
                        break
            
            if detected_lens:
                lens_profile_id = detected_lens
            else:
                # Use equipment context if provided
                if lens_model:
                    # Create custom profile based on provided info
                    lens_profile = create_custom_profile_from_context(
                        lens_model, camera_model, lens_info, 
                        manual_focal_length, manual_aperture,
                        manual_k1, manual_k2
                    )
                else:
                    # Default to a generic profile
                    lens_profile_id = 'canon-50mm-f1.4'
        
        elif lens_profile_id == 'custom':
            # Create custom profile from manual parameters
            lens_profile = create_custom_profile_from_context(
                lens_model or "Custom Lens", 
                camera_model or "Unknown Camera",
                lens_info,
                manual_focal_length, manual_aperture,
                manual_k1, manual_k2
            )
        
        # Get lens profile if not custom
        if not lens_profile and lens_profile_id != 'custom':
            lens_data = LENS_PROFILES.get(lens_profile_id)
            if lens_data and lens_data['profile']:
                lens_profile = lens_data['profile']
        
        # Process with VintageOptics if available
        if VINTAGE_OPTICS_AVAILABLE and pipeline is not None and lens_profile:
            try:
                result, stats = process_with_vintageoptics_enhanced(
                    image, lens_profile, correction_mode, defects,
                    metadata, lens_info
                )
                logger.info(f"Processing with full VintageOptics pipeline")
            except Exception as e:
                logger.error(f"VintageOptics processing failed: {e}")
                # Fallback to demo if VintageOptics fails
                result, stats = process_demo(
                    image, lens_profile_id, correction_mode, defects
                )
                logger.info(f"Fell back to demo mode due to error")
        else:
            # Use demo processing if VintageOptics not available
            result, stats = process_demo(
                image, lens_profile_id, correction_mode, defects
            )
            logger.info(f"Processing in demo mode (VintageOptics available: {VINTAGE_OPTICS_AVAILABLE})")
        
        # Add metadata to the processed image if possible
        if VINTAGE_OPTICS_AVAILABLE and exiftool and exiftool.available and lens_profile:
            try:
                # Save result temporarily
                result_path = None
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_result:
                    cv2.imwrite(tmp_result.name, result)
                    result_path = tmp_result.name
                
                # Add processing metadata
                processing_metadata = {
                    'XMP:ProcessedWith': 'VintageOptics',
                    'XMP:LensProfileUsed': lens_profile.name if lens_profile else 'Unknown',
                    'XMP:CorrectionMode': correction_mode,
                    'XMP:OriginalCamera': camera_model or metadata.get('camera', {}).get('Model', 'Unknown'),
                    'XMP:OriginalLens': lens_model or lens_info.get('model', 'Unknown')
                }
                
                if manual_k1 is not None:
                    processing_metadata['XMP:DistortionK1Applied'] = str(manual_k1)
                if manual_k2 is not None:
                    processing_metadata['XMP:DistortionK2Applied'] = str(manual_k2)
                
                exiftool.write_metadata(result_path, processing_metadata)
                
                # Read back the result with metadata
                result = cv2.imread(result_path)
                
                # Clean up
                if result_path:
                    os.unlink(result_path)
                    
            except Exception as e:
                logger.warning(f"Failed to add metadata to result: {e}")
        
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
        response.headers['x-lens-detected'] = lens_info.get('model', 'Unknown')
        response.headers['x-camera-detected'] = metadata.get('camera', {}).get('Model', 'Unknown')
        
        return response
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return jsonify({"error": str(e)}), 500


def create_custom_profile_from_context(lens_model: str, camera_model: str, lens_info: Dict,
                                      focal_length: Optional[float] = None, 
                                      aperture: Optional[float] = None,
                                      k1: Optional[float] = None, 
                                      k2: Optional[float] = None) -> Optional[Any]:
    """Create a custom lens profile from provided context and parameters."""
    
    if not VINTAGE_OPTICS_AVAILABLE:
        return None
    
    # Extract focal length from lens model if not provided
    if focal_length is None and lens_info.get('focal_length'):
        focal_length = lens_info['focal_length']
    elif focal_length is None:
        # Try to parse from lens model string
        focal_match = re.search(r'(\d+)mm', lens_model)
        if focal_match:
            focal_length = float(focal_match.group(1))
        else:
            focal_length = 50.0  # Default
    
    # Extract aperture if not provided
    if aperture is None and lens_info.get('aperture'):
        aperture = lens_info['aperture']
    elif aperture is None:
        # Try to parse from lens model string
        aperture_match = re.search(r'f[/]?(\d+\.?\d*)', lens_model)
        if aperture_match:
            aperture = float(aperture_match.group(1))
        else:
            aperture = 2.8  # Default
    
    # Determine manufacturer and era
    manufacturer = "Unknown"
    era = "Modern"
    
    if camera_model:
        for brand in ['Canon', 'Nikon', 'Sony', 'Fujifilm', 'Olympus', 'Pentax']:
            if brand.lower() in camera_model.lower():
                manufacturer = brand
                break
    
    # Create lens profile
    profile = LensProfile(
        name=lens_model or f"Custom {focal_length}mm f/{aperture}",
        focal_length=focal_length,
        max_aperture=aperture,
        min_aperture=22.0,  # Default
        aperture_blades=7,  # Default
        coating_type="multi-coated",  # Default for modern lenses
        era=era,
        manufacturer=manufacturer,
        # Distortion coefficients
        k1=k1 or -0.01,  # Mild barrel distortion default
        k2=k2 or 0.002,
        p1=0.0,
        p2=0.0,
        # Vignetting - mild defaults
        vignetting_amount=0.2,
        vignetting_falloff=2.5,
        # Chromatic aberration - mild defaults
        chromatic_aberration=0.01,
        lateral_chromatic_scale=1.001,
        # Bokeh characteristics
        bokeh_quality=0.8,
        coma_amount=0.05,
        spherical_aberration=0.02
    )
    
    logger.info(f"Created custom profile: {profile.name}")
    return profile


def process_with_vintageoptics_enhanced(image, lens_profile, correction_mode, 
                                       defects, metadata, lens_info):
    """Enhanced processing with metadata and context awareness."""
    
    # Create pipeline config
    config = PipelineConfig(
        mode=ProcessingMode[correction_mode.upper()],
        correction_strength=0.8,
        use_hd=True,
        auto_detect=True
    )
    
    # Recreate pipeline with config
    pipeline_instance = VintageOpticsPipeline(config)
    
    # Note: Pipeline uses metadata internally
    
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
    
    # Process through the pipeline
    try:
        pipeline_result = pipeline_instance.process(image, lens_profile=lens_profile)
        
        # Extract processed image
        result = pipeline_result.corrected_image
        
        # Get quality metrics
        if pipeline_result.quality_metrics:
            stats = {
                "quality_score": int(pipeline_result.quality_metrics.overall_quality * 100),
                "defects_detected": sum([pipeline_result.lens_characteristics.dust_spots,
                                         pipeline_result.lens_characteristics.scratches,
                                         pipeline_result.lens_characteristics.fungus_areas])
                                    if pipeline_result.lens_characteristics else 0,
                "correction_applied": int(config.correction_strength * 100)
            }
        else:
            stats = {
                "quality_score": 85,
                "defects_detected": 0,
                "correction_applied": 80
            }
    except Exception as e:
        logger.error(f"Pipeline processing error: {e}")
        # Fallback to basic processing
        result = image
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
        lens_info = {}
        
        # Try metadata extraction with fallback options
        metadata = {}
        lens_info = {}
        
        # First try ExifTool if available
        try:
            if exiftool and exiftool.available:
                metadata = exiftool.extract_all_metadata(temp_path)
                lens_info = exiftool.extract_lens_info(temp_path)
                logger.info("Metadata extracted with ExifTool")
            else:
                raise Exception("ExifTool not available")
        except:
            # Fallback to simple metadata extractor
            try:
                simple_extractor = SimpleMetadataExtractor()
                simple_metadata = simple_extractor.extract_metadata(temp_path)
                lens_info = simple_extractor.extract_lens_info(temp_path)
                
                # Convert to expected format
                metadata = {
                    'camera': simple_metadata.get('camera', {}),
                    'exif': simple_metadata.get('settings', {}),
                    'basic': simple_metadata.get('basic', {})
                }
                logger.info("Metadata extracted with PIL fallback")
            except Exception as e:
                logger.warning(f"All metadata extraction failed: {e}")
                metadata = {}
                lens_info = {}
        
        # Clean up
        os.unlink(temp_path)
        
        # Format response
        response = {
            "camera": {
                "make": metadata.get('exif', {}).get('Make', 'Unknown'),
                "model": metadata.get('exif', {}).get('Model', 'Unknown'),
                "serial": metadata.get('exif', {}).get('SerialNumber', 'Unknown')
            },
            "lens": {
                "make": lens_info.get('make', 'Unknown'),
                "model": lens_info.get('model', 'Unknown'),
                "focal_length": str(lens_info.get('focal_length', 'Unknown')),
                "aperture": str(lens_info.get('aperture', 'Unknown')),
                "serial": lens_info.get('serial_number', 'Unknown')
            },
            "settings": {
                "iso": str(metadata.get('exif', {}).get('ISO', 'Unknown')),
                "shutter_speed": str(metadata.get('exif', {}).get('ExposureTime', 'Unknown')),
                "aperture": str(metadata.get('exif', {}).get('FNumber', 'Unknown')),
                "focal_length": str(metadata.get('exif', {}).get('FocalLength', 'Unknown'))
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Metadata extraction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


def process_image_simple():
    """Simple processing endpoint that works without VintageOptics modules."""
    
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
        
        # Process with demo function
        result, stats = process_demo(image, lens_profile_id, correction_mode, defects)
        
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


# Keep existing endpoints for backward compatibility
@app.route('/api/process', methods=['POST'])
def process_image():
    """Original process endpoint for backward compatibility."""
    # Use the simpler processing that we know works
    return process_image_simple()


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
            "vintage_detector": vintage_detector is not None,
            "exiftool": exiftool is not None and exiftool.available
        }
    })


@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({
        "name": "VintageOptics GUI API - Enhanced",
        "version": "2.1.0",
        "status": "running",
        "mode": "full" if VINTAGE_OPTICS_AVAILABLE else "demo",
        "features": [
            "automatic_metadata_extraction",
            "equipment_context_support",
            "manual_parameter_override",
            "custom_lens_profiles"
        ]
    })


if __name__ == "__main__":
    print(f"Starting VintageOptics Enhanced GUI API on http://localhost:8000")
    print(f"Mode: {'Full VintageOptics' if VINTAGE_OPTICS_AVAILABLE else 'Demo mode'}")
    if not VINTAGE_OPTICS_AVAILABLE:
        print("To enable full VintageOptics processing, ensure the modules are properly installed.")
    else:
        if exiftool and exiftool.available:
            print("ExifTool integration: Available")
        else:
            print("ExifTool integration: Not available (install from https://exiftool.org)")
    app.run(host="0.0.0.0", port=8000, debug=False)
