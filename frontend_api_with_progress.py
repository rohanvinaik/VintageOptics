"""
VintageOptics API with Enhanced Progress Tracking
Includes Server-Sent Events (SSE) for real-time progress updates
"""

from flask import Flask, request, send_file, jsonify, Response
from flask_cors import CORS
import numpy as np
import cv2
import io
import time
from PIL import Image
import logging
import sys
import os
from typing import Dict
import json
from queue import Queue
import threading

# Import the VintageOptics implementation
sys.path.insert(0, os.path.dirname(__file__))
from vintageoptics_complete import VintageOpticsProcessor, process_with_vintageoptics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, expose_headers=["x-processing-time", "x-quality-score", "x-defects-detected", "x-correction-applied"])

# Initialize processor
processor = VintageOpticsProcessor()

# Progress tracking
progress_queues = {}

def send_progress(task_id: str, stage: str, progress: float, message: str):
    """Send progress update to the client."""
    if task_id in progress_queues:
        progress_queues[task_id].put({
            'stage': stage,
            'progress': progress,
            'message': message,
            'timestamp': time.time()
        })

@app.route('/api/progress/<task_id>')
def progress_stream(task_id):
    """Server-Sent Events endpoint for progress updates."""
    def generate():
        # Create a queue for this task
        progress_queues[task_id] = Queue()
        
        try:
            while True:
                # Wait for progress updates
                if not progress_queues[task_id].empty():
                    update = progress_queues[task_id].get()
                    yield f"data: {json.dumps(update)}\n\n"
                else:
                    # Send heartbeat to keep connection alive
                    yield f"data: {json.dumps({'heartbeat': True})}\n\n"
                time.sleep(0.1)
        finally:
            # Clean up when client disconnects
            if task_id in progress_queues:
                del progress_queues[task_id]
    
    return Response(generate(), mimetype="text/event-stream")

@app.route('/api/restore', methods=['POST'])
def restore_image():
    """
    Restore and enhance images taken with vintage lenses.
    This endpoint REMOVES defects and corrects aberrations.
    """
    start_time = time.time()
    task_id = request.headers.get('X-Task-ID', str(time.time()))
    
    try:
        send_progress(task_id, 'initialize', 0, 'Starting restoration process...')
        
        # Get file from request
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file provided"}), 400
        
        # Get restoration options
        options = {
            'remove_defects': request.args.get('remove_defects', 'true') == 'true',
            'correct_distortion': request.args.get('correct_distortion', 'true') == 'true',
            'correct_chromatic': request.args.get('correct_chromatic', 'true') == 'true',
            'reduce_vignetting': request.args.get('reduce_vignetting', 'true') == 'true',
            'preserve_character': request.args.get('preserve_character', 'true') == 'true',
            'enhance_sharpness': request.args.get('enhance_sharpness', 'false') == 'true'
        }
        
        send_progress(task_id, 'load', 10, 'Loading and decoding image...')
        
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image file"}), 400
        
        send_progress(task_id, 'analyze', 20, 'Analyzing image characteristics...')
        
        # Process with restoration mode
        try:
            # First, analyze the image to detect its characteristics
            hd_encoding = processor.hd_processor.encode_image_errors(image)
            send_progress(task_id, 'analyze', 30, 'Performing hyperdimensional analysis...')
            
            defects = processor.defect_analyzer.analyze_defects(image)
            send_progress(task_id, 'analyze', 40, f'Detected {defects["dust"]["count"]} dust spots, {defects["scratches"]["count"]} scratches')
            
            logger.info(f"Detected {defects['total_defect_score']*100:.0f}% defect severity")
            logger.info(f"Vintage confidence: {hd_encoding.get('vintage_confidence', 0):.2f}")
            
            # Apply corrections based on options
            result = image.copy()
            current_progress = 50
            
            # Remove defects if requested
            if options['remove_defects'] and defects['total_defect_score'] > 0.1:
                send_progress(task_id, 'defects', current_progress, 'Removing dust and scratches...')
                result = remove_defects_with_progress(result, defects, task_id, current_progress)
                current_progress += 15
            
            # Correct optical issues
            if options['correct_distortion']:
                send_progress(task_id, 'distortion', current_progress, 'Correcting lens distortion...')
                result = correct_distortion(result, hd_encoding)
                current_progress += 10
                
            if options['correct_chromatic']:
                send_progress(task_id, 'chromatic', current_progress, 'Fixing chromatic aberration...')
                result = correct_chromatic_aberration(result, hd_encoding)
                current_progress += 10
                
            if options['reduce_vignetting']:
                send_progress(task_id, 'vignetting', current_progress, 'Reducing vignetting...')
                result = reduce_vignetting(result, hd_encoding)
                current_progress += 10
            
            # Enhance sharpness if requested
            if options['enhance_sharpness']:
                send_progress(task_id, 'sharpness', current_progress, 'Enhancing sharpness...')
                result = enhance_sharpness(result)
                current_progress += 5
            
            send_progress(task_id, 'finalize', 90, 'Calculating quality metrics...')
            
            # Calculate quality improvement
            quality_before = processor.quality_analyzer.analyze(image)
            quality_after = processor.quality_analyzer.analyze(result)
            
            improvement = ((quality_after['overall_quality'] - quality_before['overall_quality']) / 
                          quality_before['overall_quality'] * 100)
            
            stats = {
                "quality_score": int(quality_after['overall_quality'] * 100),
                "defects_detected": defects['dust']['count'] + defects['scratches']['count'] + defects['fungus']['count'],
                "correction_applied": int(max(0, improvement)),
                "processing_time": time.time() - start_time
            }
            
            send_progress(task_id, 'complete', 100, 'Restoration complete!')
            logger.info(f"Restoration complete: {stats}")
            
        except Exception as e:
            logger.error(f"Restoration error: {e}")
            send_progress(task_id, 'error', 0, f'Error during restoration: {str(e)}')
            result = image
            stats = {
                "quality_score": 75,
                "defects_detected": 0,
                "correction_applied": 0,
                "processing_time": time.time() - start_time
            }
        
        # Convert to PIL Image
        processed_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(processed_rgb)
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        
        # Create response
        response = send_file(
            img_byte_arr,
            mimetype='image/jpeg',
            as_attachment=False
        )
        
        # Add headers
        response.headers['x-processing-time'] = f"{stats['processing_time']:.1f}s"
        response.headers['x-quality-score'] = str(stats["quality_score"])
        response.headers['x-defects-detected'] = str(stats["defects_detected"])
        response.headers['x-correction-applied'] = str(stats["correction_applied"])
        
        return response
        
    except Exception as e:
        logger.error(f"Restore error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/synthesize', methods=['POST'])
def synthesize_vintage():
    """
    Add vintage lens effects to modern digital images.
    This endpoint ADDS character and imperfections.
    """
    start_time = time.time()
    task_id = request.headers.get('X-Task-ID', str(time.time()))
    
    try:
        send_progress(task_id, 'initialize', 0, 'Starting vintage synthesis...')
        
        # Get file from request
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file provided"}), 400
        
        # Get synthesis options
        lens_profile_id = request.args.get('lens_profile', 'helios-44-2')
        
        # Get effect strengths (multipliers)
        strengths = {
            'distortion': float(request.args.get('distortion_strength', '1.0')),
            'chromatic': float(request.args.get('chromatic_strength', '1.0')),
            'vignetting': float(request.args.get('vignetting_strength', '1.0')),
            'bokeh': float(request.args.get('bokeh_intensity', '1.0'))
        }
        
        # Get defects to add
        add_defects = {
            'dust': request.args.get('add_dust', 'false') == 'true',
            'haze': request.args.get('add_haze', 'false') == 'true',
            'coating': request.args.get('add_coating', 'false') == 'true'
        }
        
        send_progress(task_id, 'load', 10, 'Loading image...')
        
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image file"}), 400
        
        send_progress(task_id, 'profile', 20, f'Loading {lens_profile_id} lens profile...')
        
        # Get lens profile and apply strength multipliers
        lens_profile = get_lens_profile_for_synthesis(lens_profile_id)
        
        # Apply strength multipliers
        if 'distortion' in lens_profile['params']:
            for key in ['k1', 'k2', 'p1', 'p2']:
                if key in lens_profile['params']['distortion']:
                    lens_profile['params']['distortion'][key] *= strengths['distortion']
        
        if 'chromatic' in lens_profile['params']:
            lens_profile['params']['chromatic']['lateral_scale'] *= strengths['chromatic']
            lens_profile['params']['chromatic']['longitudinal'] *= strengths['chromatic']
        
        if 'vignetting' in lens_profile['params']:
            lens_profile['params']['vignetting']['amount'] *= strengths['vignetting']
        
        # Apply synthesis
        try:
            send_progress(task_id, 'optics', 30, 'Applying optical characteristics...')
            
            # Apply lens characteristics
            result = processor.physics_engine.apply_lens_model(image, lens_profile['params'])
            
            send_progress(task_id, 'optics', 50, 'Simulating lens physics...')
            
            # Apply bokeh if we have depth information
            if lens_profile['params'].get('swirly_bokeh') or strengths['bokeh'] > 0:
                send_progress(task_id, 'bokeh', 60, 'Creating depth map...')
                
                # Create simple depth map
                h, w = image.shape[:2]
                y, x = np.ogrid[:h, :w]
                center_y, center_x = h/2, w/2
                radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_radius = np.sqrt(center_x**2 + center_y**2)
                depth_map = radius / max_radius
                
                send_progress(task_id, 'bokeh', 70, f'Synthesizing {lens_profile["params"]["aperture_shape"]} bokeh...')
                
                # Apply bokeh synthesis
                result = processor.bokeh_synthesizer.synthesize_bokeh(
                    result, depth_map, lens_profile['params']
                )
            
            # Add requested defects
            if any(add_defects.values()):
                send_progress(task_id, 'defects', 80, 'Adding vintage imperfections...')
                result = add_vintage_defects_with_progress(result, add_defects, task_id)
            
            send_progress(task_id, 'finalize', 90, 'Finalizing image...')
            
            # Calculate statistics
            effects_applied = sum([
                1 if strengths['distortion'] > 0 else 0,
                1 if strengths['chromatic'] > 0 else 0,
                1 if strengths['vignetting'] > 0 else 0,
                1 if strengths['bokeh'] > 0 else 0,
                sum([1 for d in add_defects.values() if d])
            ])
            
            stats = {
                "quality_score": 85,  # Synthesis intentionally reduces "technical" quality
                "defects_detected": effects_applied,  # Reuse field for effects applied
                "correction_applied": int(np.mean(list(strengths.values())) * 100),  # Average strength
                "processing_time": time.time() - start_time
            }
            
            send_progress(task_id, 'complete', 100, 'Vintage synthesis complete!')
            logger.info(f"Synthesis complete: {stats}")
            
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            send_progress(task_id, 'error', 0, f'Error during synthesis: {str(e)}')
            result = image
            stats = {
                "quality_score": 90,
                "defects_detected": 0,
                "correction_applied": 0,
                "processing_time": time.time() - start_time
            }
        
        # Convert to PIL Image
        processed_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(processed_rgb)
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        
        # Create response
        response = send_file(
            img_byte_arr,
            mimetype='image/jpeg',
            as_attachment=False
        )
        
        # Add headers
        response.headers['x-processing-time'] = f"{stats['processing_time']:.1f}s"
        response.headers['x-quality-score'] = str(stats["quality_score"])
        response.headers['x-defects-detected'] = str(stats["defects_detected"])
        response.headers['x-correction-applied'] = str(stats["correction_applied"])
        
        return response
        
    except Exception as e:
        logger.error(f"Synthesize error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# Enhanced helper functions with progress tracking
def remove_defects_with_progress(image: np.ndarray, defects: Dict, task_id: str, base_progress: float) -> np.ndarray:
    """Remove detected defects from image with progress updates."""
    result = image.copy()
    
    total_defects = defects['dust']['count'] + defects['scratches']['count'] + defects['fungus']['count']
    if total_defects == 0:
        return result
    
    current_defect = 0
    
    # Remove dust
    if defects['dust']['count'] > 0:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for i, dust in enumerate(defects['dust']['locations']):
            cv2.circle(mask, dust['position'], int(np.sqrt(dust['area'])) + 2, 255, -1)
            current_defect += 1
            progress = base_progress + (current_defect / total_defects) * 15
            send_progress(task_id, 'defects', progress, f'Removing dust spot {i+1}/{defects["dust"]["count"]}')
        result = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)
    
    # Remove scratches
    if defects['scratches']['count'] > 0:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for i, scratch in enumerate(defects['scratches']['scratches']):
            cv2.line(mask, scratch['start'], scratch['end'], 255, 3)
            current_defect += 1
            progress = base_progress + (current_defect / total_defects) * 15
            send_progress(task_id, 'defects', progress, f'Removing scratch {i+1}/{defects["scratches"]["count"]}')
        result = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)
    
    # Remove fungus
    if defects['fungus']['count'] > 0:
        for i, region in enumerate(defects['fungus']['regions']):
            x, y, w, h = region['bbox']
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[y:y+h, x:x+w] = 255
            result = cv2.inpaint(result, mask, 5, cv2.INPAINT_TELEA)
            current_defect += 1
            progress = base_progress + (current_defect / total_defects) * 15
            send_progress(task_id, 'defects', progress, f'Removing fungus region {i+1}/{defects["fungus"]["count"]}')
    
    return result


def add_vintage_defects_with_progress(image: np.ndarray, defects_to_add: Dict, task_id: str) -> np.ndarray:
    """Add vintage defects with progress updates."""
    result = image.copy()
    h, w = image.shape[:2]
    
    if defects_to_add['dust']:
        send_progress(task_id, 'defects', 82, 'Adding dust particles...')
        # Add dust particles
        num_particles = np.random.randint(20, 50)
        for i in range(num_particles):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            radius = np.random.randint(1, 4)
            intensity = np.random.randint(100, 200)
            cv2.circle(result, (x, y), radius, (intensity, intensity, intensity), -1)
            if i % 10 == 0:
                send_progress(task_id, 'defects', 82 + (i/num_particles)*3, f'Adding dust particle {i+1}/{num_particles}')
    
    if defects_to_add['haze']:
        send_progress(task_id, 'defects', 85, 'Adding atmospheric haze...')
        # Add atmospheric haze
        haze = np.ones((h, w, 3), dtype=np.uint8) * 255
        alpha = 0.15  # Haze intensity
        result = cv2.addWeighted(result, 1-alpha, haze, alpha, 0)
    
    if defects_to_add['coating']:
        send_progress(task_id, 'defects', 88, 'Simulating coating degradation...')
        # Simulate coating degradation (rainbow reflections)
        y_grad, x_grad = np.ogrid[:h, :w]
        center_y, center_x = h/2, w/2
        distance = np.sqrt((x_grad - center_x)**2 + (y_grad - center_y)**2)
        
        # Create rainbow pattern
        rainbow = np.zeros((h, w, 3), dtype=np.float32)
        rainbow[:, :, 0] = np.sin(distance * 0.01) * 20  # Blue channel
        rainbow[:, :, 1] = np.sin(distance * 0.01 + np.pi/3) * 15  # Green channel
        rainbow[:, :, 2] = np.sin(distance * 0.01 + 2*np.pi/3) * 10  # Red channel
        
        result = result.astype(np.float32) + rainbow
        result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


# Separate correction functions for better progress tracking
def correct_distortion(image: np.ndarray, hd_encoding: Dict) -> np.ndarray:
    """Apply inverse distortion correction."""
    h, w = image.shape[:2]
    vintage_confidence = hd_encoding.get('vintage_confidence', 0.5)
    
    camera_matrix = np.array([[max(w, h), 0, w/2], [0, max(w, h), h/2], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([0.1 * vintage_confidence, 0.05 * vintage_confidence, 0, 0, 0], dtype=np.float32)
    
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_32FC1)
    return cv2.remap(image, map1, map2, cv2.INTER_CUBIC)


def correct_chromatic_aberration(image: np.ndarray, hd_encoding: Dict) -> np.ndarray:
    """Fix chromatic aberration."""
    b, g, r = cv2.split(image)
    h, w = image.shape[:2]
    vintage_confidence = hd_encoding.get('vintage_confidence', 0.5)
    scale_factor = 1 + 0.002 * vintage_confidence
    
    center = (w//2, h//2)
    M_r = cv2.getRotationMatrix2D(center, 0, 1/scale_factor)
    M_b = cv2.getRotationMatrix2D(center, 0, scale_factor)
    
    r_corrected = cv2.warpAffine(r, M_r, (w, h))
    b_corrected = cv2.warpAffine(b, M_b, (w, h))
    
    return cv2.merge([b_corrected, g, r_corrected])


def reduce_vignetting(image: np.ndarray, hd_encoding: Dict) -> np.ndarray:
    """Reduce vignetting effect."""
    h, w = image.shape[:2]
    vintage_confidence = hd_encoding.get('vintage_confidence', 0.5)
    
    Y, X = np.ogrid[:h, :w]
    center_x, center_y = w/2, h/2
    dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    vignette_correction = 1 + (dist / max_dist) ** 2 * 0.5 * vintage_confidence
    vignette_correction = np.clip(vignette_correction, 1, 2)
    
    result = image.astype(np.float32)
    for i in range(3):
        result[:, :, i] *= vignette_correction
    return np.clip(result, 0, 255).astype(np.uint8)


def enhance_sharpness(image: np.ndarray) -> np.ndarray:
    """Enhance image sharpness using unsharp masking."""
    blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


# [Include all lens profiles from the separated API]
def get_lens_profile_for_synthesis(lens_id: str) -> Dict:
    """Get lens profile optimized for synthesis (adding effects)."""
    profiles = {
        "helios-44-2": {
            "name": "Helios 44-2 58mm f/2",
            "params": {
                "distortion": {"k1": 0.02, "k2": 0.01, "p1": 0.001, "p2": 0.001},
                "chromatic": {"lateral_scale": 0.004, "longitudinal": 1.5},
                "vignetting": {"amount": 0.5, "falloff": 2.0},
                "aperture": 2.0,
                "aperture_shape": "circular",
                "aperture_blades": 8,
                "bokeh_size": 35,
                "swirly_bokeh": True,
                "bokeh_fringing": True
            }
        },
        "meyer-optik": {
            "name": "Meyer-Optik Trioplan",
            "params": {
                "distortion": {"k1": -0.03, "k2": 0.02},
                "chromatic": {"lateral_scale": 0.005, "longitudinal": 2.0},
                "vignetting": {"amount": 0.4, "falloff": 1.5},
                "aperture": 2.8,
                "aperture_shape": "circular",
                "aperture_blades": 15,
                "bokeh_size": 41,
                "swirly_bokeh": False,
                "bokeh_fringing": True
            }
        },
        "custom": {
            "name": "Custom Extreme",
            "params": {
                "distortion": {"k1": -0.08, "k2": 0.03, "p1": 0.002, "p2": -0.002},
                "chromatic": {"lateral_scale": 0.008, "longitudinal": 3.0},
                "vignetting": {"amount": 0.7, "falloff": 1.2},
                "aperture": 1.2,
                "aperture_shape": "cats_eye",
                "aperture_blades": 8,
                "bokeh_size": 51,
                "swirly_bokeh": True,
                "bokeh_fringing": True
            }
        },
        "canon-50mm-f1.4": {
            "name": "Canon FD 50mm f/1.4",
            "params": {
                "distortion": {"k1": -0.02, "k2": 0.005, "p1": 0.0001, "p2": -0.0001},
                "chromatic": {"lateral_scale": 0.0015, "longitudinal": 0.5},
                "vignetting": {"amount": 0.3, "falloff": 2.2},
                "aperture": 1.4,
                "aperture_shape": "circular",
                "aperture_blades": 8,
                "bokeh_size": 21,
                "swirly_bokeh": False,
                "bokeh_fringing": False
            }
        },
        "takumar-55mm": {
            "name": "Super Takumar 55mm f/1.8",
            "params": {
                "distortion": {"k1": -0.015, "k2": 0.004},
                "chromatic": {"lateral_scale": 0.001, "longitudinal": 0.3},
                "vignetting": {"amount": 0.25, "falloff": 2.5},
                "aperture": 1.8,
                "aperture_shape": "hexagonal",
                "aperture_blades": 6,
                "bokeh_size": 19,
                "swirly_bokeh": False,
                "bokeh_fringing": False
            }
        },
        "nikkor-105mm": {
            "name": "Nikkor 105mm f/2.5",
            "params": {
                "distortion": {"k1": -0.005, "k2": 0.001},
                "chromatic": {"lateral_scale": 0.0008, "longitudinal": 0.2},
                "vignetting": {"amount": 0.15, "falloff": 3.0},
                "aperture": 2.5,
                "aperture_shape": "circular",
                "aperture_blades": 9,
                "bokeh_size": 17,
                "swirly_bokeh": False,
                "bokeh_fringing": False
            }
        },
        "zeiss-planar": {
            "name": "Zeiss Planar 50mm f/1.4",
            "params": {
                "distortion": {"k1": -0.01, "k2": 0.002},
                "chromatic": {"lateral_scale": 0.0005, "longitudinal": 0.1},
                "vignetting": {"amount": 0.1, "falloff": 2.8},
                "aperture": 1.4,
                "aperture_shape": "circular",
                "aperture_blades": 9,
                "bokeh_size": 21,
                "swirly_bokeh": False,
                "bokeh_fringing": False
            }
        }
    }
    
    # Default to helios if not found
    return profiles.get(lens_id, profiles["helios-44-2"])


# Keep the old /api/process endpoint for backward compatibility
@app.route('/api/process', methods=['POST'])
def process_image_legacy():
    """Legacy endpoint - redirects to appropriate new endpoint."""
    correction_mode = request.args.get('correction_mode', 'hybrid')
    
    if correction_mode in ['correction', 'hybrid']:
        # Redirect to restore
        return restore_image()
    else:
        # Redirect to synthesize
        return synthesize_vintage()


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get API status."""
    return jsonify({
        "status": "running",
        "version": "2.1",
        "features": ["progress_tracking", "sse_updates"],
        "endpoints": {
            "/api/restore": "Clean up vintage lens photos",
            "/api/synthesize": "Add vintage effects to modern photos",
            "/api/progress/<task_id>": "Real-time progress updates"
        }
    })


@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({
        "name": "VintageOptics API v2.1",
        "message": "Enhanced with real-time progress tracking",
        "endpoints": ["/api/restore", "/api/synthesize", "/api/progress/<task_id>", "/api/status"]
    })


if __name__ == "__main__":
    print("\n" + "="*60)
    print("VintageOptics API v2.1 - With Progress Tracking")
    print("="*60)
    print("Endpoints:")
    print("  /api/restore         - Clean up photos taken with vintage lenses")
    print("  /api/synthesize      - Add vintage effects to modern photos")
    print("  /api/progress/<id>   - Real-time progress updates via SSE")
    print("="*60)
    print(f"\nStarting on http://localhost:8000\n")
    
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)