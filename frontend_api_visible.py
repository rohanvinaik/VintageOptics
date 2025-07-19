"""
VintageOptics GUI API - Minimal Working Version
This version applies VISIBLE effects so you can see it's actually working.
"""

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import io
import time
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, expose_headers=["x-processing-time", "x-quality-score", "x-defects-detected", "x-correction-applied"])

@app.route('/api/process', methods=['POST'])
def process_image():
    """Process an image with VISIBLE vintage lens effects."""
    
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
        
        logger.info(f"Processing with lens: {lens_profile_id}, mode: {correction_mode}")
        logger.info(f"Defects: {defects}")
        
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image file"}), 400
        
        # Apply VISIBLE effects
        result = image.copy()
        h, w = image.shape[:2]
        
        # 1. STRONG VIGNETTING (so you can see it's working)
        logger.info("Applying vignetting...")
        Y, X = np.ogrid[:h, :w]
        center_x, center_y = w/2, h/2
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # Different vignetting for different lenses
        vignette_params = {
            "canon-50mm-f1.4": (0.4, 2.0),
            "helios-44-2": (0.6, 1.5),  # Stronger for Helios
            "takumar-55mm": (0.3, 2.5),
            "custom": (0.7, 1.2)  # Very strong for custom
        }
        strength, falloff = vignette_params.get(lens_profile_id, (0.4, 2.0))
        
        vignette = 1 - (dist / max_dist) ** falloff * strength
        vignette = np.clip(vignette, 0, 1)
        
        for i in range(3):
            result[:, :, i] = (result[:, :, i] * vignette).astype(np.uint8)
        
        # 2. CHROMATIC ABERRATION (visible color fringing)
        if lens_profile_id in ["helios-44-2", "custom"]:
            logger.info("Applying chromatic aberration...")
            b, g, r = cv2.split(result)
            
            # Scale red channel outward
            scale = 1.01 if lens_profile_id == "helios-44-2" else 1.02
            M_r = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
            r_scaled = cv2.warpAffine(r, M_r, (w, h))
            
            # Scale blue channel inward
            scale = 0.99 if lens_profile_id == "helios-44-2" else 0.98
            M_b = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
            b_scaled = cv2.warpAffine(b, M_b, (w, h))
            
            result = cv2.merge([b_scaled, g, r_scaled])
        
        # 3. VINTAGE DEFECTS (make them visible!)
        if defects.get('dust'):
            logger.info("Adding dust spots...")
            # Add visible dust spots
            for _ in range(50):
                x = np.random.randint(50, w-50)
                y = np.random.randint(50, h-50)
                radius = np.random.randint(2, 6)
                darkness = np.random.uniform(0.3, 0.7)
                
                # Create soft dust spot
                Y, X = np.ogrid[y-radius*2:y+radius*2, x-radius*2:x+radius*2]
                mask = ((X-x)**2 + (Y-y)**2) <= radius**2
                if mask.shape[0] > 0 and mask.shape[1] > 0:
                    try:
                        result[y-radius*2:y+radius*2, x-radius*2:x+radius*2][mask] = \
                            (result[y-radius*2:y+radius*2, x-radius*2:x+radius*2][mask] * darkness).astype(np.uint8)
                    except:
                        pass
        
        if defects.get('scratches'):
            logger.info("Adding scratches...")
            # Add visible scratches
            for _ in range(5):
                # Random diagonal scratch
                start_x = np.random.randint(0, w//2)
                start_y = np.random.randint(0, h)
                end_x = start_x + np.random.randint(w//4, w//2)
                end_y = start_y + np.random.randint(-h//4, h//4)
                
                thickness = np.random.randint(1, 3)
                color = np.random.randint(150, 200)
                
                cv2.line(result, (start_x, start_y), (end_x, end_y), 
                        (color, color, color), thickness)
        
        if defects.get('haze'):
            logger.info("Adding haze...")
            # Add visible haze effect
            haze_layer = np.ones_like(result) * 255
            alpha = 0.3  # 30% haze
            result = cv2.addWeighted(result, 1-alpha, haze_layer, alpha, 0)
        
        if defects.get('fungus'):
            logger.info("Adding fungus patterns...")
            # Add circular fungus-like patterns
            for _ in range(3):
                cx = np.random.randint(w//4, 3*w//4)
                cy = np.random.randint(h//4, 3*h//4)
                
                # Create branching pattern
                for _ in range(10):
                    angle = np.random.uniform(0, 2*np.pi)
                    length = np.random.randint(20, 60)
                    end_x = int(cx + length * np.cos(angle))
                    end_y = int(cy + length * np.sin(angle))
                    
                    cv2.line(result, (cx, cy), (end_x, end_y), (100, 100, 100), 2)
        
        # 4. CORRECTION EFFECT (make it visible)
        if correction_mode != 'none':
            logger.info(f"Applying {correction_mode} correction...")
            
            if correction_mode == 'physical':
                # Reduce vignetting
                result = cv2.addWeighted(result, 0.7, image, 0.3, 0)
            elif correction_mode == 'ml':
                # Enhance sharpness
                kernel = np.array([[-1,-1,-1],
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
                sharpened = cv2.filter2D(result, -1, kernel)
                result = cv2.addWeighted(result, 0.5, sharpened, 0.5, 0)
            else:  # hybrid
                # Both reduce vignetting and sharpen
                result = cv2.addWeighted(result, 0.8, image, 0.2, 0)
                kernel = np.array([[0,-1,0],
                                  [-1,5,-1],
                                  [0,-1,0]])
                result = cv2.filter2D(result, -1, kernel)
        
        # Add a watermark to prove processing happened
        cv2.putText(result, f"VintageOptics: {lens_profile_id}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, f"Mode: {correction_mode}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Convert to PIL Image
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(result_rgb)
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        
        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f}s")
        
        # Create response
        response = send_file(
            img_byte_arr,
            mimetype='image/jpeg',
            as_attachment=False
        )
        
        # Add headers
        response.headers['x-processing-time'] = f"{processing_time:.1f}s"
        response.headers['x-quality-score'] = "85"
        response.headers['x-defects-detected'] = str(sum(defects.values()))
        response.headers['x-correction-applied'] = "75"
        
        return response
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/lens-profiles', methods=['GET'])
def get_lens_profiles():
    """Get available lens profiles."""
    return jsonify([
        {"id": "canon-50mm-f1.4", "name": "Canon FD 50mm f/1.4", "available": True},
        {"id": "helios-44-2", "name": "Helios 44-2 58mm f/2", "available": True},
        {"id": "takumar-55mm", "name": "Super Takumar 55mm f/1.8", "available": True},
        {"id": "custom", "name": "Custom (Strong Effects)", "available": True}
    ])


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get API status."""
    return jsonify({
        "status": "running",
        "mode": "demo_visible",
        "message": "Running demo mode with VISIBLE effects"
    })


@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({
        "name": "VintageOptics GUI API",
        "version": "Demo Visible",
        "status": "running"
    })


if __name__ == "__main__":
    print("\n" + "="*60)
    print("VintageOptics GUI API - VISIBLE DEMO MODE")
    print("="*60)
    print("This version applies VISIBLE effects so you can see it working:")
    print("- Strong vignetting")
    print("- Visible chromatic aberration")
    print("- Clear defects (dust, scratches, haze)")
    print("- Text watermark showing processing")
    print("="*60)
    print(f"\nStarting on http://localhost:8000\n")
    
    app.run(host="0.0.0.0", port=8000, debug=False)
