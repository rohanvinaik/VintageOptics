"""
VintageOptics GUI API - Refined Demo Version
This version applies more realistic vintage lens effects.
"""

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import io
import time
from PIL import Image, ImageFilter
import logging
from scipy.ndimage import gaussian_filter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, expose_headers=["x-processing-time", "x-quality-score", "x-defects-detected", "x-correction-applied"])

class VintageEffects:
    """More realistic vintage lens effects."""
    
    @staticmethod
    def apply_vignetting(image, amount=0.3, falloff=2.2):
        """Apply smooth vignetting effect."""
        h, w = image.shape[:2]
        
        # Create radial gradient
        Y, X = np.ogrid[:h, :w]
        center_x, center_y = w/2, h/2
        
        # Distance from center normalized
        dist = np.sqrt(((X - center_x)/w)**2 + ((Y - center_y)/h)**2)
        
        # Smooth vignetting function
        vignette = 1 - (dist ** falloff) * amount
        vignette = gaussian_filter(vignette, sigma=50)  # Smooth the vignetting
        vignette = np.clip(vignette, 0, 1)
        
        # Apply to each channel
        result = image.copy().astype(np.float32)
        for i in range(3):
            result[:, :, i] *= vignette
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def apply_chromatic_aberration(image, shift=2):
        """Apply subtle chromatic aberration."""
        h, w = image.shape[:2]
        
        # Split channels
        b, g, r = cv2.split(image)
        
        # Create slight radial shift
        # Red channel expands
        M_r = np.float32([[1, 0, -shift/2], [0, 1, -shift/2]])
        M_r[0, 0] = M_r[1, 1] = 1 + (shift / 1000)
        r_shifted = cv2.warpAffine(r, M_r, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Blue channel contracts
        M_b = np.float32([[1, 0, shift/2], [0, 1, shift/2]])
        M_b[0, 0] = M_b[1, 1] = 1 - (shift / 1000)
        b_shifted = cv2.warpAffine(b, M_b, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return cv2.merge([b_shifted, g, r_shifted])
    
    @staticmethod
    def apply_lens_blur(image, amount=0.5):
        """Apply subtle lens blur/softness."""
        # Convert to PIL for better blur
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Apply lens blur
        blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=amount))
        
        # Convert back
        blurred_cv = cv2.cvtColor(np.array(blurred), cv2.COLOR_RGB2BGR)
        
        # Blend with original (preserve some sharpness)
        return cv2.addWeighted(image, 0.3, blurred_cv, 0.7, 0)
    
    @staticmethod
    def add_dust_particles(image, count=30, size_range=(1, 3)):
        """Add realistic dust particles."""
        result = image.copy()
        h, w = image.shape[:2]
        
        for _ in range(count):
            x = np.random.randint(10, w-10)
            y = np.random.randint(10, h-10)
            size = np.random.randint(*size_range)
            
            # Create soft dust particle
            opacity = np.random.uniform(0.3, 0.6)
            
            # Draw with anti-aliasing
            cv2.circle(result, (x, y), size, (0, 0, 0), -1, cv2.LINE_AA)
            
            # Blend
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(mask, (x, y), size + 1, 1, -1, cv2.LINE_AA)
            mask = gaussian_filter(mask, sigma=1)
            
            for i in range(3):
                result[:, :, i] = result[:, :, i] * (1 - mask * opacity)
        
        return result.astype(np.uint8)
    
    @staticmethod
    def add_subtle_scratches(image, count=3):
        """Add subtle scratches."""
        result = image.copy()
        h, w = image.shape[:2]
        
        for _ in range(count):
            # Random scratch path
            points = []
            start_x = np.random.randint(0, w)
            start_y = np.random.randint(0, h)
            
            # Create curved scratch
            num_points = np.random.randint(5, 10)
            for i in range(num_points):
                x = start_x + i * (w // num_points) + np.random.randint(-20, 20)
                y = start_y + np.random.randint(-50, 50)
                points.append([x, y])
            
            points = np.array(points, np.int32)
            
            # Draw scratch
            color = np.random.randint(180, 220)
            thickness = 1
            cv2.polylines(result, [points], False, (color, color, color), thickness, cv2.LINE_AA)
        
        return result
    
    @staticmethod
    def add_haze(image, amount=0.15):
        """Add subtle haze effect."""
        # Create haze layer
        haze = np.ones_like(image, dtype=np.float32) * 255
        
        # Add some variation to the haze
        noise = np.random.normal(0, 10, image.shape[:2])
        for i in range(3):
            haze[:, :, i] += noise
        
        haze = np.clip(haze, 0, 255)
        
        # Blend
        return cv2.addWeighted(image, 1 - amount, haze.astype(np.uint8), amount, 0)


# Lens characteristics
LENS_CHARACTERISTICS = {
    "canon-50mm-f1.4": {
        "vignetting": 0.25,
        "vignetting_falloff": 2.2,
        "chromatic_aberration": 1,
        "blur": 0.3,
        "character": "neutral"
    },
    "helios-44-2": {
        "vignetting": 0.35,
        "vignetting_falloff": 1.8,
        "chromatic_aberration": 3,
        "blur": 0.5,
        "character": "swirly"
    },
    "takumar-55mm": {
        "vignetting": 0.2,
        "vignetting_falloff": 2.5,
        "chromatic_aberration": 1,
        "blur": 0.2,
        "character": "sharp"
    },
    "nikkor-105mm": {
        "vignetting": 0.15,
        "vignetting_falloff": 3.0,
        "chromatic_aberration": 0.5,
        "blur": 0.2,
        "character": "clinical"
    },
    "zeiss-planar": {
        "vignetting": 0.1,
        "vignetting_falloff": 2.8,
        "chromatic_aberration": 0.3,
        "blur": 0.1,
        "character": "modern"
    },
    "custom": {
        "vignetting": 0.4,
        "vignetting_falloff": 1.5,
        "chromatic_aberration": 4,
        "blur": 0.8,
        "character": "artistic"
    }
}


@app.route('/api/process', methods=['POST'])
def process_image():
    """Process an image with refined vintage lens effects."""
    
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
        
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image file"}), 400
        
        # Get lens characteristics
        lens_chars = LENS_CHARACTERISTICS.get(lens_profile_id, LENS_CHARACTERISTICS["canon-50mm-f1.4"])
        
        # Apply lens characteristics
        result = image.copy()
        
        # 1. Lens blur (softness)
        if lens_chars["blur"] > 0:
            result = VintageEffects.apply_lens_blur(result, lens_chars["blur"])
        
        # 2. Chromatic aberration
        if lens_chars["chromatic_aberration"] > 0:
            result = VintageEffects.apply_chromatic_aberration(result, lens_chars["chromatic_aberration"])
        
        # 3. Vignetting
        result = VintageEffects.apply_vignetting(
            result, 
            lens_chars["vignetting"], 
            lens_chars["vignetting_falloff"]
        )
        
        # 4. Apply defects if selected
        if defects.get('dust'):
            result = VintageEffects.add_dust_particles(result, count=40)
        
        if defects.get('scratches'):
            result = VintageEffects.add_subtle_scratches(result, count=2)
        
        if defects.get('haze'):
            result = VintageEffects.add_haze(result, amount=0.15)
        
        # 5. Apply correction if not "none"
        if correction_mode != 'none':
            if correction_mode == 'physical':
                # Reduce vignetting and chromatic aberration
                corrected = cv2.addWeighted(result, 0.6, image, 0.4, 0)
                result = corrected
            elif correction_mode == 'ml':
                # Enhance sharpness and contrast
                kernel = np.array([[0, -0.5, 0],
                                  [-0.5, 3, -0.5],
                                  [0, -0.5, 0]]) / 1.5
                sharpened = cv2.filter2D(result, -1, kernel)
                result = cv2.addWeighted(result, 0.5, sharpened, 0.5, 0)
            else:  # hybrid
                # Combination of both
                corrected = cv2.addWeighted(result, 0.7, image, 0.3, 0)
                kernel = np.array([[0, -0.25, 0],
                                  [-0.25, 2, -0.25],
                                  [0, -0.25, 0]]) / 1.25
                result = cv2.filter2D(corrected, -1, kernel)
        
        # Convert to PIL Image
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(result_rgb)
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        
        processing_time = time.time() - start_time
        
        # Calculate pseudo-metrics
        defect_count = sum(defects.values())
        quality_score = 85 - (defect_count * 5)
        if correction_mode != 'none':
            quality_score += 10
        
        # Create response
        response = send_file(
            img_byte_arr,
            mimetype='image/jpeg',
            as_attachment=False
        )
        
        # Add headers
        response.headers['x-processing-time'] = f"{processing_time:.1f}s"
        response.headers['x-quality-score'] = str(quality_score)
        response.headers['x-defects-detected'] = str(defect_count)
        response.headers['x-correction-applied'] = "0" if correction_mode == 'none' else "75"
        
        return response
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/lens-profiles', methods=['GET'])
def get_lens_profiles():
    """Get available lens profiles."""
    profiles = []
    for lens_id, chars in LENS_CHARACTERISTICS.items():
        name_map = {
            "canon-50mm-f1.4": "Canon FD 50mm f/1.4",
            "helios-44-2": "Helios 44-2 58mm f/2",
            "takumar-55mm": "Super Takumar 55mm f/1.8",
            "nikkor-105mm": "Nikkor 105mm f/2.5",
            "zeiss-planar": "Zeiss Planar 50mm f/1.4",
            "custom": "Custom Profile"
        }
        profiles.append({
            "id": lens_id,
            "name": name_map.get(lens_id, lens_id),
            "available": True
        })
    return jsonify(profiles)


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get API status."""
    return jsonify({
        "status": "running",
        "mode": "refined_demo",
        "message": "Running refined demo with realistic effects"
    })


@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({
        "name": "VintageOptics GUI API",
        "version": "Refined Demo",
        "status": "running"
    })


if __name__ == "__main__":
    print("\n" + "="*60)
    print("VintageOptics GUI API - REFINED DEMO MODE")
    print("="*60)
    print("This version applies more realistic vintage lens effects:")
    print("- Subtle vignetting")
    print("- Realistic chromatic aberration")
    print("- Authentic vintage defects")
    print("- Proper correction modes")
    print("="*60)
    print(f"\nStarting on http://localhost:8000\n")
    
    app.run(host="0.0.0.0", port=8000, debug=False)
