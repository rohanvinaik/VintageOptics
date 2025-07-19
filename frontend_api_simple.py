"""
Simplified API for VintageOptics GUI - Standalone version
This version works without the full VintageOptics package installed.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import numpy as np
import cv2
import io
import time
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VintageOptics GUI API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-processing-time", "x-quality-score", "x-defects-detected", "x-correction-applied"]
)

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

@app.post("/api/process")
async def process_image(
    file: UploadFile = File(...),
    lens_profile: str = Query("canon-50mm-f1.4"),
    correction_mode: str = Query("hybrid"),
    defect_dust: Optional[str] = Query(None),
    defect_fungus: Optional[str] = Query(None),
    defect_scratches: Optional[str] = Query(None),
    defect_haze: Optional[str] = Query(None),
    defect_separation: Optional[str] = Query(None),
    defect_coating: Optional[str] = Query(None)
):
    """Process an image with vintage lens effects and corrections."""
    
    start_time = time.time()
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Get lens configuration
        lens_config = LENS_PROFILES.get(lens_profile, LENS_PROFILES["canon-50mm-f1.4"])
        
        # Collect defects
        defects = {
            "dust": defect_dust == "true",
            "fungus": defect_fungus == "true",
            "scratches": defect_scratches == "true",
            "haze": defect_haze == "true",
            "separation": defect_separation == "true",
            "coating": defect_coating == "true"
        }
        
        # Apply vintage effects
        if any(defects.values()):
            image = apply_vintage_defects(image, defects)
        
        # Apply lens characteristics
        image = apply_lens_characteristics(image, lens_config["config"])
        
        # Apply correction based on mode
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
        
        # Return image with metadata headers
        return StreamingResponse(
            img_byte_arr,
            media_type="image/jpeg",
            headers={
                "x-processing-time": f"{processing_time:.1f}s",
                "x-quality-score": str(stats["quality_score"]),
                "x-defects-detected": str(stats["defects_detected"]),
                "x-correction-applied": str(stats["correction_applied"])
            }
        )
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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


@app.get("/api/lens-profiles")
async def get_lens_profiles():
    """Get available lens profiles."""
    profiles = []
    for key, value in LENS_PROFILES.items():
        profiles.append({
            "id": key,
            "name": value["name"],
            "config": value["config"]
        })
    return profiles


@app.get("/api/synthesis-presets")
async def get_synthesis_presets():
    """Get synthesis presets."""
    return [
        {"id": "dreamy", "name": "Dreamy Portrait", "description": "Soft, ethereal look"},
        {"id": "vintage", "name": "Vintage Film", "description": "Classic film look"},
        {"id": "artistic", "name": "Artistic Bokeh", "description": "Creative bokeh patterns"}
    ]


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "VintageOptics GUI API",
        "version": "1.0.0",
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting VintageOptics GUI API on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
