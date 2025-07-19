"""
Enhanced REST API with hyperdimensional computing features.

This module provides a FastAPI-based REST API for all VintageOptics
functionality including HD analysis and correction.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import cv2
import io
import base64
import tempfile
import shutil
from pathlib import Path
import logging
from datetime import datetime
import asyncio

from ..core.pipeline import VintageOpticsPipeline, PipelineConfig
from ..hyperdimensional import quick_hd_correction, analyze_lens_defects
from ..analysis import quick_lens_analysis
from ..synthesis import synthesize_lens_effect
from ..types.optics import ProcessingMode, LensProfile

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="VintageOptics API",
    description="Advanced lens correction and synthesis with HD computing",
    version="2.0.0"
)

# Global pipeline instance
pipeline = VintageOpticsPipeline()


# Pydantic models for API
class ProcessingRequest(BaseModel):
    """Image processing request."""
    mode: str = Field("auto", description="Processing mode: auto, correction, synthesis, hybrid")
    strength: float = Field(0.8, ge=0.0, le=1.0, description="Effect strength")
    use_hd: bool = Field(True, description="Use hyperdimensional computing")
    target_quality: float = Field(0.8, ge=0.0, le=1.0, description="Target quality score")
    lens_profile: Optional[str] = Field(None, description="Lens profile name for synthesis")
    
    class Config:
        schema_extra = {
            "example": {
                "mode": "hybrid",
                "strength": 0.8,
                "use_hd": True,
                "target_quality": 0.85,
                "lens_profile": "Canon FD 50mm f/1.4"
            }
        }


class HDAnalysisRequest(BaseModel):
    """HD analysis request."""
    detect_defects: bool = Field(True, description="Detect lens defects")
    separate_errors: bool = Field(True, description="Separate vintage/digital errors")
    compute_signature: bool = Field(True, description="Compute HD signature")
    

class LensDetectionRequest(BaseModel):
    """Lens detection request."""
    return_characteristics: bool = Field(True, description="Return detailed characteristics")
    match_database: bool = Field(True, description="Match against known lenses")


class BatchProcessingRequest(BaseModel):
    """Batch processing request."""
    processing_config: ProcessingRequest
    output_format: str = Field("jpg", description="Output format: jpg, png, tiff")
    quality: int = Field(95, ge=1, le=100, description="Output quality")


# Response models
class ProcessingResponse(BaseModel):
    """Processing response."""
    status: str
    processing_time: float
    mode_used: str
    quality_score: float
    iterations: int
    image_url: Optional[str] = None
    download_url: Optional[str] = None
    

class HDAnalysisResponse(BaseModel):
    """HD analysis response."""
    dust_spots: int
    scratches: int
    fungus_areas: int
    total_defects: int
    hd_signature: Optional[List[float]] = None
    vintage_confidence: float
    digital_confidence: float
    topological_features: int
    

class LensCharacteristicsResponse(BaseModel):
    """Lens characteristics response."""
    lens_type: str
    confidence: float
    manufacturer: Optional[str]
    model: Optional[str]
    era: Optional[str]
    focal_length: float
    aperture_range: str
    coating_type: Optional[str]
    quality_metrics: Dict[str, float]


# Temporary storage for processed images
TEMP_DIR = Path(tempfile.gettempdir()) / "vintageoptics_api"
TEMP_DIR.mkdir(exist_ok=True)


# Utility functions
def decode_image(image_data: str) -> np.ndarray:
    """Decode base64 image to numpy array."""
    try:
        # Remove data URL prefix if present
        if "," in image_data:
            image_data = image_data.split(",")[1]
            
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image")
            
        return image
    except Exception as e:
        logger.error(f"Image decode error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


def encode_image(image: np.ndarray, format: str = "jpg", quality: int = 95) -> str:
    """Encode numpy array to base64 string."""
    try:
        # Encode image
        if format.lower() == "png":
            encode_param = [cv2.IMWRITE_PNG_COMPRESSION, 9]
        else:  # jpg
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
            
        _, buffer = cv2.imencode(f".{format}", image, encode_param)
        
        # Convert to base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/{format};base64,{image_base64}"
    except Exception as e:
        logger.error(f"Image encode error: {e}")
        raise HTTPException(status_code=500, detail=f"Could not encode image: {str(e)}")


async def save_temp_image(image: np.ndarray, prefix: str = "processed") -> Path:
    """Save image to temporary location."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = TEMP_DIR / filename
    
    cv2.imwrite(str(filepath), image)
    
    return filepath


async def cleanup_temp_file(filepath: Path, delay: int = 3600):
    """Clean up temporary file after delay."""
    await asyncio.sleep(delay)
    try:
        if filepath.exists():
            filepath.unlink()
            logger.info(f"Cleaned up temp file: {filepath}")
    except Exception as e:
        logger.error(f"Failed to clean up {filepath}: {e}")


# API Endpoints

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "VintageOptics API",
        "version": "2.0.0",
        "features": [
            "Lens correction",
            "Lens synthesis", 
            "Hyperdimensional analysis",
            "Defect detection",
            "Quality enhancement",
            "Batch processing"
        ]
    }


@app.post("/process", response_model=ProcessingResponse)
async def process_image(
    request: ProcessingRequest,
    image: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Process a single image."""
    try:
        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Configure pipeline
        config = PipelineConfig(
            mode=ProcessingMode(request.mode.upper()),
            use_hd=request.use_hd,
            correction_strength=request.strength,
            target_quality=request.target_quality,
            generate_report=True
        )
        
        # Create pipeline instance
        proc_pipeline = VintageOpticsPipeline(config)
        
        # Process image
        result = proc_pipeline.process(img)
        
        # Save processed image
        output_path = await save_temp_image(result.corrected_image)
        
        # Schedule cleanup
        if background_tasks:
            background_tasks.add_task(cleanup_temp_file, output_path, delay=3600)
        
        return ProcessingResponse(
            status="success",
            processing_time=result.processing_time,
            mode_used=result.mode_used.value,
            quality_score=result.quality_metrics.overall_quality if result.quality_metrics else 0.0,
            iterations=result.iterations_used,
            download_url=f"/download/{output_path.name}"
        )
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/base64", response_model=ProcessingResponse)
async def process_image_base64(
    image_data: str,
    request: ProcessingRequest
):
    """Process a base64-encoded image."""
    try:
        # Decode image
        img = decode_image(image_data)
        
        # Process using pipeline
        config = PipelineConfig(
            mode=ProcessingMode(request.mode.upper()),
            use_hd=request.use_hd,
            correction_strength=request.strength,
            target_quality=request.target_quality
        )
        
        proc_pipeline = VintageOpticsPipeline(config)
        result = proc_pipeline.process(img)
        
        # Encode result
        processed_b64 = encode_image(result.corrected_image)
        
        return ProcessingResponse(
            status="success",
            processing_time=result.processing_time,
            mode_used=result.mode_used.value,
            quality_score=result.quality_metrics.overall_quality if result.quality_metrics else 0.0,
            iterations=result.iterations_used,
            image_url=processed_b64
        )
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/hd", response_model=HDAnalysisResponse)
async def analyze_hd(
    image: UploadFile = File(...),
    request: HDAnalysisRequest = HDAnalysisRequest()
):
    """Perform HD analysis on an image."""
    try:
        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Perform HD analysis
        defects = analyze_lens_defects(img)
        
        response = HDAnalysisResponse(
            dust_spots=defects['dust_spots'],
            scratches=defects['scratches'],
            fungus_areas=defects['regions'],
            total_defects=defects['total_defects'],
            topological_features=len(defects['details']['persistence_diagram']),
            vintage_confidence=0.0,
            digital_confidence=0.0
        )
        
        # Add HD signature if requested
        if request.compute_signature and 'hypervector' in defects:
            response.hd_signature = defects['hypervector'].tolist()[:100]  # First 100 dims
        
        # Add error separation if requested
        if request.separate_errors:
            from ..hyperdimensional import separate_vintage_digital_errors
            separation = separate_vintage_digital_errors(img)
            response.vintage_confidence = separation['vintage_confidence']
            response.digital_confidence = separation['digital_confidence']
        
        return response
        
    except Exception as e:
        logger.error(f"HD analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/lens", response_model=LensCharacteristicsResponse)
async def detect_lens(
    image: UploadFile = File(...),
    request: LensDetectionRequest = LensDetectionRequest()
):
    """Detect lens characteristics."""
    try:
        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Detect lens
        from ..detection import UnifiedLensDetector
        detector = UnifiedLensDetector()
        detection = detector.detect(img)
        
        # Get characteristics if requested
        quality_metrics = {}
        if request.return_characteristics:
            characteristics = quick_lens_analysis(img)
            quality_metrics = {
                "sharpness": characteristics.sharpness_score,
                "contrast": characteristics.contrast_score,
                "color_accuracy": characteristics.color_accuracy,
                "overall_quality": characteristics.overall_quality
            }
        
        return LensCharacteristicsResponse(
            lens_type=detection.lens_type.value,
            confidence=detection.confidence,
            manufacturer=detection.manufacturer,
            model=detection.model,
            era=detection.era,
            focal_length=50.0,  # Default
            aperture_range="f/1.4-f/16",  # Default
            coating_type=detection.coating_type,
            quality_metrics=quality_metrics
        )
        
    except Exception as e:
        logger.error(f"Lens detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize/{lens_name}")
async def synthesize_lens(
    lens_name: str,
    image: UploadFile = File(...),
    strength: float = 0.8
):
    """Apply lens synthesis effect."""
    try:
        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Apply synthesis
        result = synthesize_lens_effect(img, lens_name, strength)
        
        # Save and return
        output_path = await save_temp_image(result, prefix=f"synthesized_{lens_name}")
        
        return {
            "status": "success",
            "lens_applied": lens_name,
            "strength": strength,
            "download_url": f"/download/{output_path.name}"
        }
        
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quick/hd-correction")
async def quick_hd_correct(
    image: UploadFile = File(...),
    strength: float = 0.8
):
    """Quick HD correction."""
    try:
        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Apply HD correction
        corrected = quick_hd_correction(img, strength)
        
        # Encode result
        result_b64 = encode_image(corrected)
        
        return {
            "status": "success",
            "image_url": result_b64,
            "method": "hyperdimensional"
        }
        
    except Exception as e:
        logger.error(f"HD correction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch/process")
async def batch_process(
    files: List[UploadFile] = File(...),
    request: BatchProcessingRequest = BatchProcessingRequest(
        processing_config=ProcessingRequest()
    ),
    background_tasks: BackgroundTasks = None
):
    """Process multiple images."""
    try:
        results = []
        
        # Configure pipeline
        config = PipelineConfig(
            mode=ProcessingMode(request.processing_config.mode.upper()),
            use_hd=request.processing_config.use_hd,
            correction_strength=request.processing_config.strength,
            target_quality=request.processing_config.target_quality
        )
        
        proc_pipeline = VintageOpticsPipeline(config)
        
        # Process each image
        for i, file in enumerate(files):
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": "Could not decode image"
                })
                continue
            
            try:
                # Process
                result = proc_pipeline.process(img)
                
                # Save
                output_path = await save_temp_image(
                    result.corrected_image,
                    prefix=f"batch_{i:04d}"
                )
                
                # Schedule cleanup
                if background_tasks:
                    background_tasks.add_task(cleanup_temp_file, output_path, delay=3600)
                
                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "download_url": f"/download/{output_path.name}",
                    "quality_score": result.quality_metrics.overall_quality 
                                   if result.quality_metrics else 0.0
                })
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "status": "completed",
            "total_files": len(files),
            "successful": sum(1 for r in results if r["status"] == "success"),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download processed file."""
    filepath = TEMP_DIR / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=filepath,
        media_type="image/jpeg",
        filename=filename
    )


@app.get("/profiles")
async def list_lens_profiles():
    """List available lens profiles."""
    from ..synthesis import CharacteristicLibrary
    
    library = CharacteristicLibrary()
    profiles = library.list_profiles()
    
    return {
        "profiles": profiles,
        "count": len(profiles)
    }


@app.post("/profiles/create")
async def create_custom_profile(
    name: str,
    reference_images: List[UploadFile] = File(...),
):
    """Create custom lens profile from reference images."""
    try:
        images = []
        
        # Read all images
        for file in reference_images:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                images.append(img)
        
        if not images:
            raise HTTPException(status_code=400, detail="No valid images provided")
        
        # Create profile
        from ..synthesis import LensSynthesizer
        synthesizer = LensSynthesizer()
        profile = synthesizer.create_custom_profile(name, images)
        
        # Save to library
        from ..synthesis import CharacteristicLibrary
        library = CharacteristicLibrary()
        library.add_profile(profile)
        
        return {
            "status": "success",
            "profile_name": name,
            "characteristics": {
                "vignetting": profile.vignetting_amount,
                "distortion": profile.distortion_amount,
                "chromatic_aberration": profile.chromatic_aberration,
                "bokeh_quality": profile.bokeh_quality
            }
        }
        
    except Exception as e:
        logger.error(f"Profile creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("VintageOptics API starting up...")
    
    # Ensure temp directory exists
    TEMP_DIR.mkdir(exist_ok=True)
    
    # Clean old temp files
    for file in TEMP_DIR.glob("*.jpg"):
        try:
            if file.stat().st_mtime < datetime.now().timestamp() - 86400:  # 24 hours
                file.unlink()
        except:
            pass
    
    logger.info("API ready")


@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("VintageOptics API shutting down...")
    
    # Clean temp files
    try:
        shutil.rmtree(TEMP_DIR)
    except:
        pass


# Run with: uvicorn vintageoptics.api.rest_api:app --reload
