# VintageOptics Enhanced GUI - Equipment Context Support

## Overview

The enhanced VintageOptics GUI now includes comprehensive equipment context support, allowing for more accurate lens correction and synthesis based on actual camera and lens metadata.

## New Features

### 1. **Automatic Metadata Extraction**
- Automatically extracts EXIF, XMP, and IPTC metadata from uploaded images
- Detects camera make, model, and serial number
- Identifies lens model, focal length, aperture, and other optical parameters
- Displays extracted metadata in the GUI for user verification

### 2. **Equipment Context Input**
- **Camera Model Field**: Input or auto-fill camera model from metadata
- **Lens Model Field**: Input or auto-fill lens model from metadata
- Both fields serve as initial conditions for the processing pipeline

### 3. **Manual Parameter Override**
- Toggle switch to enable manual parameter input
- Override specific optical parameters:
  - Focal Length (mm)
  - Aperture (f-stop)
  - Distortion coefficients (K1, K2)
- Useful for fine-tuning corrections or working with unknown lenses

### 4. **Custom Lens Profile Generation**
- Automatically creates custom lens profiles based on:
  - Extracted metadata
  - User-provided equipment context
  - Manual parameter overrides
- Profiles are tailored to specific lens/camera combinations

## How It Works

### Processing Flow

1. **Image Upload** → Metadata extraction via ExifTool
2. **Auto-Detection** → Match against known lens profiles
3. **Context Application** → Use equipment info to create/select profile
4. **Processing** → Apply corrections with context-aware parameters
5. **Result** → Enhanced image with metadata preserved

### Equipment Context as Initial Conditions

The equipment context serves as initial conditions for the error correction system:

- **Camera Model** → Influences sensor characteristics and processing
- **Lens Model** → Determines optical characteristics and defects
- **Manual Parameters** → Override automatic detection when needed

This aligns with the orthogonal error correction approach:
- Vintage lens errors (physics-based, continuous)
- Digital sensor errors (binary, noise-based)
- Equipment context helps separate and correct both types

## Installation Requirements

### Required
- Python 3.7+
- Flask and Flask-CORS
- NumPy, OpenCV, Pillow
- VintageOptics core modules

### Optional but Recommended
- **ExifTool** for comprehensive metadata extraction
  - macOS: `brew install exiftool`
  - Linux: `sudo apt-get install exiftool`
  - Windows: Download from https://exiftool.org

## Running the Enhanced GUI

```bash
# Make the script executable (first time only)
chmod +x run_enhanced_gui.sh

# Run the enhanced GUI
./run_enhanced_gui.sh
```

Or manually:

```bash
# Start the enhanced backend
python frontend_api_enhanced.py

# Open frontend/index_enhanced.html in your browser
```

## API Endpoints

### Enhanced Processing Endpoint
```
POST /api/process/enhanced
```

Query Parameters:
- `lens_profile`: Profile ID or 'auto'/'custom'
- `correction_mode`: 'hybrid'/'correction'/'synthesis'/'none'
- `camera_model`: Camera model string
- `lens_model`: Lens model string
- `focal_length`: Manual focal length override
- `aperture`: Manual aperture override
- `distortion_k1`: Manual K1 coefficient
- `distortion_k2`: Manual K2 coefficient

### Metadata Extraction Endpoint
```
POST /api/extract-metadata
```

Returns:
```json
{
  "camera": {
    "make": "Canon",
    "model": "EOS 5D Mark IV",
    "serial": "123456"
  },
  "lens": {
    "make": "Canon",
    "model": "EF 50mm f/1.4 USM",
    "focal_length": "50",
    "aperture": "1.4",
    "serial": "789012"
  },
  "settings": {
    "iso": "400",
    "shutter_speed": "1/125",
    "aperture": "f/2.8",
    "focal_length": "50mm"
  }
}
```

## Usage Examples

### Basic Usage with Auto-Detection
1. Upload an image with EXIF data
2. Metadata is automatically extracted and displayed
3. Camera and lens fields are auto-filled
4. Select correction mode and process

### Manual Override for Unknown Lens
1. Upload image
2. Enter camera and lens models manually
3. Toggle "Manual Parameter Override"
4. Enter known distortion coefficients
5. Process with custom parameters

### Creating Custom Profiles
1. Set lens profile to "Custom"
2. Fill in equipment context
3. Enable manual overrides if needed
4. The system creates a tailored profile
5. Process with the custom profile

## Technical Details

### Metadata Preservation
- Processed images retain original metadata
- Additional XMP tags are added:
  - `XMP:ProcessedWith`: "VintageOptics"
  - `XMP:LensProfileUsed`: Profile name used
  - `XMP:CorrectionMode`: Mode applied
  - `XMP:OriginalCamera`: Source camera
  - `XMP:OriginalLens`: Source lens
  - `XMP:DistortionK1Applied`: If manually set
  - `XMP:DistortionK2Applied`: If manually set

### Profile Matching Algorithm
1. Extract lens model from metadata
2. Normalize and tokenize lens name
3. Match against known profiles database
4. If no match, create custom profile using:
   - Focal length parsing
   - Aperture detection
   - Manufacturer identification
   - Era estimation

### Error Correction Strategy
The equipment context enables the orthogonal error correction approach:

```
Digital Error = Sensor Noise + Quantization + Compression
Vintage Error = Optical Distortion + Physical Defects + Age Effects

Total Error = Digital Error ⊕ Vintage Error
Correction = Context-Aware(Separate(Digital, Vintage))
```

## Troubleshooting

### ExifTool Not Found
- Install ExifTool for full metadata extraction
- Without it, basic metadata extraction still works but is limited

### Metadata Not Detected
- Ensure image has EXIF data (not stripped)
- Try manual entry of equipment info
- Use manual parameter override

### Custom Profile Issues
- Verify lens model format (e.g., "Canon EF 50mm f/1.4")
- Check focal length and aperture values
- Use manual overrides for fine-tuning

## Future Enhancements

1. **Profile Database Expansion**
   - Community-contributed lens profiles
   - Auto-learning from processed images
   - Cloud profile sharing

2. **Advanced Context Usage**
   - Sensor-specific corrections
   - Era-based processing styles
   - Mount adapter compensation

3. **Metadata Enhancement**
   - GPS location for atmospheric effects
   - Time/date for seasonal adjustments
   - Multi-image profile learning

## Contributing

To add new lens profiles or improve equipment detection:

1. Add profiles to `LENS_PROFILES` in `frontend_api_enhanced.py`
2. Include full optical parameters
3. Test with sample images
4. Submit pull request with examples

## Support

For issues or questions:
- Check the main VintageOptics documentation
- Review the API logs for errors
- Ensure all dependencies are installed
- Verify ExifTool is accessible

The enhanced GUI brings VintageOptics closer to the vision of constraint-oriented AI, where equipment context serves as physical constraints that guide intelligent image processing.
