# Orthogonal Error Correction in VintageOptics

## Overview

The Orthogonal Error Correction system leverages the fundamental insight that analog (vintage lens) and digital (sensor) error sources are statistically independent due to their different physical origins. This allows us to use a "logic puzzle" approach to mutual error rejection and signal enhancement.

## Key Innovation

Traditional approaches try to model and remove errors from a single source. Our approach recognizes that:

- **Analog errors** are physics-based, continuous, and stable:
  - Smooth spatial gradients (vignetting)
  - Wavelength-dependent patterns (chromatic aberration)
  - Radial symmetry (barrel/pincushion distortion)
  
- **Digital errors** are discrete, noisy, and variable:
  - Random noise patterns (shot noise, read noise)
  - Fixed pattern noise (hot pixels, column defects)
  - Quantization artifacts

Since these error types are orthogonal in origin, we can use Independent Component Analysis (ICA) and confidence-based fusion to separate and reject errors from both domains.

## Usage

### Basic Command Line

```bash
# Process a single image pair
python -m vintageoptics.orchestration.integration_pipeline \
    path/to/vintage.jpg \
    path/to/digital.jpg \
    -o corrected.jpg

# Batch processing
python -m vintageoptics.orchestration.integration_pipeline \
    --batch \
    vintage_dir/ \
    digital_dir/ \
    -o output_dir/
```

### Python API

```python
from vintageoptics.analysis import HybridErrorCorrector
import cv2

# Load images
vintage = cv2.imread('vintage_photo.jpg')
digital = cv2.imread('digital_reference.jpg')

# Initialize corrector
corrector = HybridErrorCorrector()

# Process
corrected, report = corrector.process(vintage, digital)

# Check confidence
print(f"Correction confidence: {report['correction_confidence']:.1%}")
```

### Pipeline Configuration

Create a YAML configuration file:

```yaml
stages:
  - name: orthogonal_correction
    params:
      confidence_threshold: 0.7
      ica_components: 4
```

Then use it:

```bash
python -m vintageoptics.orchestration.integration_pipeline \
    --config my_config.yaml \
    vintage.jpg digital.jpg
```

## Algorithm Details

### 1. Error Characterization
- Analyze frequency content, spatial correlation, and statistical moments
- Classify errors as analog or digital based on their signatures

### 2. Confidence Mapping
- Generate pixel-wise confidence maps for each source
- Analog confidence: inversely related to aberration strength
- Digital confidence: inversely related to local noise

### 3. Mutual Error Rejection
Decision logic per pixel:
- High vintage confidence → trust vintage
- High digital confidence → trust digital  
- Both confident → weighted blend
- Neither confident → ICA separation

### 4. Component Separation
When both sources have low confidence:
- Apply ICA to separate independent error components
- Classify components by frequency content
- Reconstruct signal with error components removed

## Performance

- **10x-100x** faster than generative AI approaches
- **Interpretable** - every decision can be visualized
- **Modular** - fits into the constraint-oriented architecture
- **GPU-optional** - uses CPU by default, GPU for acceleration

## Integration with VintageOptics

This module exemplifies the modular, constraint-oriented architecture:
- **Physics-grounded**: Based on real optical and sensor physics
- **Composable**: Can be combined with other VintageOptics modules
- **Transparent**: Provides detailed analysis reports
- **Efficient**: Avoids brute-force ML in favor of principled algorithms

## Future Enhancements

1. **Learned Confidence Models**: Train lens/sensor-specific confidence predictors
2. **Temporal Integration**: Use video sequences for better error separation  
3. **HDR Fusion**: Extend to multiple exposures
4. **Neural Enhancement**: Optional ML post-processing while preserving character

## References

The orthogonal error correction approach is inspired by:
- Independent Component Analysis (ICA) for blind source separation
- Multi-sensor fusion techniques from robotics
- Logic-based reasoning systems
- Physical optics and sensor modeling

---

Part of the VintageOptics constraint-oriented AI architecture.
