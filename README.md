# VintageOptics

Advanced lens correction system with character preservation for vintage and modern lenses.

## Overview

VintageOptics is an open-source lens correction system that goes beyond simple geometric corrections. It's designed to preserve the unique character of vintage lenses while removing distracting defects, implementing the philosophy of "corrected but not clinical."

## Key Features

### 🔍 Comprehensive Lens Detection
- **Electronic Lens Detection**: Automatic identification from EXIF metadata
- **Optical Fingerprinting**: Unique signatures for individual lens copies
- **Manual Lens Detection**: Pattern matching in metadata for adapted vintage lenses
- **Lensfun Integration**: Access to 1000+ lens profiles from the open-source database

### 🧹 Intelligent Defect Cleanup
- **Defect Detection**: Identifies dust, scratches, fungus, coating wear, haze, and oil spots
- **Character Preservation**: Distinguishes between defects and character-adding imperfections
- **Adaptive Cleanup**: Different strategies for different types of defects

### 🎨 Character Preservation System
- **Bokeh Analysis**: Preserves swirly bokeh, smooth rendering, and unique highlight shapes
- **Rendering Style**: Maintains lens-specific contrast curves and color response
- **Selective Preservation**: Adapts preservation based on image content (portraits, landscapes, etc.)
- **Multiple Modes**: Full, Selective, Minimal, Adaptive, and Artistic preservation

### ⚙️ Advanced Correction Engine
- **Multi-Model Support**: Brown-Conrady, Division Model, Rational Function distortion models
- **Manufacturing Variations**: Handles lens-to-lens variations within the same model
- **Depth-Aware Processing**: Corrections that adapt based on depth information
- **Physics-Based**: Deterministic corrections based on optical principles

### 🧠 Vintage ML Integration
- **AI-Winter First**: Classic ML algorithms (1950s-1980s) as primary detection layer
- **Transparent Detection**: Perceptron, ADALINE, k-NN, SOM for interpretable results
- **Hybrid Physics-ML**: Iterative refinement between physics and vintage ML
- **PAC Learning**: Entropy-based adaptation and rule extraction
- **Lightweight**: CPU-only processing, no GPU required

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenCV
- NumPy, SciPy, scikit-learn, scikit-image

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/rohanvinaik/VintageOptics.git
cd VintageOptics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements/base.txt
```

### Optional Dependencies

For full functionality, install optional dependencies:

```bash
# For GPU acceleration
pip install -r requirements/gpu.txt

# For machine learning features
pip install -r requirements/ml.txt

# For development
pip install -r requirements/dev.txt
```

### External Tools (Optional but Recommended)

1. **ExifTool**: For advanced metadata extraction
   - Download from https://exiftool.org
   - Add to system PATH

2. **Lensfun**: For the lens database
   - The system will auto-download the database on first use
   - Or install system-wide: `apt-get install liblensfun-dev` (Linux) or `brew install lensfun` (macOS)

## Quick Start

```python
from vintageoptics import VintageOpticsPipeline
from vintageoptics.core import ProcessingRequest, ProcessingMode

# Initialize pipeline
pipeline = VintageOpticsPipeline('config/default.yaml')

# Process an image with standard correction
request = ProcessingRequest(
    image_path='path/to/image.jpg',
    mode=ProcessingMode.CORRECT,
    output_path='path/to/corrected.jpg',
    preserve_metadata=True
)

result = pipeline.process(request)
print(f"Lens detected: {result.lens_profile}")
print(f"Corrections applied: {result.quality_metrics}")

# Or use the hybrid physics-ML approach
request.mode = ProcessingMode.HYBRID
result = pipeline.process(request)
print(f"Converged in {result.iterations} iterations")
print(f"ML confidence: {result.ml_confidence:.2f}")
```

### Vintage ML Detection

Use classic machine learning for transparent defect detection:

```python
from vintageoptics.vintageml import VintageMLDefectDetector

detector = VintageMLDefectDetector(config)
results = detector.detect_defects(image)

for result in results:
    print(f"Detected {result.defect_type} using {result.method_used}")
    print(f"Confidence: {result.confidence:.2f}")
```

Train on your own data:

```python
from vintageoptics.vintageml import VintageMLTrainer

trainer = VintageMLTrainer(config)
trainer.train(
    data_dir='path/to/training/data',
    save_path='models/my_vintage_ml.pkl'
)
```

## Advanced Usage

### Creating Lens Instance Profiles

```python
from vintageoptics.detection import UnifiedLensDetector

detector = UnifiedLensDetector(config)

# Detect lens
detection_result = detector.detect_comprehensive({'path': 'image.jpg'})

# Create instance profile with calibration images
calibration_images = [...]  # Load multiple images from same lens
fingerprint = detector.create_instance_profile(
    detection_result,
    calibration_images
)
```

### Character Preservation Modes

```python
# Adaptive mode (default) - adapts based on content
request.settings = {'preservation_mode': 'adaptive'}

# Full preservation - maximum character retention
request.settings = {'preservation_mode': 'full'}

# Artistic mode - enhances desirable characteristics
request.settings = {'preservation_mode': 'artistic'}
```

### Batch Processing

```python
results = pipeline.batch_process(
    input_dir='path/to/images',
    output_dir='path/to/output',
    mode=ProcessingMode.CORRECT,
    preserve_character=True
)
```

## Configuration

The system is highly configurable through YAML files:

- `config/default.yaml`: Standard processing configuration
- `config/synthesis.yaml`: Lens synthesis settings
- `config/depth_aware.yaml`: Depth-based processing

Example configuration:

```yaml
character_preservation:
  mode: adaptive
  strength: 0.7
  
cleanup:
  dust_sensitivity: 0.8
  preserve_character: true
  
physics:
  model_selection: adaptive
  preserve_bokeh: true
```

## Project Structure

```
VintageOptics/
├── src/vintageoptics/
│   ├── detection/          # Lens detection modules
│   ├── physics/            # Optical correction engine
│   ├── statistical/        # Defect detection and cleanup
│   ├── synthesis/          # Character synthesis and preservation
│   ├── vintageml/          # Vintage ML algorithms (1950s-1980s)
│   ├── calibration/        # Lens calibration methods
│   ├── core/               # Pipeline and hybrid processing
│   └── integrations/       # External library integrations
├── config/                 # Configuration files
├── data/                   # Data storage
├── tests/                  # Test suite
└── docs/                   # Documentation
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution
- Additional lens profiles
- New defect detection algorithms
- Character preservation improvements
- Documentation and tutorials
- Bug fixes and optimizations

## Philosophy

VintageOptics is built on the principle that lens "imperfections" often contribute to the artistic quality of images. Our goal is to remove distracting defects while preserving the unique rendering characteristics that give vintage lenses their charm.

Unlike clinical correction software that aims for mathematical perfection, VintageOptics:
- Preserves beneficial aberrations like smooth bokeh and gentle vignetting
- Maintains lens-specific rendering styles
- Adapts corrections based on image content
- Allows fine control over character preservation

### Vintage ML Approach

Our "AI-winter first" philosophy extends to defect detection. By using classic algorithms from the 1950s-1980s as the primary layer:
- **Transparency**: Every decision is interpretable - you can inspect perceptron weights and decision boundaries
- **Education**: Learn ML history through working implementations of foundational algorithms
- **Efficiency**: Runs on CPU without GPU requirements, suitable for real-time processing
- **Effectiveness**: These simple algorithms often suffice for detecting dust, scratches, and other defects

The system uses modern ML only as a fallback for complex cases, maintaining our commitment to explainable, character-preserving processing.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Lensfun project for the comprehensive lens database
- OpenCV community for computer vision tools
- Vintage lens enthusiast community for inspiration and testing

## Citation

If you use VintageOptics in your research, please cite:

```bibtex
@software{vintageoptics2024,
  title = {VintageOptics: Character-Preserving Lens Correction},
  author = {Vinaik, Rohan},
  year = {2024},
  url = {https://github.com/rohanvinaik/VintageOptics}
}
```
