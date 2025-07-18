# VintageOptics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Advanced Lens Correction and Synthesis with Hyperdimensional Computing**

VintageOptics is a comprehensive Python library for vintage lens correction, defect removal, and characteristic synthesis. It combines physical optics simulation with cutting-edge hyperdimensional computing for robust error correction and pattern recognition.

## 🌟 Key Features

### Hyperdimensional Computing Integration
- **10,000-dimensional vector representations** for robust defect encoding
- **Orthogonal error separation** leveraging analog vs digital error characteristics  
- **Topological defect analysis** for noise-invariant feature detection
- **Constraint-based correction** treating errors as logic puzzle constraints

### Advanced Lens Analysis
- **Unified lens detection** combining vintage and electronic characteristics
- **Comprehensive quality metrics** with perceptual and aesthetic scoring
- **Lens fingerprinting** and signature matching
- **Automatic defect detection** (dust, scratches, fungus, haze)

### Physical Optics Simulation
- **Brown-Conrady distortion model** with full parameter estimation
- **Chromatic aberration modeling** with wavelength-dependent effects
- **Vignetting simulation** with customizable falloff profiles
- **Diffraction-limited PSF** calculation

### Lens Synthesis
- **Characteristic synthesis** from lens profiles
- **Depth-aware bokeh rendering** with custom aperture shapes
- **Coating-specific effects** (uncoated, single-coated, multi-coated)
- **Lens flare and ghost generation** based on optical formula

## 🎨 Graphical User Interface

VintageOptics now includes a modern web-based GUI for easy testing and deployment!

### GUI Features
- Intuitive drag-and-drop interface
- Real-time preview of processing results
- Pre-configured vintage lens profiles
- Interactive defect simulation
- Processing statistics and quality metrics

### Running the GUI

```bash
# Quick start - runs both backend and frontend
./run_with_gui.sh
```

Or manually:
```bash
# Terminal 1: Start the backend API
python frontend_api.py

# Terminal 2: Start the frontend
cd frontend && npm start
```

The GUI will be available at `http://localhost:3000`

See [frontend/README.md](frontend/README.md) for detailed GUI documentation.

## 🚀 Quick Start

### Installation

```bash
pip install vintageoptics
```

For GPU acceleration:
```bash
pip install vintageoptics[gpu]
```

### Basic Usage

```python
import vintageoptics as vo

# Quick HD correction
corrected = vo.quick_hd_correction('vintage_photo.jpg', strength=0.8)

# Analyze lens defects
defects = vo.analyze_lens_defects('lens_image.jpg')
print(f"Found {defects['dust_spots']} dust spots, {defects['scratches']} scratches")

# Separate vintage and digital errors
separation = vo.separate_vintage_digital_errors('mixed_image.jpg')
vintage_errors = separation['vintage_errors']
digital_errors = separation['digital_errors']
```

### Advanced Pipeline

```python
from vintageoptics import VintageOpticsPipeline, PipelineConfig, ProcessingMode

# Configure pipeline
config = PipelineConfig(
    mode=ProcessingMode.HYBRID,
    use_hd=True,
    correction_strength=0.8,
    target_quality=0.85
)

# Create pipeline
pipeline = VintageOpticsPipeline(config)

# Process image
result = pipeline.process('vintage_photo.jpg')

print(f"Quality improved from {result.initial_quality:.1%} to {result.quality_metrics.overall_quality:.1%}")
print(f"Processing time: {result.processing_time:.2f}s")
```

### Lens Synthesis

```python
from vintageoptics import synthesize_lens_effect

# Apply classic lens characteristics
helios_style = synthesize_lens_effect(
    'modern_photo.jpg',
    'Helios 44-2 58mm f/2',
    strength=0.7
)

# Create custom lens profile
from vintageoptics import LensSynthesizer

synthesizer = LensSynthesizer()
custom_profile = synthesizer.create_custom_profile(
    'My Vintage Lens',
    reference_images=['ref1.jpg', 'ref2.jpg', 'ref3.jpg']
)
```

## 🔬 Hyperdimensional Computing Features

### How It Works

VintageOptics uses hyperdimensional computing to treat vintage lens errors and digital sensor errors as orthogonal constraints in high-dimensional space:

```python
from vintageoptics import HyperdimensionalLensAnalyzer

# Create HD analyzer
hd_analyzer = HyperdimensionalLensAnalyzer(dimension=10000)

# Analyze and correct with HD methods
result = hd_analyzer.analyze_and_correct(
    image,
    mode='auto',  # Automatically detects vintage vs digital errors
    strength=0.8
)

# Create lens signature for matching
signature = hd_analyzer.create_lens_signature(
    'Canon FD 50mm f/1.4',
    sample_images=[img1, img2, img3]
)

# Match unknown lens
matched_lens = hd_analyzer.match_lens(unknown_image)
```

### Topological Defect Analysis

```python
from vintageoptics import TopologicalDefectAnalyzer

analyzer = TopologicalDefectAnalyzer()
analysis = analyzer.analyze_defects(image)

# Results include topological features
print(f"0-dimensional features (dust): {len(analysis['dust_features'])}")
print(f"1-dimensional features (scratches): {len(analysis['scratch_features'])}")
print(f"2-dimensional features (regions): {len(analysis['region_features'])}")
```

## 📊 REST API

VintageOptics includes a FastAPI-based REST API:

```bash
# Start the API server
uvicorn vintageoptics.api:app --reload

# Or programmatically
from vintageoptics.api import app
# Use with your favorite ASGI server
```

### API Endpoints

- `POST /process` - Process single image
- `POST /analyze/hd` - HD analysis
- `POST /detect/lens` - Lens detection
- `POST /synthesize/{lens_name}` - Apply lens effect
- `POST /batch/process` - Batch processing
- `GET /profiles` - List available lens profiles

Example:
```bash
curl -X POST "http://localhost:8000/process" \
  -F "image=@vintage_photo.jpg" \
  -F "mode=hybrid" \
  -F "strength=0.8" \
  -F "use_hd=true"
```

## 🎯 Use Cases

### Professional Photography
- Correct vintage lens defects while preserving character
- Match modern shots to vintage lens aesthetics
- Create consistent look across mixed lens footage

### Film Restoration
- Remove age-related defects from archival footage
- Separate intentional vintage look from unwanted artifacts
- Enhance quality while preserving artistic intent

### Computer Vision
- Preprocess images with lens artifacts for ML pipelines
- Generate training data with realistic optical defects
- Calibrate multi-camera systems with different lenses

### Creative Applications
- Achieve specific vintage looks without physical lenses
- Blend characteristics from multiple classic lenses
- Create impossible lens effects through profile synthesis

## 🛠️ Advanced Configuration

### Pipeline Configuration

```python
config = PipelineConfig(
    # Processing mode
    mode=ProcessingMode.HYBRID,  # AUTO, CORRECTION, SYNTHESIS, HYBRID
    
    # HD settings
    use_hd=True,
    hd_dimension=10000,
    
    # Correction settings
    correction_strength=0.8,
    preserve_character=True,
    adaptive_strength=True,
    
    # Quality settings
    target_quality=0.85,
    max_iterations=3,
    
    # Performance
    use_gpu=True,
    enable_caching=True,
    parallel_processing=True
)
```

### Custom Lens Profiles

```python
from vintageoptics import LensProfile, BokehShape

profile = LensProfile(
    name="Custom Vintage",
    focal_length=85.0,
    max_aperture=1.8,
    
    # Optical characteristics
    vignetting_amount=0.4,
    vignetting_falloff=2.5,
    distortion_k1=-0.05,
    chromatic_aberration=2.0,
    
    # Bokeh properties
    bokeh_quality=0.9,
    bokeh_shape=BokehShape.HEXAGONAL,
    aperture_blades=6,
    
    # Coating
    coating_type="single-coated",
    flare_intensity=0.3
)
```

## 📈 Performance

VintageOptics is optimized for performance:

- **GPU acceleration** for physics simulation
- **Parallel processing** for batch operations
- **Intelligent caching** of computed results
- **Efficient HD operations** using vectorized computing

Benchmark results (on RTX 3080):
- 4K image correction: ~0.8s
- HD defect analysis: ~0.3s
- Full pipeline with synthesis: ~1.5s

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas of interest:
- Additional lens profiles
- New defect detection algorithms
- Performance optimizations
- Documentation improvements

## 📚 Documentation

Full documentation available at: [https://vintageoptics.readthedocs.io](https://vintageoptics.readthedocs.io)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Lens profile data from Lensfun project
- HD computing concepts inspired by Kanerva's work
- Optical formulas from Zemax OpticStudio documentation

## 📞 Contact

- Issues: [GitHub Issues](https://github.com/yourusername/vintageoptics/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/vintageoptics/discussions)
- Email: vintageoptics@example.com

---

**Made with ❤️ for photographers, filmmakers, and computer vision researchers**
