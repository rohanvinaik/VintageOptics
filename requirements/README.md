# VintageOptics Requirements

This directory contains the various requirement files for different installation scenarios.

## Files

### base.txt
Core dependencies required for basic VintageOptics functionality. Install with:
```bash
pip install -r requirements/base.txt
```

### dev.txt
Additional dependencies for development, including testing and linting tools. Install with:
```bash
pip install -r requirements/dev.txt
```

### gpu.txt
GPU-specific dependencies for CUDA acceleration. Install with:
```bash
pip install -r requirements/gpu.txt
```

### ml.txt
Machine learning dependencies for ML-based correction methods. Install with:
```bash
pip install -r requirements/ml.txt
```

### gui.txt
Dependencies for the GUI interface. Install with:
```bash
pip install -r requirements/gui.txt
```

### gui_full.txt
Complete GUI dependencies including optional features. Install with:
```bash
pip install -r requirements/gui_full.txt
```

## Installation Examples

For basic usage:
```bash
pip install -r requirements/base.txt
```

For development:
```bash
pip install -r requirements/base.txt -r requirements/dev.txt
```

For full installation with all features:
```bash
pip install -r requirements/base.txt -r requirements/ml.txt -r requirements/gpu.txt -r requirements/gui_full.txt
```
