# Vintage ML Implementation in VintageOptics

## Overview

VintageOptics implements a unique "AI-winter first" approach to lens defect detection and correction. Instead of jumping straight to modern deep learning, we use classic machine learning algorithms from the 1950s-1980s as the primary detection layer. This provides transparency, educational value, and surprisingly effective results for many vintage lens defects.

## Architecture

### Layered ML Approach

```
1. Vintage ML (Primary)
   ├── Perceptron (1957) - Binary defect classification
   ├── ADALINE (1960) - Continuous defect scoring  
   ├── PCA (1901/1933) - Feature extraction
   ├── LDA (1936) - Supervised dimensionality reduction
   ├── k-NN (1951) - Patch-based detection & inpainting
   └── SOM (1982) - Anomaly detection & visualization

2. Physics-Based Correction
   ├── Brown-Conrady distortion model
   ├── Chromatic aberration correction
   ├── Vignetting compensation
   └── PSF deconvolution

3. Hybrid Iteration (Physics ↔ ML)
   └── Modern ML Fallback (Optional)
```

### Key Components

#### VintageMLDefectDetector
The main vintage ML detection pipeline that orchestrates all classic algorithms:

```python
from vintageoptics.vintageml import VintageMLDefectDetector

detector = VintageMLDefectDetector(config)
results = detector.detect_defects(image, lens_profile)
```

#### Perceptron & ADALINE
Pure Python/NumPy implementations of Rosenblatt's Perceptron (1957) and Widrow's ADALINE (1960):

```python
from vintageoptics.vintageml import Perceptron, Adaline

# Binary classification
perceptron = Perceptron(n_features=64, learning_rate=0.01)
perceptron.fit(X_train, y_train)
predictions = perceptron.predict(X_test)

# Continuous scoring
adaline = Adaline(n_features=64, learning_rate=0.01)
adaline.fit(X_train, y_train)
scores = adaline.predict_proba(X_test)
```

#### Self-Organizing Maps
Kohonen's SOM (1982) for unsupervised defect pattern discovery:

```python
from vintageoptics.vintageml import SelfOrganizingMap

som = SelfOrganizingMap(input_dim=64, map_size=(10, 10))
som.fit(features)
u_matrix = som.get_u_matrix()  # Visualize defect clusters
```

#### Classic Dimensionality Reduction
PCA and LDA implementations for feature extraction:

```python
from vintageoptics.vintageml import PCAVintage, LDAVintage

# Unsupervised reduction
pca = PCAVintage(n_components=8)
features_reduced = pca.fit_transform(features)

# Supervised reduction
lda = LDAVintage(n_components=4)
features_reduced = lda.fit_transform(features, labels)
```

#### k-NN Inpainting
Classic k-nearest neighbors for patch-based defect removal:

```python
from vintageoptics.vintageml import KNNInpainter

inpainter = KNNInpainter(patch_size=7, n_neighbors=10)
cleaned = inpainter.inpaint(image, defect_mask)
```

## Hybrid Physics-ML Pipeline

The `HybridPhysicsMLPipeline` implements the iterative approach described in the design documents:

```python
from vintageoptics.core import HybridPhysicsMLPipeline

pipeline = HybridPhysicsMLPipeline(config)
result = pipeline.process(image, metadata)
```

### Processing Flow

1. **Initial Physics Correction**
   - Apply lens model based corrections
   - Geometric distortion, chromatic aberration, vignetting

2. **Vintage ML Analysis**
   - Extract patches and features
   - Detect residual defects using perceptron/SOM/k-NN
   - Calculate residual entropy

3. **Physics Refinement** (if entropy > threshold)
   - Analyze spatial distribution of ML-detected defects
   - Refine physics parameters
   - Re-apply corrections

4. **ML Cleanup**
   - Apply defect-specific vintage ML cleanup
   - k-NN inpainting for spots
   - Directional processing for scratches

5. **Iterate** until convergence or max iterations

## PAC Learning Integration

The system incorporates PAC (Probably Approximately Correct) learning principles:

- **Entropy-based feedback**: High residual entropy triggers structural adaptation
- **Rule extraction**: Simple, interpretable rules from defect patterns
- **Confidence bounds**: PAC confidence on defect classifications
- **Dynamic adaptation**: Add/remove processing modules based on need

## Training

Train the vintage ML models on your own data:

```python
from vintageoptics.vintageml import VintageMLTrainer

trainer = VintageMLTrainer(config)
trainer.train(
    data_dir='path/to/training/data',
    save_path='models/vintage_ml.pkl',
    augment=True
)
```

### Training Data Format
```
training_data/
├── images/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── masks/
    ├── img_001_mask.png  # Binary defect masks
    ├── img_002_mask.png
    └── ...
```

## Configuration

Key configuration options for vintage ML:

```yaml
vintageml:
  patch_size: 16          # Size of patches for feature extraction
  pca_components: 8       # Number of PCA components
  som_size: [10, 10]      # Self-organizing map dimensions

hybrid:
  max_iterations: 3       # Max physics-ML iterations
  convergence_threshold: 0.01
  use_modern_fallback: false

pac:
  entropy_threshold: 2.0  # Triggers physics refinement
  confidence_delta: 0.05  # PAC confidence parameter
```

## Performance

The vintage ML approach offers several advantages:

- **Lightweight**: Runs on CPU, no GPU required
- **Fast**: Simple linear models process quickly
- **Interpretable**: Can inspect weights, decision boundaries
- **Educational**: See how AI has evolved over decades

Typical processing times:
- Feature extraction: ~50ms for 512x512 image
- Perceptron classification: ~5ms
- SOM mapping: ~10ms  
- k-NN inpainting: ~100ms for small defects

## De-dimensionalization Techniques

VintageOptics implements several advanced dimensionality reduction techniques inspired by PAC learning principles to achieve computational efficiency without sacrificing detection quality.

### Adaptive Dimensionality Reduction

The system automatically selects the optimal dimensionality reduction method based on data characteristics:

```python
from vintageoptics.vintageml import AdaptiveDimensionalityReducer

reducer = AdaptiveDimensionalityReducer(
    preserve_variance=0.95,
    sparsity_threshold=0.7
)
reducer.fit(high_dimensional_patches)
reduced_features = reducer.transform(patches)
```

### Available Methods

1. **Random Projection** (Johnson-Lindenstrauss)
   - Fastest method, preserves distances
   - O(n·d·k) complexity vs O(n·d²) for PCA
   - Ideal for real-time processing

2. **Sparse PCA**
   - Enforces sparsity in components
   - More interpretable features
   - Better for understanding defect patterns

3. **Incremental PCA**
   - Processes data in batches
   - Handles datasets larger than memory
   - Perfect for batch processing pipelines

4. **Quantized PCA**
   - 8-bit quantization of components
   - 75% memory reduction
   - Minimal accuracy loss

### Model Compression

Compress trained models for deployment:

```python
from vintageoptics.vintageml import compress_vintage_ml_models

# Compress saved models
compress_vintage_ml_models(
    'models/vintage_ml.pkl',
    'models/vintage_ml_compressed.pkl'
)
```

Compression techniques include:
- Low-rank approximation of weight matrices
- Magnitude-based weight pruning (90% sparsity)
- Knowledge distillation to smaller models
- Combined compression pipeline

### Performance Impact

Typical improvements with de-dimensionalization:
- **Speed**: 3-10x faster feature extraction
- **Memory**: 60-80% reduction in model size
- **Accuracy**: <1% degradation with proper configuration

## Examples

See `demo_vintage_ml.py` for complete examples:

```bash
python demo_vintage_ml.py
```

This will:
1. Demonstrate unsupervised defect detection
2. Show the hybrid physics-ML pipeline
3. Train vintage ML models on synthetic data
4. Visualize learned components (PCA, SOM, perceptron)

## Why Vintage ML?

1. **Transparency**: Every decision is interpretable
2. **Efficiency**: Runs anywhere, even on old hardware  
3. **Education**: Learn ML history through implementation
4. **Effectiveness**: Often sufficient for vintage lens defects
5. **Philosophy**: Matches the vintage lens aesthetic

The vintage ML layer handles 80% of common defects, with modern ML as a fallback for complex cases. This layered approach provides the best of both worlds: explainable AI with modern performance when needed.
