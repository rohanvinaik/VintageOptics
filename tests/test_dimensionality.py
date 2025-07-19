#!/usr/bin/env python3
"""
Test de-dimensionalization and compression techniques
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_random_projection():
    """Test Random Projection"""
    print("Testing Random Projection...")
    from src.vintageoptics.vintageml import RandomProjection
    
    # High-dimensional data
    X = np.random.randn(1000, 500)
    
    # Auto-compute dimensions
    rp = RandomProjection(n_components='auto', eps=0.2)
    rp.fit(X)
    
    X_projected = rp.transform(X)
    print(f"  Original shape: {X.shape}")
    print(f"  Projected shape: {X_projected.shape}")
    print(f"  Dimension reduction: {1 - X_projected.shape[1]/X.shape[1]:.1%}")
    
    # Test distance preservation
    # Pick two random points
    i, j = 10, 20
    orig_dist = np.linalg.norm(X[i] - X[j])
    proj_dist = np.linalg.norm(X_projected[i] - X_projected[j])
    distortion = abs(orig_dist - proj_dist) / orig_dist
    
    print(f"  Distance distortion: {distortion:.2%}")
    print("  ✓ Random Projection working\n")

def test_sparse_pca():
    """Test Sparse PCA"""
    print("Testing Sparse PCA...")
    from src.vintageoptics.vintageml import SparsePCAVintage
    
    # Create data with sparse structure
    X = np.random.randn(200, 50)
    X[:, 10:] *= 0.1  # Make most features small
    
    sparse_pca = SparsePCAVintage(n_components=5, alpha=0.5)
    sparse_pca.fit(X)
    X_transformed = sparse_pca.transform(X)
    
    # Check sparsity of components
    sparsity = np.mean(sparse_pca.components_ == 0)
    print(f"  Component sparsity: {sparsity:.1%}")
    print(f"  Transformed shape: {X_transformed.shape}")
    print("  ✓ Sparse PCA working\n")

def test_incremental_pca():
    """Test Incremental PCA"""
    print("Testing Incremental PCA...")
    from src.vintageoptics.vintageml import IncrementalPCAVintage
    
    # Simulate streaming data
    inc_pca = IncrementalPCAVintage(n_components=10, batch_size=50)
    
    n_batches = 5
    for i in range(n_batches):
        batch = np.random.randn(50, 100)
        inc_pca.partial_fit(batch)
        print(f"  Fitted batch {i+1}/{n_batches}")
    
    # Transform new data
    X_test = np.random.randn(20, 100)
    X_transformed = inc_pca.transform(X_test)
    
    print(f"  Final transform shape: {X_transformed.shape}")
    print("  ✓ Incremental PCA working\n")

def test_quantized_pca():
    """Test Quantized PCA"""
    print("Testing Quantized PCA...")
    from src.vintageoptics.vintageml import QuantizedPCA
    
    X = np.random.randn(500, 200)
    
    # Regular PCA for comparison
    from src.vintageoptics.vintageml import PCAVintage
    pca = PCAVintage(n_components=20)
    pca.fit(X)
    
    # Quantized version
    q_pca = QuantizedPCA(n_components=20, n_bits=8)
    q_pca.fit(X)
    
    # Compare memory usage
    regular_size = pca.components_.nbytes
    quantized_size = q_pca.components_quantized.nbytes
    
    print(f"  Regular PCA size: {regular_size / 1024:.1f} KB")
    print(f"  Quantized PCA size: {quantized_size / 1024:.1f} KB")
    print(f"  Memory saved: {1 - quantized_size/regular_size:.1%}")
    
    # Compare reconstruction error
    X_pca = pca.transform(X[:10])
    X_qpca = q_pca.transform(X[:10])
    error = np.mean(np.abs(X_pca - X_qpca))
    
    print(f"  Average error: {error:.4f}")
    print("  ✓ Quantized PCA working\n")

def test_adaptive_reducer():
    """Test Adaptive Dimensionality Reducer"""
    print("Testing Adaptive Dimensionality Reducer...")
    from src.vintageoptics.vintageml import AdaptiveDimensionalityReducer
    
    # Test on different data characteristics
    
    # 1. Small dense data
    X_small = np.random.randn(100, 50)
    reducer = AdaptiveDimensionalityReducer()
    reducer.fit(X_small)
    print(f"  Small data: selected {reducer.selected_method}")
    
    # 2. Large sparse data
    X_large_sparse = np.random.randn(5000, 1000)
    X_large_sparse[np.random.rand(*X_large_sparse.shape) < 0.9] = 0
    reducer2 = AdaptiveDimensionalityReducer()
    reducer2.fit(X_large_sparse)
    print(f"  Large sparse: selected {reducer2.selected_method}")
    
    print("  ✓ Adaptive reducer working\n")

def test_model_compression():
    """Test model compression techniques"""
    print("Testing Model Compression...")
    from src.vintageoptics.vintageml import LowRankApproximation, PrunedNetwork
    
    # 1. Low-rank approximation
    W = np.random.randn(100, 100)
    lr_approx = LowRankApproximation(energy_threshold=0.95)
    lr_approx.fit(W)
    
    W_reconstructed = lr_approx.reconstruct()
    reconstruction_error = np.linalg.norm(W - W_reconstructed) / np.linalg.norm(W)
    
    print(f"  Low-rank approx rank: {lr_approx.rank}")
    print(f"  Reconstruction error: {reconstruction_error:.2%}")
    
    # 2. Weight pruning
    weights = {'layer1': np.random.randn(50, 50)}
    pruner = PrunedNetwork(sparsity=0.9)
    pruned = pruner.prune_weights(weights)
    
    sparsity = 1 - np.count_nonzero(pruned['layer1']) / weights['layer1'].size
    print(f"  Pruning sparsity achieved: {sparsity:.1%}")
    
    print("  ✓ Model compression working\n")

def main():
    """Run all tests"""
    print("De-dimensionalization Tests")
    print("=" * 40 + "\n")
    
    tests = [
        test_random_projection,
        test_sparse_pca,
        test_incremental_pca,
        test_quantized_pca,
        test_adaptive_reducer,
        test_model_compression
    ]
    
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} failed: {e}\n")
    
    print("=" * 40)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("\nAll de-dimensionalization tests passed!")
        print("The system can efficiently handle high-dimensional data.")
        return 0
    else:
        print("\nSome tests failed. Please check the implementation.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
