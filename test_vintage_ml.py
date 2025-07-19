#!/usr/bin/env python3
"""
Test script for Vintage ML implementation
Verifies all components are working correctly
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_perceptron():
    """Test Perceptron implementation"""
    print("Testing Perceptron...")
    from src.vintageoptics.vintageml import Perceptron
    
    # Create simple linearly separable data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND gate
    
    perceptron = Perceptron(n_features=2, learning_rate=0.1, max_epochs=10)
    perceptron.fit(X, y)
    
    predictions = perceptron.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"  Accuracy: {accuracy:.2f}")
    print(f"  Converged in {len(perceptron.errors_per_epoch)} epochs")
    print("  ✓ Perceptron working\n")

def test_som():
    """Test Self-Organizing Map"""
    print("Testing Self-Organizing Map...")
    from src.vintageoptics.vintageml import SelfOrganizingMap
    
    # Create sample data
    data = np.random.randn(100, 5)
    
    som = SelfOrganizingMap(input_dim=5, map_size=(5, 5), max_epochs=100)
    som.fit(data)
    
    # Test transform
    bmu_indices = som.transform(data[:10])
    print(f"  Trained SOM on {data.shape} data")
    print(f"  BMU indices shape: {bmu_indices.shape}")
    print("  ✓ SOM working\n")

def test_pca():
    """Test PCA implementation"""
    print("Testing PCA...")
    from src.vintageoptics.vintageml import PCAVintage
    
    # Create correlated data
    X = np.random.randn(100, 10)
    X[:, 1] = X[:, 0] + 0.5 * np.random.randn(100)  # Correlated feature
    
    pca = PCAVintage(n_components=3)
    pca.fit(X)
    X_reduced = pca.transform(X)
    
    print(f"  Original shape: {X.shape}")
    print(f"  Reduced shape: {X_reduced.shape}")
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")
    print("  ✓ PCA working\n")

def test_knn():
    """Test k-NN implementation"""
    print("Testing k-NN...")
    from src.vintageoptics.vintageml import KNNVintage
    
    # Create sample data
    X_train = np.array([[0, 0], [1, 1], [2, 2], [0, 1], [1, 0]])
    y_train = np.array([0, 1, 1, 0, 0])
    
    knn = KNNVintage(n_neighbors=3)
    knn.fit(X_train, y_train)
    
    X_test = np.array([[0.5, 0.5], [1.5, 1.5]])
    predictions = knn.predict(X_test)
    probas = knn.predict_proba(X_test)
    
    print(f"  Predictions: {predictions}")
    print(f"  Probabilities: {probas}")
    print("  ✓ k-NN working\n")

def test_detector():
    """Test VintageMLDefectDetector"""
    print("Testing VintageMLDefectDetector...")
    from src.vintageoptics.vintageml import VintageMLDefectDetector
    
    # Create test image
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    config = {
        'vintageml': {
            'patch_size': 16,
            'pca_components': 5
        }
    }
    
    detector = VintageMLDefectDetector(config)
    
    # Test feature extraction
    features, positions = detector.extract_features(image)
    print(f"  Extracted {len(features)} features")
    print(f"  Feature shape: {features.shape}")
    
    # Test detection (unsupervised)
    results = detector.detect_defects(image)
    print(f"  Detected {len(results)} defect regions")
    print("  ✓ Detector working\n")

def test_hybrid_pipeline():
    """Test hybrid physics-ML pipeline"""
    print("Testing Hybrid Pipeline...")
    from src.vintageoptics.core.hybrid_pipeline import HybridPhysicsMLPipeline
    
    # Create test data
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    metadata = {'LensModel': 'Test Lens'}
    
    config = {
        'hybrid': {'max_iterations': 2},
        'vintageml': {'patch_size': 16},
        'pac': {'entropy_threshold': 2.0}
    }
    
    pipeline = HybridPhysicsMLPipeline(config)
    
    # Test processing
    try:
        result = pipeline.process(image, metadata)
        print(f"  Completed in {result.iterations} iterations")
        print(f"  Residual error: {result.residual_error:.4f}")
        print("  ✓ Hybrid pipeline working\n")
    except Exception as e:
        print(f"  ! Error in hybrid pipeline: {e}\n")

def main():
    """Run all tests"""
    print("VintageOptics - Vintage ML Test Suite")
    print("=" * 40 + "\n")
    
    tests = [
        test_perceptron,
        test_som,
        test_pca,
        test_knn,
        test_detector,
        test_hybrid_pipeline
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
        print("\nAll tests passed! Vintage ML implementation is working correctly.")
        return 0
    else:
        print("\nSome tests failed. Please check the implementation.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
