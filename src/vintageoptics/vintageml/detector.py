# src/vintageoptics/vintageml/detector.py
"""
Vintage ML-based defect detector integrating all classic algorithms
Main entry point for the vintage ML detection pipeline
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from .perceptron import Perceptron, Adaline, MulticlassPerceptron
from .clustering import SelfOrganizingMap, KMeansVintage
from .dimensionality import PCAVintage, LDAVintage
from .dimensionality_enhanced import (
    AdaptiveDimensionalityReducer, RandomProjection, 
    SparsePCAVintage, QuantizedPCA
)
from .neighbors import KNNVintage, KNNInpainter

logger = logging.getLogger(__name__)


@dataclass
class VintageDefectResult:
    """Result from vintage ML detection"""
    defect_mask: np.ndarray
    defect_type: str
    confidence: float
    method_used: str
    features: Optional[np.ndarray] = None


class VintageMLDefectDetector:
    """
    Primary defect detector using vintage ML algorithms
    Implements the layered approach: vintage ML first, modern ML fallback
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.patch_size = config.get('vintageml', {}).get('patch_size', 16)
        self.pca_components = config.get('vintageml', {}).get('pca_components', 8)
        self.som_size = config.get('vintageml', {}).get('som_size', (10, 10))
        self.use_adaptive_reduction = config.get('vintageml', {}).get('use_adaptive_reduction', True)
        
        # Initialize components
        if self.use_adaptive_reduction:
            # Use adaptive dimensionality reduction
            self.dimensionality_reducer = AdaptiveDimensionalityReducer(
                target_dims=self.pca_components,
                preserve_variance=0.95
            )
            # Fast random projection for KNN
            self.random_projector = RandomProjection(n_components=min(32, self.pca_components))
        else:
            # Fallback to standard PCA
            self.pca = PCAVintage(n_components=self.pca_components)
            self.dimensionality_reducer = self.pca
            self.random_projector = None
            
        self.som = SelfOrganizingMap(
            input_dim=self.pca_components,  # Reduced dimensionality
            map_size=self.som_size
        )
        self.perceptron = None  # Initialized during training
        self.kmeans = KMeansVintage(n_clusters=5)  # Defect types
        self.knn = KNNVintage(n_neighbors=7)
        
        # Memory-efficient storage
        self.quantized_pca = None  # For model compression
        self.feature_cache = {}
        self.trained = False
        
    def extract_features(self, image: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Extract features from image using sliding window
        Returns features and patch positions
        """
        features = []
        positions = []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Normalize
        gray = gray.astype(np.float32) / 255.0
        
        # Sliding window feature extraction
        h, w = gray.shape
        stride = self.patch_size // 2
        
        for y in range(0, h - self.patch_size, stride):
            for x in range(0, w - self.patch_size, stride):
                # Extract patch
                patch = gray[y:y+self.patch_size, x:x+self.patch_size]
                
                # Basic feature engineering for vintage ML
                patch_features = self._engineer_patch_features(patch)
                
                features.append(patch_features)
                positions.append((y + self.patch_size//2, x + self.patch_size//2))
        
        return np.array(features), positions
    
    def _engineer_patch_features(self, patch: np.ndarray) -> np.ndarray:
        """
        Engineer features from patch suitable for vintage ML
        Simple, interpretable features from the pre-deep learning era
        """
        features = []
        
        # 1. Raw pixel values (flattened)
        features.extend(patch.flatten())
        
        # 2. Statistical moments
        features.append(np.mean(patch))
        features.append(np.std(patch))
        features.append(np.median(patch))
        
        # 3. Gradient features (simple edge detection)
        dx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        features.append(np.mean(np.abs(dx)))
        features.append(np.mean(np.abs(dy)))
        
        # 4. Local contrast
        local_contrast = np.std(patch) / (np.mean(patch) + 1e-6)
        features.append(local_contrast)
        
        # 5. Histogram features (4 bins)
        hist, _ = np.histogram(patch, bins=4, range=(0, 1))
        features.extend(hist / (patch.size + 1e-6))
        
        return np.array(features)
    
    def detect_defects(self, image: np.ndarray, 
                      lens_profile: Optional[Dict] = None) -> List[VintageDefectResult]:
        """
        Main detection pipeline using vintage ML methods
        """
        logger.info("Starting vintage ML defect detection")
        
        # Extract features
        features, positions = self.extract_features(image)
        
        if not self.trained:
            logger.warning("Detector not trained, using unsupervised methods only")
            return self._unsupervised_detection(image, features, positions)
        
        # Apply dimensionality reduction
        features_reduced = self.dimensionality_reducer.transform(features)
        
        # Apply random projection for fast KNN if available
        if self.random_projector and hasattr(self.random_projector, 'projection_matrix'):
            features_projected = self.random_projector.transform(features)
        else:
            features_projected = features_reduced
        
        # Multiple detection approaches
        results = []
        
        # 1. Perceptron-based binary detection
        perceptron_results = self._perceptron_detection(
            features_reduced, positions, image.shape[:2]
        )
        results.extend(perceptron_results)
        
        # 2. SOM-based anomaly detection
        som_results = self._som_detection(
            features_reduced, positions, image.shape[:2]
        )
        results.extend(som_results)
        
        # 3. KNN-based detection (use projected features for speed)
        knn_results = self._knn_detection(
            features_projected, positions, image.shape[:2]
        )
        results.extend(knn_results)
        
        # Combine and filter results
        final_results = self._combine_results(results)
        
        return final_results
    
    def _unsupervised_detection(self, image: np.ndarray, 
                               features: np.ndarray,
                               positions: List[Tuple[int, int]]) -> List[VintageDefectResult]:
        """
        Fallback unsupervised detection when not trained
        """
        results = []
        h, w = image.shape[:2]
        
        # Simple statistical outlier detection
        feature_mean = np.mean(features, axis=0)
        feature_std = np.std(features, axis=0)
        
        # Calculate z-scores
        z_scores = np.abs((features - feature_mean) / (feature_std + 1e-6))
        outlier_scores = np.max(z_scores, axis=1)
        
        # Threshold for outliers
        threshold = 3.0  # 3 standard deviations
        outlier_mask = outlier_scores > threshold
        
        if np.any(outlier_mask):
            # Create defect mask
            defect_mask = np.zeros((h, w), dtype=np.uint8)
            for idx, is_outlier in enumerate(outlier_mask):
                if is_outlier:
                    y, x = positions[idx]
                    # Mark patch region
                    y1 = max(0, y - self.patch_size//2)
                    y2 = min(h, y + self.patch_size//2)
                    x1 = max(0, x - self.patch_size//2)
                    x2 = min(w, x + self.patch_size//2)
                    defect_mask[y1:y2, x1:x2] = 255
            
            results.append(VintageDefectResult(
                defect_mask=defect_mask,
                defect_type="unknown",
                confidence=0.5,
                method_used="statistical_outlier"
            ))
        
        return results
    
    def _perceptron_detection(self, features: np.ndarray,
                            positions: List[Tuple[int, int]],
                            image_shape: Tuple[int, int]) -> List[VintageDefectResult]:
        """Perceptron-based defect detection"""
        if self.perceptron is None:
            return []
        
        # Get predictions
        predictions = self.perceptron.predict(features)
        
        # Create mask from predictions
        h, w = image_shape
        defect_mask = np.zeros((h, w), dtype=np.uint8)
        
        for idx, (y, x) in enumerate(positions):
            if predictions[idx] == 1:  # Defect class
                # Mark patch region
                y1 = max(0, y - self.patch_size//2)
                y2 = min(h, y + self.patch_size//2)
                x1 = max(0, x - self.patch_size//2)
                x2 = min(w, x + self.patch_size//2)
                defect_mask[y1:y2, x1:x2] = 255
        
        if np.any(defect_mask):
            return [VintageDefectResult(
                defect_mask=defect_mask,
                defect_type="perceptron_detected",
                confidence=0.7,
                method_used="perceptron"
            )]
        return []
    
    def _som_detection(self, features: np.ndarray,
                      positions: List[Tuple[int, int]],
                      image_shape: Tuple[int, int]) -> List[VintageDefectResult]:
        """SOM-based anomaly detection"""
        if not hasattr(self.som, 'trained') or not self.som.trained:
            return []
        
        # Map features to SOM
        bmu_indices = self.som.transform(features)
        
        # Calculate U-matrix for anomaly detection
        u_matrix = self.som.get_u_matrix()
        
        # Find anomalous BMUs (high U-matrix values)
        u_threshold = np.percentile(u_matrix, 90)
        
        h, w = image_shape
        defect_mask = np.zeros((h, w), dtype=np.uint8)
        
        for idx, (bmu_y, bmu_x) in enumerate(bmu_indices):
            if u_matrix[bmu_y, bmu_x] > u_threshold:
                y, x = positions[idx]
                # Mark patch region
                y1 = max(0, y - self.patch_size//2)
                y2 = min(h, y + self.patch_size//2)
                x1 = max(0, x - self.patch_size//2)
                x2 = min(w, x + self.patch_size//2)
                defect_mask[y1:y2, x1:x2] = 255
        
        if np.any(defect_mask):
            return [VintageDefectResult(
                defect_mask=defect_mask,
                defect_type="som_anomaly",
                confidence=0.6,
                method_used="som"
            )]
        return []
    
    def _knn_detection(self, features: np.ndarray,
                      positions: List[Tuple[int, int]],
                      image_shape: Tuple[int, int]) -> List[VintageDefectResult]:
        """KNN-based defect detection"""
        if self.knn.X_train is None:
            return []
        
        # Get predictions and probabilities
        predictions = self.knn.predict(features)
        probas = self.knn.predict_proba(features)
        
        h, w = image_shape
        defect_masks = {}
        
        # Create separate masks for each defect type
        for idx, (y, x) in enumerate(positions):
            if predictions[idx] != 0:  # Non-normal class
                defect_type = int(predictions[idx])
                if defect_type not in defect_masks:
                    defect_masks[defect_type] = np.zeros((h, w), dtype=np.uint8)
                
                # Mark patch region with confidence
                confidence = probas[idx, defect_type]
                y1 = max(0, y - self.patch_size//2)
                y2 = min(h, y + self.patch_size//2)
                x1 = max(0, x - self.patch_size//2)
                x2 = min(w, x + self.patch_size//2)
                defect_masks[defect_type][y1:y2, x1:x2] = int(255 * confidence)
        
        results = []
        defect_names = ["normal", "dust", "scratch", "fungus", "other"]
        
        for defect_type, mask in defect_masks.items():
            if np.any(mask):
                results.append(VintageDefectResult(
                    defect_mask=mask,
                    defect_type=defect_names[min(defect_type, len(defect_names)-1)],
                    confidence=np.mean(mask[mask > 0]) / 255.0,
                    method_used="knn"
                ))
        
        return results
    
    def _combine_results(self, results: List[VintageDefectResult]) -> List[VintageDefectResult]:
        """
        Combine results from different methods
        Simple voting or consensus approach
        """
        if not results:
            return []
        
        # Group by method
        by_method = {}
        for result in results:
            method = result.method_used
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(result)
        
        # If multiple methods agree on a region, increase confidence
        # For now, just return all results
        # TODO: Implement proper consensus mechanism
        
        return results
    
    def train(self, training_data: List[Tuple[np.ndarray, np.ndarray]]):
        """
        Train the vintage ML models
        
        Args:
            training_data: List of (image, defect_mask) tuples
        """
        logger.info(f"Training vintage ML models on {len(training_data)} images")
        
        # Collect all features and labels
        all_features = []
        all_labels = []
        
        for image, defect_mask in training_data:
            features, positions = self.extract_features(image)
            
            # Generate labels from defect mask
            labels = []
            for y, x in positions:
                # Check if patch center is in defect region
                if defect_mask[y, x] > 128:
                    labels.append(1)  # Defect
                else:
                    labels.append(0)  # Normal
            
            all_features.append(features)
            all_labels.extend(labels)
        
        # Combine all features
        all_features = np.vstack(all_features)
        all_labels = np.array(all_labels)
        
        # Fit dimensionality reduction
        logger.info("Fitting dimensionality reduction")
        self.dimensionality_reducer.fit(all_features)
        features_reduced = self.dimensionality_reducer.transform(all_features)
        
        # Fit random projector if available
        if self.random_projector:
            self.random_projector.fit(all_features)
            
        # Optional: Create quantized version for memory efficiency
        if self.config.get('vintageml', {}).get('use_quantization', False):
            self.quantized_pca = QuantizedPCA(n_components=self.pca_components)
            self.quantized_pca.fit(all_features)
        
        # Train perceptron
        logger.info("Training perceptron for binary classification")
        self.perceptron = Perceptron(
            n_features=features_reduced.shape[1],
            learning_rate=0.01,
            max_epochs=100
        )
        self.perceptron.fit(features_reduced, all_labels)
        
        # Train SOM
        logger.info("Training SOM for pattern discovery")
        self.som.fit(features_reduced)
        
        # Train KNN
        logger.info("Training KNN classifier")
        self.knn.fit(features_reduced, all_labels)
        
        self.trained = True
        logger.info("Vintage ML training complete")
    
    def save_models(self, path: str):
        """Save trained models"""
        import pickle
        model_data = {
            'pca': self.pca,
            'som': self.som,
            'perceptron': self.perceptron,
            'knn': self.knn,
            'trained': self.trained
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Saved vintage ML models to {path}")
    
    def load_models(self, path: str):
        """Load trained models"""
        import pickle
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.pca = model_data['pca']
        self.som = model_data['som']
        self.perceptron = model_data['perceptron']
        self.knn = model_data['knn']
        self.trained = model_data['trained']
        logger.info(f"Loaded vintage ML models from {path}")
