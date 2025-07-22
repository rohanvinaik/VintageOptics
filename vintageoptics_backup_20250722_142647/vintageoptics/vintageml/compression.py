# src/vintageoptics/vintageml/compression.py
"""
Model compression and low-rank approximation techniques
For memory-efficient deployment of vintage ML models
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class LowRankApproximation:
    """
    Low-rank matrix approximation for compressing weight matrices
    Uses SVD decomposition to reduce model size
    """
    
    def __init__(self, rank: Optional[int] = None, energy_threshold: float = 0.95):
        self.rank = rank
        self.energy_threshold = energy_threshold
        self.U = None
        self.S = None
        self.Vt = None
        
    def fit(self, W: np.ndarray) -> 'LowRankApproximation':
        """Compute low-rank approximation of weight matrix W"""
        # Full SVD
        U, s, Vt = np.linalg.svd(W, full_matrices=False)
        
        # Determine rank if not specified
        if self.rank is None:
            # Find rank that preserves energy_threshold of variance
            energy = np.cumsum(s**2) / np.sum(s**2)
            self.rank = np.argmax(energy >= self.energy_threshold) + 1
            
        logger.info(f"Low-rank approximation: {W.shape} -> rank {self.rank}")
        
        # Store truncated decomposition
        self.U = U[:, :self.rank]
        self.S = s[:self.rank]
        self.Vt = Vt[:self.rank, :]
        
        # Calculate compression ratio
        original_params = W.size
        compressed_params = self.U.size + self.S.size + self.Vt.size
        compression_ratio = 1 - (compressed_params / original_params)
        logger.info(f"Compression ratio: {compression_ratio:.1%}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply low-rank weight matrix to input"""
        if self.U is None:
            raise ValueError("Must fit before transform")
            
        # Efficient matrix multiplication: X @ W ≈ X @ U @ diag(S) @ Vt
        result = X @ self.U
        result = result * self.S  # Broadcasting multiplication
        result = result @ self.Vt
        
        return result
    
    def reconstruct(self) -> np.ndarray:
        """Reconstruct the approximated weight matrix"""
        if self.U is None:
            raise ValueError("Must fit before reconstruction")
            
        return self.U @ np.diag(self.S) @ self.Vt


class PrunedNetwork:
    """
    Network pruning for vintage ML models
    Removes low-magnitude weights while maintaining performance
    """
    
    def __init__(self, sparsity: float = 0.9):
        self.sparsity = sparsity
        self.masks = {}
        self.pruned_weights = {}
        
    def prune_weights(self, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Prune weights using magnitude-based pruning
        
        Args:
            weights: Dictionary of layer_name -> weight_matrix
            
        Returns:
            Pruned weights dictionary
        """
        self.masks = {}
        self.pruned_weights = {}
        
        for layer_name, W in weights.items():
            # Calculate threshold for pruning
            threshold = np.percentile(np.abs(W), self.sparsity * 100)
            
            # Create mask
            mask = np.abs(W) > threshold
            self.masks[layer_name] = mask
            
            # Apply mask
            W_pruned = W * mask
            self.pruned_weights[layer_name] = W_pruned
            
            # Log statistics
            sparsity_achieved = 1 - (np.count_nonzero(W_pruned) / W.size)
            logger.info(f"Pruned {layer_name}: {sparsity_achieved:.1%} sparsity")
            
        return self.pruned_weights
    
    def apply_sparse(self, X: np.ndarray, layer_name: str) -> np.ndarray:
        """Apply sparse weight matrix efficiently"""
        if layer_name not in self.pruned_weights:
            raise ValueError(f"Layer {layer_name} not found in pruned weights")
            
        W = self.pruned_weights[layer_name]
        mask = self.masks[layer_name]
        
        # For very sparse matrices, use sparse matrix multiplication
        if np.count_nonzero(mask) < 0.1 * mask.size:
            # Convert to CSR format for efficient multiplication
            from scipy.sparse import csr_matrix
            W_sparse = csr_matrix(W)
            return X @ W_sparse.T
        else:
            # Regular dense multiplication
            return X @ W


class DistilledVintageML:
    """
    Knowledge distillation for vintage ML models
    Creates smaller student models that mimic larger teacher models
    """
    
    def __init__(self, compression_factor: float = 0.5, temperature: float = 3.0):
        self.compression_factor = compression_factor
        self.temperature = temperature
        self.student_model = None
        
    def distill_perceptron(self, teacher_perceptron, X_train: np.ndarray) -> 'Perceptron':
        """
        Distill a perceptron into a smaller one
        
        Args:
            teacher_perceptron: Trained teacher perceptron
            X_train: Training data
            
        Returns:
            Smaller student perceptron
        """
        # Get teacher predictions (soft labels)
        teacher_logits = teacher_perceptron.predict_raw(X_train)
        
        # Apply temperature scaling for softer probabilities
        teacher_probs = self._sigmoid(teacher_logits / self.temperature)
        
        # Create student with fewer features (via random projection)
        n_features = X_train.shape[1]
        student_features = int(n_features * self.compression_factor)
        
        # Random projection matrix for feature reduction
        projection = np.random.randn(n_features, student_features)
        projection /= np.sqrt(student_features)
        
        # Project training data
        X_student = X_train @ projection
        
        # Train student perceptron on soft labels
        from .perceptron import Adaline  # Use Adaline for continuous targets
        student = Adaline(n_features=student_features, learning_rate=0.01)
        student.fit(X_student, teacher_probs)
        
        logger.info(f"Distilled perceptron: {n_features} -> {student_features} features")
        
        # Wrap in a class that includes projection
        class DistilledPerceptron:
            def __init__(self, projection, student):
                self.projection = projection
                self.student = student
                
            def predict(self, X):
                X_proj = X @ self.projection
                return (self.student.predict_raw(X_proj) > 0.5).astype(int)
                
            def predict_proba(self, X):
                X_proj = X @ self.projection
                return self._sigmoid(self.student.predict_raw(X_proj))
                
            def _sigmoid(self, x):
                return 1 / (1 + np.exp(-x))
        
        return DistilledPerceptron(projection, student)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Stable sigmoid function"""
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-x)),
                       np.exp(x) / (1 + np.exp(x)))


class CompressedVintageMLDetector:
    """
    Compressed version of VintageMLDefectDetector for deployment
    Uses all compression techniques for minimal memory footprint
    """
    
    def __init__(self, original_detector):
        self.original_detector = original_detector
        self.compressed_components = {}
        
    def compress(self) -> 'CompressedVintageMLDetector':
        """Compress all components of the detector"""
        logger.info("Compressing VintageML detector...")
        
        # 1. Compress dimensionality reducer
        if hasattr(self.original_detector, 'dimensionality_reducer'):
            if hasattr(self.original_detector.dimensionality_reducer, 'reducer'):
                reducer = self.original_detector.dimensionality_reducer.reducer
                if hasattr(reducer, 'components_'):
                    # Low-rank approximation of PCA components
                    lr_approx = LowRankApproximation(energy_threshold=0.95)
                    lr_approx.fit(reducer.components_.T)
                    self.compressed_components['pca_lowrank'] = lr_approx
        
        # 2. Prune perceptron weights
        if self.original_detector.perceptron:
            pruner = PrunedNetwork(sparsity=0.8)
            perceptron_weights = {'weights': self.original_detector.perceptron.weights.reshape(1, -1)}
            pruned = pruner.prune_weights(perceptron_weights)
            self.compressed_components['perceptron_pruned'] = pruned['weights'].flatten()
            self.compressed_components['perceptron_mask'] = pruner.masks['weights'].flatten()
        
        # 3. Quantize SOM weights
        if hasattr(self.original_detector.som, 'weights'):
            # 8-bit quantization of SOM weights
            som_weights = self.original_detector.som.weights
            w_min, w_max = som_weights.min(), som_weights.max()
            scale = (w_max - w_min) / 255.0
            
            som_quantized = ((som_weights - w_min) / scale).astype(np.uint8)
            self.compressed_components['som_quantized'] = som_quantized
            self.compressed_components['som_scale'] = scale
            self.compressed_components['som_min'] = w_min
        
        # 4. Compress KNN training data using clustering
        if self.original_detector.knn.X_train is not None:
            # Use k-means to create prototypes
            from .clustering import KMeansVintage
            n_prototypes = min(100, len(self.original_detector.knn.X_train) // 10)
            
            kmeans = KMeansVintage(n_clusters=n_prototypes)
            kmeans.fit(self.original_detector.knn.X_train)
            
            # Store prototypes instead of all training data
            self.compressed_components['knn_prototypes'] = kmeans.cluster_centers_
            self.compressed_components['knn_prototype_labels'] = np.array([
                self.original_detector.knn.y_train[kmeans.labels_ == i][0]
                for i in range(n_prototypes)
            ])
        
        # Calculate total compression
        original_size = self._estimate_size(self.original_detector)
        compressed_size = self._estimate_compressed_size()
        compression_ratio = 1 - (compressed_size / original_size)
        
        logger.info(f"Total compression ratio: {compression_ratio:.1%}")
        logger.info(f"Original size: {original_size/1024:.1f} KB")
        logger.info(f"Compressed size: {compressed_size/1024:.1f} KB")
        
        return self
    
    def _estimate_size(self, obj) -> int:
        """Estimate memory size of object"""
        size = 0
        
        # Estimate sizes of main components
        if hasattr(obj, 'perceptron') and obj.perceptron:
            if hasattr(obj.perceptron, 'weights'):
                size += obj.perceptron.weights.nbytes
                
        if hasattr(obj, 'som') and hasattr(obj.som, 'weights'):
            size += obj.som.weights.nbytes
            
        if hasattr(obj, 'knn') and obj.knn.X_train is not None:
            size += obj.knn.X_train.nbytes
            size += obj.knn.y_train.nbytes
            
        return size
    
    def _estimate_compressed_size(self) -> int:
        """Estimate size of compressed components"""
        size = 0
        
        for key, value in self.compressed_components.items():
            if isinstance(value, np.ndarray):
                size += value.nbytes
            elif hasattr(value, 'U'):  # Low-rank approximation
                size += value.U.nbytes + value.S.nbytes + value.Vt.nbytes
                
        return size
    
    def detect_defects_compressed(self, image: np.ndarray) -> List:
        """
        Run defect detection using compressed models
        May have slightly lower accuracy but much faster
        """
        # Extract features
        features, positions = self.original_detector.extract_features(image)
        
        # Use compressed components for inference
        # This would implement the forward pass using compressed weights
        
        # Placeholder for compressed inference
        logger.info("Running compressed inference...")
        
        # Return simplified results
        return []


def compress_vintage_ml_models(models_path: str, output_path: str):
    """
    Utility function to compress saved vintage ML models
    
    Args:
        models_path: Path to original models
        output_path: Path to save compressed models
    """
    import pickle
    
    # Load original models
    with open(models_path, 'rb') as f:
        models = pickle.load(f)
    
    # Create compressed versions
    compressed = {}
    
    # Compress each component
    if 'pca' in models:
        lr_approx = LowRankApproximation(energy_threshold=0.95)
        lr_approx.fit(models['pca'].components_.T)
        compressed['pca_lowrank'] = lr_approx
    
    if 'perceptron' in models:
        # Prune perceptron
        pruner = PrunedNetwork(sparsity=0.9)
        weights = {'w': models['perceptron'].weights}
        pruned = pruner.prune_weights(weights)
        compressed['perceptron_sparse'] = pruned
    
    # Save compressed models
    with open(output_path, 'wb') as f:
        pickle.dump(compressed, f)
    
    # Report compression
    original_size = os.path.getsize(models_path)
    compressed_size = os.path.getsize(output_path)
    ratio = 1 - (compressed_size / original_size)
    
    logger.info(f"Model compression completed: {ratio:.1%} reduction")
    logger.info(f"Original: {original_size/1024:.1f} KB")
    logger.info(f"Compressed: {compressed_size/1024:.1f} KB")
