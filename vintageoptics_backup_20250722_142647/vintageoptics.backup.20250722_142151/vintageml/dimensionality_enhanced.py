# src/vintageoptics/vintageml/dimensionality_enhanced.py
"""
Enhanced dimensionality reduction with PAC-inspired techniques
Implements advanced de-dimensionalization for computational efficiency
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
import logging
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

logger = logging.getLogger(__name__)


class RandomProjection:
    """
    Johnson-Lindenstrauss random projection for fast dimensionality reduction
    Preserves distances with high probability using random matrices
    """
    
    def __init__(self, n_components: int = 'auto', eps: float = 0.1):
        self.n_components = n_components
        self.eps = eps
        self.projection_matrix = None
        
    def fit(self, X: np.ndarray) -> 'RandomProjection':
        """Generate random projection matrix"""
        n_samples, n_features = X.shape
        
        # Auto-compute n_components using JL lemma
        if self.n_components == 'auto':
            # Johnson-Lindenstrauss: k >= 4 * log(n) / (eps^2/2 - eps^3/3)
            eps_squared = self.eps ** 2
            eps_cubed = self.eps ** 3
            self.n_components = int(4 * np.log(n_samples) / (eps_squared/2 - eps_cubed/3))
            self.n_components = min(self.n_components, n_features)
            logger.info(f"Auto-computed {self.n_components} components for {n_samples} samples")
        
        # Generate sparse random projection matrix (Achlioptas, 2003)
        # Elements are {-1, 0, +1} with probabilities {1/6, 2/3, 1/6}
        s = np.sqrt(3)
        rng = np.random.RandomState(42)
        
        # Sparse random matrix
        self.projection_matrix = np.zeros((n_features, self.n_components))
        for i in range(n_features):
            for j in range(self.n_components):
                r = rng.rand()
                if r < 1/6:
                    self.projection_matrix[i, j] = -s
                elif r < 1/3:
                    self.projection_matrix[i, j] = s
                # else: remains 0 (sparse)
        
        # Normalize
        self.projection_matrix *= np.sqrt(1.0 / self.n_components)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply random projection"""
        if self.projection_matrix is None:
            raise ValueError("RandomProjection must be fitted before transform")
        
        return X @ self.projection_matrix


class SparsePCAVintage:
    """
    Sparse PCA implementation for better interpretability
    Enforces sparsity in principal components
    """
    
    def __init__(self, n_components: int = 10, alpha: float = 1.0, max_iter: int = 100):
        self.n_components = n_components
        self.alpha = alpha  # Sparsity parameter
        self.max_iter = max_iter
        self.components_ = None
        self.mean_ = None
        
    def fit(self, X: np.ndarray) -> 'SparsePCAVintage':
        """Fit Sparse PCA using iterative thresholding"""
        logger.info(f"Fitting Sparse PCA with {self.n_components} components, alpha={self.alpha}")
        
        # Center data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        n_samples, n_features = X_centered.shape
        self.components_ = np.zeros((self.n_components, n_features))
        
        # Initialize with regular PCA
        U, s, Vt = np.linalg.svd(X_centered.T @ X_centered / n_samples, full_matrices=False)
        
        for k in range(self.n_components):
            # Initialize with PCA solution
            v = Vt[k].copy()
            
            # Iterative soft thresholding
            for _ in range(self.max_iter):
                # Compute gradient
                Xv = X_centered @ v
                gradient = X_centered.T @ Xv / n_samples
                
                # Soft thresholding
                v_new = self._soft_threshold(gradient, self.alpha)
                
                # Normalize
                norm = np.linalg.norm(v_new)
                if norm > 0:
                    v_new /= norm
                
                # Check convergence
                if np.allclose(v, v_new, rtol=1e-4):
                    break
                    
                v = v_new
            
            self.components_[k] = v
            
            # Deflate for next component
            X_centered = X_centered - np.outer(X_centered @ v, v)
        
        return self
    
    def _soft_threshold(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """Soft thresholding operator for sparsity"""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using sparse components"""
        if self.components_ is None:
            raise ValueError("SparsePCA must be fitted before transform")
        
        X_centered = X - self.mean_
        return X_centered @ self.components_.T


class IncrementalPCAVintage:
    """
    Incremental PCA for memory-efficient processing of large datasets
    Processes data in batches without loading entire dataset
    """
    
    def __init__(self, n_components: int = 10, batch_size: int = 100):
        self.n_components = n_components
        self.batch_size = batch_size
        self.components_ = None
        self.mean_ = None
        self.var_ = None
        self.n_samples_seen_ = 0
        
    def partial_fit(self, X: np.ndarray) -> 'IncrementalPCAVintage':
        """Update PCA with a batch of samples"""
        n_samples, n_features = X.shape
        
        if self.mean_ is None:
            # First batch
            self.mean_ = np.zeros(n_features)
            self.var_ = np.zeros(n_features)
            self.components_ = np.zeros((self.n_components, n_features))
        
        # Update mean incrementally
        batch_mean = np.mean(X, axis=0)
        batch_var = np.var(X, axis=0)
        
        # Welford's online algorithm for mean and variance
        last_n = self.n_samples_seen_
        self.n_samples_seen_ += n_samples
        
        if last_n == 0:
            self.mean_ = batch_mean
            self.var_ = batch_var
        else:
            last_mean = self.mean_.copy()
            self.mean_ = (last_n * self.mean_ + n_samples * batch_mean) / self.n_samples_seen_
            self.var_ = (last_n * self.var_ + n_samples * batch_var + 
                         last_n * (last_mean - self.mean_)**2 +
                         n_samples * (batch_mean - self.mean_)**2) / self.n_samples_seen_
        
        # Update components using incremental SVD
        X_centered = X - self.mean_
        
        if last_n == 0:
            # First batch: regular SVD
            U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
            self.components_ = Vt[:self.n_components]
        else:
            # Incremental update
            self._incremental_svd_update(X_centered)
        
        return self
    
    def _incremental_svd_update(self, X_batch: np.ndarray):
        """Update SVD incrementally with new batch"""
        # Simplified incremental SVD
        # Project batch onto current components
        projected = X_batch @ self.components_.T
        residual = X_batch - projected @ self.components_
        
        # Update components with residual information
        residual_norm = np.linalg.norm(residual, axis=0)
        significant_residual = residual_norm > 1e-10
        
        if np.any(significant_residual):
            # Add residual direction to components
            residual_direction = residual[:, significant_residual]
            residual_direction /= np.linalg.norm(residual_direction, axis=0)
            
            # Recompute SVD on extended basis
            extended_basis = np.vstack([self.components_, residual_direction.T])
            U, s, Vt = np.linalg.svd(extended_basis, full_matrices=False)
            self.components_ = Vt[:self.n_components]
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using incremental PCA"""
        if self.components_ is None:
            raise ValueError("IncrementalPCA must be fitted before transform")
        
        X_centered = X - self.mean_
        return X_centered @ self.components_.T


class AdaptiveDimensionalityReducer:
    """
    Adaptive dimensionality reduction that selects method based on data characteristics
    Implements the unified pipeline from the de-dimensionalization paper
    """
    
    def __init__(self, target_dims: Optional[int] = None, 
                 preserve_variance: float = 0.95,
                 sparsity_threshold: float = 0.7):
        self.target_dims = target_dims
        self.preserve_variance = preserve_variance
        self.sparsity_threshold = sparsity_threshold
        
        # Initialize all methods
        self.methods = {
            'pca': PCAVintage(n_components=20),
            'sparse_pca': SparsePCAVintage(n_components=20),
            'random_projection': RandomProjection(),
            'incremental_pca': IncrementalPCAVintage(n_components=20)
        }
        
        self.selected_method = None
        self.reducer = None
        
    def fit(self, X: np.ndarray) -> 'AdaptiveDimensionalityReducer':
        """Adaptively select and fit the best dimensionality reduction method"""
        n_samples, n_features = X.shape
        
        # Analyze data characteristics
        sparsity = 1.0 - (np.count_nonzero(X) / (n_samples * n_features))
        memory_usage = X.nbytes / (1024 ** 2)  # MB
        
        logger.info(f"Data analysis: sparsity={sparsity:.2f}, memory={memory_usage:.1f}MB")
        
        # Select method based on characteristics
        if n_samples > 10000 and memory_usage > 100:
            # Large dataset - use random projection
            self.selected_method = 'random_projection'
            if self.target_dims is None:
                self.target_dims = int(np.sqrt(n_features))
            self.reducer = RandomProjection(n_components=self.target_dims)
            
        elif sparsity > self.sparsity_threshold:
            # Sparse data - use sparse PCA
            self.selected_method = 'sparse_pca'
            if self.target_dims is None:
                self.target_dims = min(20, n_features // 2)
            self.reducer = SparsePCAVintage(n_components=self.target_dims)
            
        elif memory_usage > 50:
            # Medium-large dataset - use incremental PCA
            self.selected_method = 'incremental_pca'
            if self.target_dims is None:
                self.target_dims = self._estimate_components_for_variance(X)
            self.reducer = IncrementalPCAVintage(n_components=self.target_dims)
            
            # Fit in batches
            batch_size = min(1000, n_samples // 10)
            for i in range(0, n_samples, batch_size):
                batch = X[i:i+batch_size]
                self.reducer.partial_fit(batch)
            return self
            
        else:
            # Small dataset - use regular PCA
            self.selected_method = 'pca'
            if self.target_dims is None:
                self.target_dims = self._estimate_components_for_variance(X)
            self.reducer = PCAVintage(n_components=self.target_dims)
        
        logger.info(f"Selected method: {self.selected_method} with {self.target_dims} components")
        
        # Fit the selected method
        self.reducer.fit(X)
        
        return self
    
    def _estimate_components_for_variance(self, X: np.ndarray) -> int:
        """Estimate number of components to preserve desired variance"""
        # Quick SVD to estimate
        _, s, _ = svds(X, k=min(50, min(X.shape) - 1))
        explained_variance = s ** 2
        explained_variance_ratio = explained_variance / explained_variance.sum()
        cumsum = np.cumsum(sorted(explained_variance_ratio, reverse=True))
        
        n_components = np.argmax(cumsum >= self.preserve_variance) + 1
        return max(2, min(n_components, X.shape[1] // 2))
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform using selected method"""
        if self.reducer is None:
            raise ValueError("AdaptiveDimensionalityReducer must be fitted before transform")
        
        return self.reducer.transform(X)
    
    def get_method_info(self) -> Dict:
        """Get information about the selected method"""
        return {
            'method': self.selected_method,
            'n_components': self.target_dims,
            'reducer': self.reducer
        }


class QuantizedPCA:
    """
    PCA with quantized components for memory efficiency
    Reduces memory usage by quantizing components to int8
    """
    
    def __init__(self, n_components: int = 10, n_bits: int = 8):
        self.n_components = n_components
        self.n_bits = n_bits
        self.components_quantized = None
        self.components_scale = None
        self.components_zero = None
        self.mean_ = None
        
    def fit(self, X: np.ndarray) -> 'QuantizedPCA':
        """Fit PCA and quantize components"""
        # Regular PCA
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        U, s, Vt = np.linalg.svd(X_centered.T @ X_centered / X.shape[0], full_matrices=False)
        components = Vt[:self.n_components]
        
        # Quantize components
        self.components_quantized = []
        self.components_scale = []
        self.components_zero = []
        
        for comp in components:
            # Symmetric quantization
            max_val = np.max(np.abs(comp))
            scale = max_val / (2**(self.n_bits-1) - 1)
            
            # Quantize
            if scale > 0:
                comp_q = np.round(comp / scale).astype(np.int8)
            else:
                comp_q = np.zeros_like(comp, dtype=np.int8)
            
            self.components_quantized.append(comp_q)
            self.components_scale.append(scale)
            self.components_zero.append(0)  # Symmetric quantization
        
        self.components_quantized = np.array(self.components_quantized)
        self.components_scale = np.array(self.components_scale)
        
        memory_saved = (components.nbytes - self.components_quantized.nbytes) / components.nbytes
        logger.info(f"Quantized PCA: {memory_saved:.1%} memory saved")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform using quantized components"""
        if self.components_quantized is None:
            raise ValueError("QuantizedPCA must be fitted before transform")
        
        X_centered = X - self.mean_
        
        # Dequantize on-the-fly during transformation
        result = np.zeros((X.shape[0], self.n_components))
        for i in range(self.n_components):
            comp_dequant = self.components_quantized[i].astype(np.float32) * self.components_scale[i]
            result[:, i] = X_centered @ comp_dequant
        
        return result


# Update the main detector to use enhanced dimensionality reduction
def enhance_vintage_ml_detector(detector):
    """
    Enhance the VintageMLDefectDetector with advanced dimensionality reduction
    """
    # Replace standard PCA with adaptive reducer
    detector.dimensionality_reducer = AdaptiveDimensionalityReducer(
        target_dims=detector.pca_components,
        preserve_variance=0.95
    )
    
    # Add random projection for fast nearest neighbor search
    detector.random_projector = RandomProjection(n_components=32)
    
    # Add quantized PCA for memory-efficient storage
    detector.quantized_pca = QuantizedPCA(n_components=detector.pca_components)
    
    logger.info("Enhanced VintageMLDefectDetector with advanced dimensionality reduction")
    
    return detector
