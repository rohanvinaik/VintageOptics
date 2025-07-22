"""
Classic dimensionality reduction for defect analysis.
PCA (1901/1933) and LDA (1936) implementations.
"""
import numpy as np
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class VintagePCA:
    """
    Principal Component Analysis - Pearson (1901) / Hotelling (1933).
    Fully transparent implementation for feature extraction.
    """
    
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        
    def fit(self, X: np.ndarray):
        """
        Compute principal components via eigendecomposition.
        
        Args:
            X: Data matrix (n_samples, n_features)
        """
        # Center the data
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        n_samples = X.shape[0]
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store components
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / eigenvalues.sum()
        
        logger.info(f"PCA: {self.n_components} components explain "
                   f"{self.explained_variance_ratio_.sum():.2%} of variance")
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data onto principal components."""
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Reconstruct data from components."""
        return np.dot(X_transformed, self.components_) + self.mean_
    
    def get_component_importance(self, component_idx: int) -> np.ndarray:
        """Get feature importance for a specific component."""
        return np.abs(self.components_[component_idx])


class VintageLDA:
    """
    Linear Discriminant Analysis - Fisher (1936).
    For supervised dimensionality reduction.
    """
    
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.scalings_ = None
        self.means_ = None
        self.xbar_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Compute LDA transformation.
        
        Args:
            X: Data matrix (n_samples, n_features)
            y: Class labels
        """
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)
        
        # Compute class means
        self.means_ = np.zeros((n_classes, n_features))
        for idx, cls in enumerate(classes):
            self.means_[idx] = X[y == cls].mean(axis=0)
        
        # Overall mean
        self.xbar_ = X.mean(axis=0)
        
        # Within-class scatter matrix
        S_W = np.zeros((n_features, n_features))
        for idx, cls in enumerate(classes):
            class_samples = X[y == cls]
            class_mean = self.means_[idx]
            diff = class_samples - class_mean
            S_W += np.dot(diff.T, diff)
            
        # Between-class scatter matrix
        S_B = np.zeros((n_features, n_features))
        for idx, cls in enumerate(classes):
            n_cls = (y == cls).sum()
            mean_diff = (self.means_[idx] - self.xbar_).reshape(-1, 1)
            S_B += n_cls * np.dot(mean_diff, mean_diff.T)
            
        # Solve generalized eigenvalue problem
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(
                np.linalg.inv(S_W).dot(S_B)
            )
        except np.linalg.LinAlgError:
            # Use pseudo-inverse for singular matrices
            eigenvalues, eigenvectors = np.linalg.eigh(
                np.linalg.pinv(S_W).dot(S_B)
            )
            
        # Sort by eigenvalue
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select components
        if self.n_components is None:
            self.n_components = min(n_classes - 1, n_features)
        
        self.scalings_ = eigenvectors[:, :self.n_components]
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data onto discriminant components."""
        return np.dot(X, self.scalings_)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classify based on nearest class mean in LDA space."""
        X_lda = self.transform(X)
        
        # Transform class means
        means_lda = np.dot(self.means_, self.scalings_)
        
        # Find nearest mean
        predictions = []
        for sample in X_lda:
            distances = np.sum((means_lda - sample)**2, axis=1)
            predictions.append(distances.argmin())
            
        return np.array(predictions)


class DefectFeatureExtractor:
    """
    Combines PCA and LDA for defect characterization.
    Provides interpretable features for vintage ML pipeline.
    """
    
    def __init__(self, n_pca_components: int = 20, n_lda_components: int = 3):
        self.pca = VintagePCA(n_components=n_pca_components)
        self.lda = VintageLDA(n_components=n_lda_components)
        self.fitted = False
        
    def fit(self, patches: np.ndarray, defect_labels: Optional[np.ndarray] = None):
        """
        Fit both unsupervised (PCA) and supervised (LDA) models.
        
        Args:
            patches: Flattened image patches (n_samples, n_pixels)
            defect_labels: Optional defect type labels for LDA
        """
        # Always fit PCA
        self.pca.fit(patches)
        
        # Fit LDA if labels provided
        if defect_labels is not None:
            # Use PCA-reduced features for LDA to avoid singularity
            patches_pca = self.pca.transform(patches)
            self.lda.fit(patches_pca, defect_labels)
            
        self.fitted = True
        
    def extract_features(self, patch: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract both PCA and LDA features.
        
        Returns:
            Dict with 'pca' and optionally 'lda' features
        """
        if not self.fitted:
            raise RuntimeError("Feature extractor not fitted")
            
        features = {}
        
        # PCA features
        patch_flat = patch.flatten().reshape(1, -1)
        features['pca'] = self.pca.transform(patch_flat)[0]
        
        # LDA features if available
        if self.lda.scalings_ is not None:
            pca_features = features['pca'].reshape(1, -1)
            features['lda'] = self.lda.transform(pca_features)[0]
            
        return features
    
    def reconstruct_from_pca(self, pca_features: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """Reconstruct patch from PCA features."""
        reconstructed = self.pca.inverse_transform(pca_features.reshape(1, -1))[0]
        return reconstructed.reshape(original_shape)
    
    def get_defect_components(self) -> Dict[str, np.ndarray]:
        """
        Get the principal components most associated with defects.
        Returns components as images for visualization.
        """
        if not self.fitted:
            return {}
            
        components = {}
        
        # First few PCA components often capture defects
        for i in range(min(5, self.pca.n_components)):
            components[f'pca_{i}'] = self.pca.components_[i]
            
        # LDA components directly separate defect types
        if self.lda.scalings_ is not None:
            # LDA components are in PCA space, need to transform back
            for i in range(self.lda.n_components):
                lda_in_pca = self.lda.scalings_[:, i]
                # Pad with zeros if needed
                if len(lda_in_pca) < self.pca.n_components:
                    lda_in_pca = np.pad(lda_in_pca, 
                                       (0, self.pca.n_components - len(lda_in_pca)))
                # Transform back to pixel space
                lda_in_pixels = np.dot(lda_in_pca, self.pca.components_)
                components[f'lda_{i}'] = lda_in_pixels
                
        return components
