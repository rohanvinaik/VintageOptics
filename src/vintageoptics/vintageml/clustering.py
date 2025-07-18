"""
Classic clustering algorithms for defect analysis.
k-NN (Fix & Hodges, 1951) and SOM (Kohonen, 1982).
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class VintageKNN:
    """
    k-Nearest Neighbors for patch matching and inpainting.
    Classic non-parametric method from Fix & Hodges (1951).
    """
    
    def __init__(self, k: int = 5, distance_metric: str = 'euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.training_data = None
        self.training_labels = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Store training data."""
        self.training_data = X.copy()
        if y is not None:
            self.training_labels = y.copy()
            
    def compute_distances(self, x: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Compute distances between query point and all training points.
        
        Args:
            x: Query point (n_features,)
            X: Training data (n_samples, n_features)
            
        Returns:
            Distance array (n_samples,)
        """
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((X - x)**2, axis=1))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(X - x), axis=1)
        elif self.distance_metric == 'cosine':
            # Cosine similarity converted to distance
            x_norm = x / (np.linalg.norm(x) + 1e-8)
            X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            similarities = np.dot(X_norm, x_norm)
            return 1 - similarities
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
            
    def find_neighbors(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors.
        
        Returns:
            (indices, distances) of k nearest neighbors
        """
        distances = self.compute_distances(x, self.training_data)
        k_nearest_idx = np.argpartition(distances, self.k)[:self.k]
        k_nearest_idx = k_nearest_idx[np.argsort(distances[k_nearest_idx])]
        
        return k_nearest_idx, distances[k_nearest_idx]
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classify based on majority vote of neighbors."""
        if self.training_labels is None:
            raise ValueError("No labels provided during fit")
            
        predictions = []
        for x in X:
            neighbor_idx, _ = self.find_neighbors(x)
            neighbor_labels = self.training_labels[neighbor_idx]
            
            # Majority vote
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            predictions.append(unique[np.argmax(counts)])
            
        return np.array(predictions)
    
    def inpaint(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint missing values using weighted average of neighbors.
        
        Args:
            x: Patch with missing values
            mask: Binary mask (1 = missing, 0 = valid)
            
        Returns:
            Inpainted patch
        """
        # Use only valid pixels for matching
        valid_mask = ~mask.astype(bool)
        x_valid = x[valid_mask]
        
        # Find neighbors based on valid pixels
        neighbor_idx, distances = self.find_neighbors(x_valid)
        
        # Weighted average (inverse distance weighting)
        weights = 1 / (distances + 1e-8)
        weights /= weights.sum()
        
        # Reconstruct from neighbors
        result = x.copy()
        for idx, weight in zip(neighbor_idx, weights):
            neighbor_patch = self.training_data[idx]
            result[mask.astype(bool)] += weight * neighbor_patch[mask.astype(bool)]
            
        return result


class KohonenSOM:
    """
    Self-Organizing Map (Kohonen, 1982).
    Unsupervised learning for defect pattern discovery.
    """
    
    def __init__(self, 
                 map_size: Tuple[int, int] = (10, 10),
                 n_features: int = None,
                 learning_rate: float = 0.5,
                 sigma: float = 1.0,
                 n_iterations: int = 1000):
        self.map_size = map_size
        self.n_features = n_features
        self.learning_rate_init = learning_rate
        self.sigma_init = sigma
        self.n_iterations = n_iterations
        self.weights = None
        
    def _initialize_weights(self, n_features: int):
        """Initialize weight vectors randomly."""
        n_neurons = self.map_size[0] * self.map_size[1]
        self.weights = np.random.randn(self.map_size[0], self.map_size[1], n_features)
        
    def _get_bmu(self, x: np.ndarray) -> Tuple[int, int]:
        """
        Find Best Matching Unit (BMU) for input vector.
        
        Returns:
            (row, col) coordinates of BMU
        """
        # Compute distances to all neurons
        distances = np.sum((self.weights - x)**2, axis=2)
        
        # Find minimum
        bmu_idx = np.unravel_index(distances.argmin(), distances.shape)
        return bmu_idx
    
    def _get_neighborhood(self, bmu: Tuple[int, int], radius: float) -> np.ndarray:
        """
        Compute neighborhood function (Gaussian) around BMU.
        
        Returns:
            Neighborhood weights array
        """
        # Create coordinate grids
        rows, cols = np.mgrid[0:self.map_size[0], 0:self.map_size[1]]
        
        # Euclidean distance from BMU
        distances = np.sqrt((rows - bmu[0])**2 + (cols - bmu[1])**2)
        
        # Gaussian neighborhood
        neighborhood = np.exp(-(distances**2) / (2 * radius**2))
        
        return neighborhood
    
    def fit(self, X: np.ndarray):
        """
        Train the SOM.
        
        Args:
            X: Training data (n_samples, n_features)
        """
        n_samples, n_features = X.shape
        
        if self.n_features is None:
            self.n_features = n_features
        elif self.n_features != n_features:
            raise ValueError(f"Expected {self.n_features} features, got {n_features}")
            
        # Initialize weights
        if self.weights is None:
            self._initialize_weights(n_features)
            
        # Training loop
        for iteration in range(self.n_iterations):
            # Decay learning rate and radius
            learning_rate = self.learning_rate_init * np.exp(-iteration / self.n_iterations)
            radius = self.sigma_init * np.exp(-iteration / self.n_iterations)
            
            # Random sample
            idx = np.random.randint(0, n_samples)
            x = X[idx]
            
            # Find BMU
            bmu = self._get_bmu(x)
            
            # Update weights
            neighborhood = self._get_neighborhood(bmu, radius)
            for i in range(self.map_size[0]):
                for j in range(self.map_size[1]):
                    self.weights[i, j] += (learning_rate * neighborhood[i, j] * 
                                         (x - self.weights[i, j]))
                    
            if iteration % 100 == 0:
                logger.debug(f"SOM iteration {iteration}, lr={learning_rate:.4f}, radius={radius:.4f}")
                
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Map input to BMU coordinates.
        
        Returns:
            Array of (row, col) BMU coordinates
        """
        bmus = []
        for x in X:
            bmu = self._get_bmu(x)
            bmus.append(bmu)
        return np.array(bmus)
    
    def quantize(self, X: np.ndarray) -> np.ndarray:
        """
        Quantize input using learned prototypes.
        
        Returns:
            Quantized vectors
        """
        quantized = []
        for x in X:
            bmu = self._get_bmu(x)
            quantized.append(self.weights[bmu])
        return np.array(quantized)
    
    def get_activation_map(self, x: np.ndarray) -> np.ndarray:
        """
        Get activation map for input vector.
        
        Returns:
            2D activation map
        """
        distances = np.sum((self.weights - x)**2, axis=2)
        # Convert distances to activations (inverse relationship)
        activations = 1 / (1 + distances)
        return activations
    
    def get_u_matrix(self) -> np.ndarray:
        """
        Compute U-Matrix (unified distance matrix) for visualization.
        Shows distances between neighboring map units.
        """
        u_matrix = np.zeros(self.map_size)
        
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                # Compute average distance to neighbors
                distances = []
                
                # Check all 8 neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.map_size[0] and 0 <= nj < self.map_size[1]:
                            dist = np.linalg.norm(self.weights[i, j] - self.weights[ni, nj])
                            distances.append(dist)
                            
                u_matrix[i, j] = np.mean(distances) if distances else 0
                
        return u_matrix


class DefectClusterer:
    """
    Combines k-NN and SOM for defect analysis and inpainting.
    """
    
    def __init__(self, 
                 k_neighbors: int = 5,
                 som_size: Tuple[int, int] = (15, 15)):
        self.knn = VintageKNN(k=k_neighbors)
        self.som = KohonenSOM(map_size=som_size)
        self.defect_prototypes = None
        
    def train(self, patches: np.ndarray, defect_masks: Optional[np.ndarray] = None):
        """
        Train both k-NN and SOM on patch data.
        
        Args:
            patches: Array of image patches
            defect_masks: Optional binary masks indicating defects
        """
        # Flatten patches for processing
        n_patches = len(patches)
        patch_shape = patches[0].shape
        patches_flat = patches.reshape(n_patches, -1)
        
        # Train k-NN for inpainting
        self.knn.fit(patches_flat)
        
        # Train SOM for pattern discovery
        self.som.fit(patches_flat)
        
        # If defect masks provided, learn defect prototypes
        if defect_masks is not None:
            self._learn_defect_prototypes(patches_flat, defect_masks)
            
    def _learn_defect_prototypes(self, patches: np.ndarray, masks: np.ndarray):
        """Learn typical defect patterns from labeled data."""
        # Group patches by defect presence
        defect_patches = patches[masks.any(axis=(1, 2))]
        clean_patches = patches[~masks.any(axis=(1, 2))]
        
        # Map to SOM
        if len(defect_patches) > 0:
            defect_bmus = self.som.transform(defect_patches)
            # Create defect frequency map
            self.defect_prototypes = np.zeros(self.som.map_size)
            for bmu in defect_bmus:
                self.defect_prototypes[bmu[0], bmu[1]] += 1
            self.defect_prototypes /= self.defect_prototypes.sum()
            
    def smooth_patch(self, patch: np.ndarray, defect_mask: np.ndarray) -> np.ndarray:
        """
        Smooth a patch using SOM quantization while preserving character.
        
        Args:
            patch: Input patch
            defect_mask: Binary mask of defects
            
        Returns:
            Smoothed patch
        """
        patch_flat = patch.flatten()
        
        # Quantize using SOM
        quantized = self.som.quantize(patch_flat.reshape(1, -1))[0]
        
        # Blend based on defect mask
        result = patch.copy()
        if defect_mask.any():
            # Only replace defect regions
            result[defect_mask] = quantized.reshape(patch.shape)[defect_mask]
            
        return result
    
    def inpaint_patch(self, patch: np.ndarray, defect_mask: np.ndarray) -> np.ndarray:
        """
        Inpaint defects using k-NN.
        
        Args:
            patch: Input patch
            defect_mask: Binary mask of defects
            
        Returns:
            Inpainted patch
        """
        patch_flat = patch.flatten()
        mask_flat = defect_mask.flatten()
        
        # Inpaint using k-NN
        inpainted_flat = self.knn.inpaint(patch_flat, mask_flat)
        
        return inpainted_flat.reshape(patch.shape)
    
    def analyze_patch(self, patch: np.ndarray) -> Dict[str, float]:
        """
        Analyze patch using both methods.
        
        Returns:
            Analysis results including defect likelihood
        """
        patch_flat = patch.flatten()
        
        # Get SOM activation
        bmu = self.som.transform(patch_flat.reshape(1, -1))[0]
        activation_map = self.som.get_activation_map(patch_flat)
        
        # Defect likelihood from prototype map
        defect_likelihood = 0.0
        if self.defect_prototypes is not None:
            defect_likelihood = self.defect_prototypes[bmu[0], bmu[1]]
            
        # k-NN density (how similar to training patches)
        _, distances = self.knn.find_neighbors(patch_flat)
        density = 1 / (1 + distances.mean())
        
        return {
            'som_bmu': bmu,
            'som_activation_max': activation_map.max(),
            'defect_likelihood': defect_likelihood,
            'knn_density': density,
            'knn_min_distance': distances.min()
        }
