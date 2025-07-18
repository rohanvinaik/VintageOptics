# src/vintageoptics/vintageml/neighbors.py
"""
K-Nearest Neighbors implementation (Fix & Hodges, 1951)
For patch-based defect detection and inpainting
"""

import numpy as np
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class KNNVintage:
    """
    K-Nearest Neighbors classifier/regressor
    Classic non-parametric method for defect classification
    """
    
    def __init__(self, n_neighbors: int = 5, metric: str = 'euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train = None
        self.y_train = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNVintage':
        """Store training data"""
        logger.info(f"Fitting KNN with {X.shape[0]} training samples")
        self.X_train = X.copy()
        self.y_train = y.copy()
        return self
    
    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute distances between query points and training data"""
        n_queries = X.shape[0]
        n_train = self.X_train.shape[0]
        distances = np.zeros((n_queries, n_train))
        
        if self.metric == 'euclidean':
            for i, query in enumerate(X):
                distances[i] = np.linalg.norm(self.X_train - query, axis=1)
        elif self.metric == 'manhattan':
            for i, query in enumerate(X):
                distances[i] = np.sum(np.abs(self.X_train - query), axis=1)
        elif self.metric == 'cosine':
            # Cosine similarity (convert to distance)
            for i, query in enumerate(X):
                query_norm = np.linalg.norm(query)
                if query_norm > 0:
                    similarities = np.dot(self.X_train, query) / (
                        np.linalg.norm(self.X_train, axis=1) * query_norm + 1e-10
                    )
                    distances[i] = 1 - similarities
                else:
                    distances[i] = 1.0
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        return distances
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        if self.X_train is None:
            raise ValueError("KNN must be fitted before prediction")
        
        distances = self._compute_distances(X)
        predictions = []
        
        for i in range(X.shape[0]):
            # Find k nearest neighbors
            neighbor_indices = np.argpartition(distances[i], self.n_neighbors)[:self.n_neighbors]
            neighbor_labels = self.y_train[neighbor_indices]
            
            # Vote (for classification)
            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            predictions.append(unique_labels[np.argmax(counts)])
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if self.X_train is None:
            raise ValueError("KNN must be fitted before prediction")
        
        distances = self._compute_distances(X)
        classes = np.unique(self.y_train)
        n_classes = len(classes)
        probabilities = np.zeros((X.shape[0], n_classes))
        
        for i in range(X.shape[0]):
            # Find k nearest neighbors
            neighbor_indices = np.argpartition(distances[i], self.n_neighbors)[:self.n_neighbors]
            neighbor_labels = self.y_train[neighbor_indices]
            neighbor_distances = distances[i][neighbor_indices]
            
            # Weight by inverse distance
            weights = 1.0 / (neighbor_distances + 1e-10)
            weights /= weights.sum()
            
            # Calculate weighted probabilities
            for j, cls in enumerate(classes):
                mask = neighbor_labels == cls
                probabilities[i, j] = weights[mask].sum()
        
        return probabilities
    
    def kneighbors(self, X: np.ndarray, n_neighbors: Optional[int] = None,
                   return_distance: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors"""
        if self.X_train is None:
            raise ValueError("KNN must be fitted before finding neighbors")
        
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        distances = self._compute_distances(X)
        neighbor_indices = np.zeros((X.shape[0], n_neighbors), dtype=int)
        neighbor_distances = np.zeros((X.shape[0], n_neighbors))
        
        for i in range(X.shape[0]):
            # Find k nearest neighbors
            idx = np.argpartition(distances[i], n_neighbors)[:n_neighbors]
            # Sort by distance
            sorted_idx = idx[np.argsort(distances[i][idx])]
            neighbor_indices[i] = sorted_idx
            neighbor_distances[i] = distances[i][sorted_idx]
        
        if return_distance:
            return neighbor_distances, neighbor_indices
        else:
            return neighbor_indices


class KNNInpainter:
    """
    KNN-based inpainting for defect removal
    Finds similar patches to fill in defects
    """
    
    def __init__(self, patch_size: int = 7, n_neighbors: int = 10,
                 search_radius: int = 50):
        self.patch_size = patch_size
        self.n_neighbors = n_neighbors
        self.search_radius = search_radius
        self.half_patch = patch_size // 2
        
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint masked regions using KNN patch matching
        
        Args:
            image: Input image (H, W) or (H, W, C)
            mask: Binary mask of regions to inpaint (H, W)
        """
        logger.info(f"KNN inpainting with patch size {self.patch_size}")
        
        result = image.copy()
        is_color = len(image.shape) == 3
        
        # Find pixels to inpaint
        inpaint_pixels = np.where(mask > 0)
        n_pixels = len(inpaint_pixels[0])
        
        # Process each pixel
        for idx in range(n_pixels):
            y, x = inpaint_pixels[0][idx], inpaint_pixels[1][idx]
            
            # Define search region
            y_min = max(0, y - self.search_radius)
            y_max = min(image.shape[0], y + self.search_radius)
            x_min = max(0, x - self.search_radius)
            x_max = min(image.shape[1], x + self.search_radius)
            
            # Extract patches from valid (non-masked) areas
            patches = []
            positions = []
            
            for py in range(y_min + self.half_patch, y_max - self.half_patch):
                for px in range(x_min + self.half_patch, x_max - self.half_patch):
                    # Check if patch center is valid
                    if mask[py, px] == 0:
                        # Extract patch
                        patch = self._extract_patch(image, py, px)
                        if patch is not None:
                            patches.append(patch.flatten())
                            positions.append((py, px))
            
            if len(patches) > self.n_neighbors:
                # Find current patch (with masked center)
                current_patch = self._extract_patch_with_mask(image, mask, y, x)
                
                if current_patch is not None:
                    # Compute distances to valid patches
                    patches_array = np.array(patches)
                    current_flat = current_patch.flatten()
                    
                    # Only compare non-masked pixels
                    mask_patch = self._extract_patch(mask, y, x)
                    valid_pixels = mask_patch.flatten() == 0
                    
                    if np.any(valid_pixels):
                        distances = np.array([
                            np.linalg.norm(p[valid_pixels] - current_flat[valid_pixels])
                            for p in patches_array
                        ])
                        
                        # Find k nearest patches
                        nearest_idx = np.argpartition(distances, self.n_neighbors)[:self.n_neighbors]
                        
                        # Average the center pixels of nearest patches
                        if is_color:
                            values = [image[positions[i][0], positions[i][1]] for i in nearest_idx]
                            result[y, x] = np.mean(values, axis=0)
                        else:
                            values = [image[positions[i][0], positions[i][1]] for i in nearest_idx]
                            result[y, x] = np.mean(values)
        
        return result
    
    def _extract_patch(self, image: np.ndarray, cy: int, cx: int) -> Optional[np.ndarray]:
        """Extract a patch centered at (cy, cx)"""
        y_min = cy - self.half_patch
        y_max = cy + self.half_patch + 1
        x_min = cx - self.half_patch
        x_max = cx + self.half_patch + 1
        
        if (y_min >= 0 and y_max <= image.shape[0] and 
            x_min >= 0 and x_max <= image.shape[1]):
            return image[y_min:y_max, x_min:x_max]
        return None
    
    def _extract_patch_with_mask(self, image: np.ndarray, mask: np.ndarray,
                                 cy: int, cx: int) -> Optional[np.ndarray]:
        """Extract patch but zero out masked pixels"""
        patch = self._extract_patch(image, cy, cx)
        if patch is not None:
            mask_patch = self._extract_patch(mask, cy, cx)
            if len(patch.shape) == 3:
                # Color image
                patch = patch * (1 - mask_patch[:, :, np.newaxis] / 255.0)
            else:
                # Grayscale
                patch = patch * (1 - mask_patch / 255.0)
        return patch
