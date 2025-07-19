"""
Vintage Perceptron and Adaline implementations for defect detection.
Implements classic AI-winter era algorithms with full transparency.
"""
import numpy as np
from typing import Tuple, Optional, Dict, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PerceptronWeights:
    """Transparent weight structure for inspection."""
    weights: np.ndarray
    bias: float
    feature_names: List[str]
    training_history: List[float]
    
    def to_dict(self) -> Dict:
        """Export weights for inspection."""
        return {
            'weights': self.weights.tolist(),
            'bias': self.bias,
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }


class VintagePerceptron:
    """
    Classic Rosenblatt perceptron (1957) for binary classification.
    Used for defect vs. character discrimination.
    """
    
    def __init__(self, learning_rate: float = 0.01, max_epochs: int = 100):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = 0.0
        self.training_history = []
        self.feature_names = []
        
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Train perceptron using classic delta rule.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (0 or 1)
            feature_names: Optional names for features
        """
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # Feature names for interpretability
        if feature_names:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Training loop
        for epoch in range(self.max_epochs):
            errors = 0
            
            for i in range(n_samples):
                # Forward pass
                z = np.dot(X[i], self.weights) + self.bias
                prediction = 1 if z > 0 else 0
                
                # Update if misclassified
                error = y[i] - prediction
                if error != 0:
                    self.weights += self.learning_rate * error * X[i]
                    self.bias += self.learning_rate * error
                    errors += 1
            
            self.training_history.append(errors)
            
            # Early stopping if converged
            if errors == 0:
                logger.info(f"Perceptron converged at epoch {epoch}")
                break
                
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary prediction."""
        z = np.dot(X, self.weights) + self.bias
        return (z > 0).astype(int)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Raw decision values for soft thresholding."""
        return np.dot(X, self.weights) + self.bias
    
    def get_weights(self) -> PerceptronWeights:
        """Return interpretable weight structure."""
        return PerceptronWeights(
            weights=self.weights.copy(),
            bias=self.bias,
            feature_names=self.feature_names,
            training_history=self.training_history.copy()
        )


class Adaline:
    """
    Adaptive Linear Neuron (Widrow-Hoff, 1960).
    Uses gradient descent for smoother convergence.
    """
    
    def __init__(self, learning_rate: float = 0.001, max_epochs: int = 100):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = 0.0
        self.cost_history = []
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train using gradient descent on squared error."""
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        
        for epoch in range(self.max_epochs):
            # Forward pass
            z = self.net_input(X)
            errors = y - z
            
            # Gradient descent update
            self.weights += self.learning_rate * X.T.dot(errors) / n_samples
            self.bias += self.learning_rate * errors.mean()
            
            # Track cost
            cost = (errors**2).mean()
            self.cost_history.append(cost)
            
            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}, Cost: {cost:.4f}")
                
    def net_input(self, X: np.ndarray) -> np.ndarray:
        """Calculate net input."""
        return np.dot(X, self.weights) + self.bias
    
    def activation(self, X: np.ndarray) -> np.ndarray:
        """Linear activation (identity function)."""
        return self.net_input(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary prediction via thresholding."""
        return (self.activation(X) > 0).astype(int)


class DefectPerceptron:
    """
    Specialized perceptron for vintage lens defect detection.
    Combines multiple simple perceptrons for different defect types.
    """
    
    def __init__(self):
        self.dust_detector = VintagePerceptron(learning_rate=0.01)
        self.scratch_detector = VintagePerceptron(learning_rate=0.02)
        self.fungus_detector = Adaline(learning_rate=0.001)
        self.trained = False
        
    def extract_patch_features(self, patch: np.ndarray) -> np.ndarray:
        """
        Extract simple, interpretable features from image patch.
        Classic computer vision features, no deep learning.
        """
        features = []
        
        # Intensity statistics
        features.append(patch.mean())
        features.append(patch.std())
        features.append(patch.min())
        features.append(patch.max())
        
        # Gradient features (edge detection)
        dx = np.diff(patch, axis=1).mean()
        dy = np.diff(patch, axis=0).mean()
        features.extend([dx, dy, np.sqrt(dx**2 + dy**2)])
        
        # Texture: local binary pattern surrogate
        center = patch[patch.shape[0]//2, patch.shape[1]//2]
        lbp_sum = (patch > center).sum()
        features.append(lbp_sum)
        
        # Contrast
        features.append(patch.max() - patch.min())
        
        # Shape: aspect ratio of bounding box after thresholding
        binary = patch > patch.mean()
        if binary.any():
            rows = np.any(binary, axis=1)
            cols = np.any(binary, axis=0)
            height = rows.sum()
            width = cols.sum()
            aspect = width / (height + 1e-6)
            features.append(min(aspect, 10.0))  # Cap extreme values
        else:
            features.append(1.0)
            
        return np.array(features)
    
    def train(self, patches: List[np.ndarray], labels: Dict[str, np.ndarray]):
        """
        Train all defect detectors.
        
        Args:
            patches: List of image patches
            labels: Dict with 'dust', 'scratch', 'fungus' binary arrays
        """
        # Extract features
        X = np.array([self.extract_patch_features(p) for p in patches])
        
        # Feature names for interpretability
        feature_names = [
            'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity',
            'gradient_x', 'gradient_y', 'gradient_magnitude',
            'lbp_sum', 'contrast', 'aspect_ratio'
        ]
        
        # Train each detector
        logger.info("Training dust detector...")
        self.dust_detector.fit(X, labels['dust'], feature_names)
        
        logger.info("Training scratch detector...")
        self.scratch_detector.fit(X, labels['scratch'], feature_names)
        
        logger.info("Training fungus detector...")
        self.fungus_detector.fit(X, labels['fungus'])
        
        self.trained = True
        
    def detect(self, patch: np.ndarray) -> Dict[str, float]:
        """
        Detect defects in a patch.
        
        Returns:
            Dict with defect type -> confidence scores
        """
        if not self.trained:
            raise RuntimeError("Detectors not trained yet")
            
        features = self.extract_patch_features(patch).reshape(1, -1)
        
        # Get decision values (not just binary)
        dust_score = self.dust_detector.decision_function(features)[0]
        scratch_score = self.scratch_detector.decision_function(features)[0]
        fungus_score = self.fungus_detector.activation(features)[0]
        
        # Normalize to [0, 1] with sigmoid
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
            
        return {
            'dust': sigmoid(dust_score),
            'scratch': sigmoid(scratch_score),
            'fungus': sigmoid(fungus_score),
            'character': 1.0 - max(sigmoid(dust_score), sigmoid(scratch_score), sigmoid(fungus_score))
        }
    
    def inspect_weights(self) -> Dict[str, PerceptronWeights]:
        """Get all detector weights for analysis."""
        return {
            'dust': self.dust_detector.get_weights(),
            'scratch': self.scratch_detector.get_weights(),
            'fungus': PerceptronWeights(
                weights=self.fungus_detector.weights,
                bias=self.fungus_detector.bias,
                feature_names=[f"feature_{i}" for i in range(len(self.fungus_detector.weights))],
                training_history=self.fungus_detector.cost_history
            )
        }
