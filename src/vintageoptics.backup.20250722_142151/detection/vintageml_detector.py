"""
Vintage ML-based defect detector implementing classic algorithms.
Integrates with the main detection pipeline.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from .base_detector import BaseDefectDetector
from ..vintageml import (
    DefectPerceptron,
    DefectFeatureExtractor,
    DefectClusterer
)

logger = logging.getLogger(__name__)


class VintageMLDetector(BaseDefectDetector):
    """
    Defect detector using vintage AI-winter algorithms.
    Implements transparent, explainable detection methods.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Initialize vintage ML components
        self.perceptron = DefectPerceptron()
        self.feature_extractor = DefectFeatureExtractor(
            n_pca_components=config.get('vintageml', {}).get('pca_components', 20),
            n_lda_components=config.get('vintageml', {}).get('lda_components', 3)
        )
        self.clusterer = DefectClusterer(
            k_neighbors=config.get('vintageml', {}).get('k_neighbors', 5),
            som_size=tuple(config.get('vintageml', {}).get('som_size', [15, 15]))
        )
        
        # Detection parameters
        self.patch_size = config.get('detection', {}).get('patch_size', 32)
        self.stride = config.get('detection', {}).get('stride', 16)
        self.confidence_threshold = config.get('detection', {}).get('confidence_threshold', 0.5)
        
        # Training state
        self.is_trained = False
        
    def detect(self, image: np.ndarray, lens_info: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Detect defects using vintage ML methods.
        
        Returns:
            Dict containing:
            - 'dust_mask': Binary mask for dust
            - 'scratch_mask': Binary mask for scratches
            - 'fungus_mask': Binary mask for fungus
            - 'defect_mask': Combined defect mask
            - 'confidence_map': Confidence values for each pixel
            - 'character_map': Map of lens character vs defects
        """
        height, width = image.shape[:2]
        
        # Initialize result maps
        dust_map = np.zeros((height, width), dtype=np.float32)
        scratch_map = np.zeros((height, width), dtype=np.float32)
        fungus_map = np.zeros((height, width), dtype=np.float32)
        character_map = np.zeros((height, width), dtype=np.float32)
        
        # Extract and analyze patches
        patch_results = []
        positions = []
        
        for y in range(0, height - self.patch_size, self.stride):
            for x in range(0, width - self.patch_size, self.stride):
                # Extract patch
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                
                # Convert to grayscale if needed
                if len(patch.shape) == 3:
                    patch_gray = np.mean(patch, axis=2)
                else:
                    patch_gray = patch
                
                # Analyze patch
                if self.is_trained:
                    scores = self.perceptron.detect(patch_gray)
                else:
                    scores = self._analyze_patch_heuristic(patch_gray)
                
                patch_results.append(scores)
                positions.append((y, x))
        
        # Aggregate patch results into maps
        for (y, x), scores in zip(positions, patch_results):
            # Update confidence maps with max aggregation
            dust_map[y:y+self.patch_size, x:x+self.patch_size] = np.maximum(
                dust_map[y:y+self.patch_size, x:x+self.patch_size],
                scores['dust']
            )
            scratch_map[y:y+self.patch_size, x:x+self.patch_size] = np.maximum(
                scratch_map[y:y+self.patch_size, x:x+self.patch_size],
                scores['scratch']
            )
            fungus_map[y:y+self.patch_size, x:x+self.patch_size] = np.maximum(
                fungus_map[y:y+self.patch_size, x:x+self.patch_size],
                scores['fungus']
            )
            character_map[y:y+self.patch_size, x:x+self.patch_size] = np.maximum(
                character_map[y:y+self.patch_size, x:x+self.patch_size],
                scores['character']
            )
        
        # Apply lens-specific adjustments
        if lens_info:
            dust_map, scratch_map, fungus_map = self._apply_lens_priors(
                dust_map, scratch_map, fungus_map, lens_info
            )
        
        # Create binary masks
        dust_mask = dust_map > self.confidence_threshold
        scratch_mask = scratch_map > self.confidence_threshold
        fungus_mask = fungus_map > self.confidence_threshold
        
        # Combined defect mask
        defect_mask = dust_mask | scratch_mask | fungus_mask
        
        # Overall confidence map
        confidence_map = np.maximum.reduce([dust_map, scratch_map, fungus_map])
        
        return {
            'dust_mask': dust_mask,
            'scratch_mask': scratch_mask,
            'fungus_mask': fungus_mask,
            'defect_mask': defect_mask,
            'confidence_map': confidence_map,
            'character_map': character_map,
            'dust_confidence': dust_map,
            'scratch_confidence': scratch_map,
            'fungus_confidence': fungus_map
        }
    
    def _analyze_patch_heuristic(self, patch: np.ndarray) -> Dict[str, float]:
        """
        Heuristic analysis when ML not trained.
        Uses classical image processing.
        """
        # Extract simple features
        mean_val = patch.mean()
        std_val = patch.std()
        
        # Gradient analysis
        grad_x = np.abs(np.diff(patch, axis=1)).mean()
        grad_y = np.abs(np.diff(patch, axis=0)).mean()
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Local contrast
        local_contrast = patch.max() - patch.min()
        
        # Initialize scores
        scores = {
            'dust': 0.0,
            'scratch': 0.0,
            'fungus': 0.0,
            'character': 1.0
        }
        
        # Dust: small, dark spots with low variance
        if mean_val < 0.3 and std_val < 0.1 and local_contrast < 0.2:
            scores['dust'] = 0.7
            scores['character'] = 0.3
            
        # Scratches: linear features with high gradient in one direction
        if grad_x > 2 * grad_y or grad_y > 2 * grad_x:
            scores['scratch'] = 0.6
            scores['character'] = 0.4
            
        # Fungus: irregular patches with medium contrast
        if std_val > 0.15 and local_contrast > 0.3 and local_contrast < 0.6:
            scores['fungus'] = 0.5
            scores['character'] = 0.5
            
        return scores
    
    def _apply_lens_priors(self, dust_map: np.ndarray, scratch_map: np.ndarray,
                          fungus_map: np.ndarray, lens_info: Dict) -> Tuple[np.ndarray, ...]:
        """
        Apply lens-specific priors to adjust detection confidence.
        Different vintage lenses have characteristic defect patterns.
        """
        lens_type = lens_info.get('type', 'unknown')
        age = lens_info.get('age_estimate', 0)
        
        # Older lenses more likely to have fungus
        if age > 30:
            fungus_map *= 1.2
            
        # Soviet lenses often have specific dust patterns
        if 'helios' in lens_type.lower() or 'jupiter' in lens_type.lower():
            # Central dust less likely (better sealed)
            h, w = dust_map.shape
            y_center, x_center = h // 2, w // 2
            y_coords, x_coords = np.ogrid[:h, :w]
            mask = np.sqrt((x_coords - x_center)**2 + (y_coords - y_center)**2)
            mask = mask / mask.max()
            dust_map *= (0.5 + 0.5 * mask)  # Reduce center confidence
            
        # Japanese lenses tend to have better coatings
        if any(brand in lens_type.lower() for brand in ['takumar', 'nikkor', 'canon']):
            fungus_map *= 0.8  # Less likely to have fungus
            
        return np.clip(dust_map, 0, 1), np.clip(scratch_map, 0, 1), np.clip(fungus_map, 0, 1)
    
    def train(self, training_images: List[np.ndarray], 
              annotations: Dict[str, List[np.ndarray]]):
        """
        Train the vintage ML components.
        
        Args:
            training_images: List of training images
            annotations: Dict with 'dust', 'scratch', 'fungus' mask lists
        """
        logger.info("Training Vintage ML Detector...")
        
        # Extract all patches and labels
        all_patches = []
        dust_labels = []
        scratch_labels = []
        fungus_labels = []
        
        for img_idx, image in enumerate(training_images):
            # Get corresponding masks
            dust_mask = annotations['dust'][img_idx] if img_idx < len(annotations.get('dust', [])) else None
            scratch_mask = annotations['scratch'][img_idx] if img_idx < len(annotations.get('scratch', [])) else None
            fungus_mask = annotations['fungus'][img_idx] if img_idx < len(annotations.get('fungus', [])) else None
            
            # Extract patches
            height, width = image.shape[:2]
            for y in range(0, height - self.patch_size, self.stride):
                for x in range(0, width - self.patch_size, self.stride):
                    patch = image[y:y+self.patch_size, x:x+self.patch_size]
                    
                    # Convert to grayscale
                    if len(patch.shape) == 3:
                        patch = np.mean(patch, axis=2)
                        
                    all_patches.append(patch)
                    
                    # Get labels from masks
                    if dust_mask is not None:
                        dust_label = dust_mask[y:y+self.patch_size, x:x+self.patch_size].any()
                        dust_labels.append(int(dust_label))
                    else:
                        dust_labels.append(0)
                        
                    if scratch_mask is not None:
                        scratch_label = scratch_mask[y:y+self.patch_size, x:x+self.patch_size].any()
                        scratch_labels.append(int(scratch_label))
                    else:
                        scratch_labels.append(0)
                        
                    if fungus_mask is not None:
                        fungus_label = fungus_mask[y:y+self.patch_size, x:x+self.patch_size].any()
                        fungus_labels.append(int(fungus_label))
                    else:
                        fungus_labels.append(0)
        
        # Convert to arrays
        all_patches = np.array(all_patches)
        labels = {
            'dust': np.array(dust_labels),
            'scratch': np.array(scratch_labels),
            'fungus': np.array(fungus_labels)
        }
        
        # Train perceptron
        self.perceptron.train(all_patches, labels)
        
        # Train feature extractor
        patches_flat = all_patches.reshape(len(all_patches), -1)
        
        # Create combined defect labels for LDA
        defect_labels = np.zeros(len(all_patches), dtype=int)
        defect_labels[labels['dust'] == 1] = 1
        defect_labels[labels['scratch'] == 1] = 2
        defect_labels[labels['fungus'] == 1] = 3
        
        self.feature_extractor.fit(patches_flat, defect_labels)
        
        # Train clusterer
        self.clusterer.train(all_patches)
        
        self.is_trained = True
        logger.info(f"Training complete. Processed {len(all_patches)} patches.")
    
    def analyze_detection_quality(self, detections: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Analyze the quality and confidence of detections.
        Returns metrics about the detection results.
        """
        metrics = {}
        
        # Coverage metrics
        total_pixels = detections['defect_mask'].size
        defect_pixels = detections['defect_mask'].sum()
        metrics['defect_coverage'] = defect_pixels / total_pixels
        
        # Confidence metrics
        metrics['avg_confidence'] = detections['confidence_map'].mean()
        metrics['max_confidence'] = detections['confidence_map'].max()
        
        # Type-specific metrics
        metrics['dust_coverage'] = detections['dust_mask'].sum() / total_pixels
        metrics['scratch_coverage'] = detections['scratch_mask'].sum() / total_pixels
        metrics['fungus_coverage'] = detections['fungus_mask'].sum() / total_pixels
        
        # Character preservation metric
        metrics['character_preservation'] = detections['character_map'].mean()
        
        # Detection quality assessment
        if metrics['defect_coverage'] > 0.5:
            metrics['quality_assessment'] = 'over-detection'
        elif metrics['defect_coverage'] < 0.001:
            metrics['quality_assessment'] = 'under-detection'
        elif metrics['avg_confidence'] < 0.3:
            metrics['quality_assessment'] = 'low-confidence'
        else:
            metrics['quality_assessment'] = 'good'
            
        return metrics
    
    def get_explainable_results(self) -> Dict:
        """
        Get explainable results from the vintage ML models.
        Returns interpretable information about the detection process.
        """
        results = {}
        
        if self.is_trained:
            # Get perceptron weights
            weights = self.perceptron.inspect_weights()
            
            # Format for readability
            for defect_type, weight_info in weights.items():
                results[f'{defect_type}_weights'] = {
                    'features': weight_info.feature_names,
                    'weights': weight_info.weights.tolist(),
                    'bias': weight_info.bias,
                    'training_history': weight_info.training_history[-10:]  # Last 10 epochs
                }
                
            # Get PCA components
            if self.feature_extractor.fitted:
                results['pca_variance_explained'] = (
                    self.feature_extractor.pca.explained_variance_ratio_[:5].tolist()
                )
                
            # Get SOM statistics
            if self.clusterer.som.weights is not None:
                u_matrix = self.clusterer.som.get_u_matrix()
                results['som_cluster_separation'] = {
                    'mean': float(u_matrix.mean()),
                    'std': float(u_matrix.std()),
                    'max': float(u_matrix.max())
                }
                
        return results
