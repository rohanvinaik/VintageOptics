# src/vintageoptics/core/hybrid_pipeline.py
"""
Hybrid Physics-ML Pipeline implementing the iterative approach
Combines vintage ML detection with physics-based correction
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from ..vintageml import VintageMLDefectDetector
from ..physics import OpticsEngine
from ..statistical import AdaptiveCleanup
from .performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass 
class HybridProcessingResult:
    """Result from hybrid physics-ML processing"""
    corrected_image: np.ndarray
    defect_masks: Dict[str, np.ndarray]
    physics_params: Dict
    ml_confidence: float
    iterations: int
    residual_error: float
    performance_metrics: Dict[str, float]
    metadata: Dict


class HybridPhysicsMLPipeline:
    """
    Implements the iterative physics <-> vintage ML approach
    as described in the skeleton pipeline
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_iterations = config.get('hybrid', {}).get('max_iterations', 3)
        self.convergence_threshold = config.get('hybrid', {}).get('convergence_threshold', 0.01)
        
        # Initialize components
        self.vintage_ml = VintageMLDefectDetector(config)
        self.physics_engine = OpticsEngine(config)
        self.cleanup_engine = AdaptiveCleanup(config)
        self.performance_monitor = PerformanceMonitor()
        
        # PAC learning parameters
        self.entropy_threshold = config.get('pac', {}).get('entropy_threshold', 2.0)
        self.confidence_delta = config.get('pac', {}).get('confidence_delta', 0.05)
        
        self.ml_trained = False
        
    def process(self, image: np.ndarray, metadata: Dict) -> HybridProcessingResult:
        """
        Main hybrid processing pipeline
        """
        logger.info("Starting hybrid physics-ML processing")
        
        with self.performance_monitor.track("hybrid_total"):
            # 1. Radiometric preprocessing
            working_image = self._preprocess_image(image)
            
            # 2. Initial physics-based correction
            with self.performance_monitor.track("initial_physics"):
                physics_params = self._estimate_initial_physics_params(working_image, metadata)
                physics_corrected = self._apply_physics_correction(working_image, physics_params)
            
            # 3. Iterative refinement loop
            best_result = physics_corrected.copy()
            best_params = physics_params.copy()
            best_error = float('inf')
            
            defect_masks = {}
            ml_confidence = 0.0
            
            for iteration in range(self.max_iterations):
                logger.info(f"Hybrid iteration {iteration + 1}/{self.max_iterations}")
                
                # 3a. Vintage ML residual analysis
                with self.performance_monitor.track(f"ml_analysis_{iteration}"):
                    ml_results = self.vintage_ml.detect_defects(physics_corrected)
                    
                    # Calculate entropy of residuals
                    residual_entropy = self._calculate_residual_entropy(
                        working_image, physics_corrected
                    )
                    
                    # PAC-based decision: should we refine physics params?
                    if residual_entropy > self.entropy_threshold:
                        logger.info(f"High entropy detected ({residual_entropy:.2f}), refining physics")
                        
                        # 3b. Physics parameter refinement
                        with self.performance_monitor.track(f"physics_refine_{iteration}"):
                            refined_params = self._refine_physics_params(
                                working_image, physics_corrected, ml_results, physics_params
                            )
                            
                            # Re-apply physics with refined params
                            physics_corrected = self._apply_physics_correction(
                                working_image, refined_params
                            )
                            physics_params = refined_params
                    
                    # 3c. Vintage ML cleanup
                    with self.performance_monitor.track(f"ml_cleanup_{iteration}"):
                        ml_cleaned = self._apply_ml_cleanup(physics_corrected, ml_results)
                    
                    # Calculate iteration error
                    error = self._calculate_error(working_image, ml_cleaned)
                    
                    if error < best_error:
                        best_result = ml_cleaned.copy()
                        best_params = physics_params.copy()
                        best_error = error
                        
                        # Update defect masks
                        for i, result in enumerate(ml_results):
                            defect_masks[f"{result.defect_type}_{i}"] = result.defect_mask
                        
                        ml_confidence = np.mean([r.confidence for r in ml_results]) if ml_results else 0.0
                    
                    # Check convergence
                    if iteration > 0 and abs(error - best_error) < self.convergence_threshold:
                        logger.info(f"Converged at iteration {iteration + 1}")
                        break
            
            # 4. Optional modern ML fallback
            if self.config.get('hybrid', {}).get('use_modern_fallback', False) and best_error > 0.1:
                with self.performance_monitor.track("modern_ml_fallback"):
                    best_result = self._apply_modern_ml_fallback(best_result, defect_masks)
            
            # 5. Final color management
            final_result = self._postprocess_image(best_result)
            
            return HybridProcessingResult(
                corrected_image=final_result,
                defect_masks=defect_masks,
                physics_params=best_params,
                ml_confidence=ml_confidence,
                iterations=iteration + 1,
                residual_error=best_error,
                performance_metrics=self.performance_monitor.get_summary(),
                metadata=metadata
            )
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Radiometric preprocessing"""
        # Convert to float32 for processing
        working = image.astype(np.float32)
        
        # Linearize if needed (simple gamma approximation)
        if self.config.get('preprocessing', {}).get('linearize', True):
            working = np.power(working / 255.0, 2.2) * 255.0
        
        return working
    
    def _postprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Final color management and conversion"""
        # Apply gamma if linearized
        if self.config.get('preprocessing', {}).get('linearize', True):
            image = np.power(image / 255.0, 1.0 / 2.2) * 255.0
        
        # Clip and convert to uint8
        return np.clip(image, 0, 255).astype(np.uint8)
    
    def _estimate_initial_physics_params(self, image: np.ndarray, metadata: Dict) -> Dict:
        """Estimate initial physics parameters from metadata"""
        params = {
            'distortion_model': 'brown_conrady',
            'k1': 0.0,
            'k2': 0.0,
            'p1': 0.0,
            'p2': 0.0,
            'chromatic_shift_r': 1.0,
            'chromatic_shift_b': 1.0,
            'vignetting_a': 0.0,
            'vignetting_b': 0.0,
            'vignetting_c': 0.0
        }
        
        # Extract from EXIF if available
        if 'LensModel' in metadata:
            lens_model = metadata['LensModel'].lower()
            
            # Load known lens parameters
            if 'helios' in lens_model:
                params.update({
                    'k1': 0.015,
                    'k2': -0.002,
                    'chromatic_shift_r': 1.01,
                    'chromatic_shift_b': 0.99,
                    'vignetting_a': 0.3
                })
            elif 'canon' in lens_model and '50' in lens_model:
                params.update({
                    'k1': -0.02,
                    'k2': 0.001,
                    'vignetting_a': 0.2
                })
        
        return params
    
    def _apply_physics_correction(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """Apply physics-based lens corrections"""
        return self.physics_engine.apply_corrections(image, params)
    
    def _calculate_residual_entropy(self, original: np.ndarray, corrected: np.ndarray) -> float:
        """
        Calculate Shannon entropy of residual errors
        High entropy indicates remaining structured errors
        """
        # Calculate residual
        residual = np.abs(original - corrected)
        
        # Convert to grayscale if needed
        if len(residual.shape) == 3:
            residual = cv2.cvtColor(residual.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Calculate histogram
        hist, _ = np.histogram(residual, bins=256, range=(0, 256))
        hist = hist.astype(float) / (hist.sum() + 1e-10)
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        return entropy
    
    def _refine_physics_params(self, original: np.ndarray, corrected: np.ndarray,
                             ml_results: List, current_params: Dict) -> Dict:
        """
        Refine physics parameters based on ML-detected residuals
        """
        refined_params = current_params.copy()
        
        # Analyze spatial distribution of defects
        if ml_results:
            # Combine all defect masks
            combined_mask = np.zeros(original.shape[:2], dtype=np.float32)
            for result in ml_results:
                combined_mask += result.defect_mask / 255.0
            
            # Analyze radial distribution (suggests distortion)
            h, w = combined_mask.shape
            cy, cx = h // 2, w // 2
            
            y_coords, x_coords = np.ogrid[:h, :w]
            distances = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
            max_dist = np.sqrt(cx**2 + cy**2)
            
            # Radial profile of defects
            radial_bins = 10
            radial_profile = []
            for i in range(radial_bins):
                r_min = i * max_dist / radial_bins
                r_max = (i + 1) * max_dist / radial_bins
                mask = (distances >= r_min) & (distances < r_max)
                radial_profile.append(np.mean(combined_mask[mask]))
            
            # If defects increase radially, increase distortion correction
            if len(radial_profile) > 1:
                radial_gradient = np.polyfit(range(len(radial_profile)), radial_profile, 1)[0]
                if radial_gradient > 0.01:
                    refined_params['k1'] *= 1.1
                    refined_params['k2'] *= 1.1
            
            # Check for chromatic patterns
            if len(original.shape) == 3:
                # Calculate channel-wise defect correlation
                for result in ml_results:
                    if 'chromatic' in result.defect_type.lower():
                        refined_params['chromatic_shift_r'] *= 1.05
                        refined_params['chromatic_shift_b'] *= 0.95
                        break
        
        return refined_params
    
    def _apply_ml_cleanup(self, image: np.ndarray, ml_results: List) -> np.ndarray:
        """Apply vintage ML-based cleanup"""
        cleaned = image.copy()
        
        # Group defects by type
        defects_by_type = {}
        for result in ml_results:
            if result.defect_type not in defects_by_type:
                defects_by_type[result.defect_type] = []
            defects_by_type[result.defect_type].append(result)
        
        # Apply appropriate cleanup for each type
        for defect_type, defects in defects_by_type.items():
            if defect_type in ['dust', 'dust_spot']:
                # Use KNN inpainting for dust
                from ..vintageml.neighbors import KNNInpainter
                inpainter = KNNInpainter(patch_size=7, n_neighbors=5)
                
                for defect in defects:
                    cleaned = inpainter.inpaint(cleaned, defect.defect_mask)
                    
            elif defect_type == 'scratch':
                # Use directional inpainting for scratches
                for defect in defects:
                    cleaned = cv2.inpaint(
                        cleaned.astype(np.uint8), 
                        defect.defect_mask,
                        inpaintRadius=3, 
                        flags=cv2.INPAINT_NS
                    ).astype(np.float32)
        
        return cleaned
    
    def _calculate_error(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate processing error metric"""
        # Simple MSE for now
        mse = np.mean((original - processed) ** 2)
        return np.sqrt(mse) / 255.0  # Normalized RMSE
    
    def _apply_modern_ml_fallback(self, image: np.ndarray, defect_masks: Dict) -> np.ndarray:
        """
        Optional modern ML fallback for difficult cases
        This would use contemporary deep learning if available
        """
        # Placeholder - would integrate modern models here
        logger.info("Modern ML fallback not implemented, returning vintage ML result")
        return image
    
    def train_vintage_ml(self, training_images: List[np.ndarray], 
                        annotations: Optional[Dict] = None):
        """Train the vintage ML components"""
        
        if not training_images:
            logger.warning("No training images provided")
            return
        
        # Generate training data
        training_data = []
        
        for i, image in enumerate(training_images):
            # If we have annotations, use them
            if annotations and str(i) in annotations:
                defect_mask = annotations[str(i)]
            else:
                # Generate synthetic defect masks for training
                defect_mask = self._generate_synthetic_defects(image)
            
            training_data.append((image, defect_mask))
        
        # Train vintage ML detector
        self.vintage_ml.train(training_data)
        self.ml_trained = True
        
        logger.info(f"Trained vintage ML on {len(training_data)} samples")
    
    def _generate_synthetic_defects(self, image: np.ndarray) -> np.ndarray:
        """Generate synthetic defect masks for training"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Add some random dust spots
        n_dust = np.random.randint(5, 20)
        for _ in range(n_dust):
            x = np.random.randint(10, w - 10)
            y = np.random.randint(10, h - 10)
            radius = np.random.randint(2, 8)
            cv2.circle(mask, (x, y), radius, 255, -1)
        
        # Add a few scratches
        n_scratches = np.random.randint(0, 3)
        for _ in range(n_scratches):
            pt1 = (np.random.randint(0, w), np.random.randint(0, h))
            pt2 = (np.random.randint(0, w), np.random.randint(0, h))
            cv2.line(mask, pt1, pt2, 255, 2)
        
        return mask
    
    def _extract_patches(self, image: np.ndarray, patch_size: int = 32) -> List[np.ndarray]:
        """Extract patches for analysis"""
        patches = []
        h, w = image.shape[:2]
        
        for y in range(0, h - patch_size, patch_size):
            for x in range(0, w - patch_size, patch_size):
                patch = image[y:y+patch_size, x:x+patch_size]
                patches.append(patch)
        
        return patches
    
    def apply_pac_learning(self, residuals: np.ndarray) -> Dict:
        """
        Apply PAC learning principles to refine detection
        Extract high-confidence rules from residual patterns
        """
        # This would implement the PAC learning approach
        # For now, return basic analysis
        return {
            'entropy': self._calculate_residual_entropy(residuals, residuals),
            'confidence': 0.95,
            'rules_extracted': 0
        }
