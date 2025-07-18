# src/vintageoptics/vintageml/trainer.py
"""
Training utilities for vintage ML models
Provides command-line interface and training pipeline
"""

import numpy as np
import cv2
import os
from typing import List, Tuple, Dict
import logging
import pickle

from .detector import VintageMLDefectDetector

logger = logging.getLogger(__name__)


class VintageMLTrainer:
    """
    Training pipeline for vintage ML models
    Handles data preparation, training, and evaluation
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.detector = VintageMLDefectDetector(config)
        
    def prepare_training_data(self, data_dir: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Load and prepare training data from directory
        Expects subdirectories: 'images' and 'masks'
        """
        logger.info(f"Loading training data from {data_dir}")
        
        images_dir = os.path.join(data_dir, 'images')
        masks_dir = os.path.join(data_dir, 'masks')
        
        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            raise ValueError(f"Expected 'images' and 'masks' subdirectories in {data_dir}")
        
        training_data = []
        
        # Load all image/mask pairs
        for img_file in sorted(os.listdir(images_dir)):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                # Load image
                img_path = os.path.join(images_dir, img_file)
                image = cv2.imread(img_path)
                if image is None:
                    logger.warning(f"Could not load image: {img_path}")
                    continue
                
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Load corresponding mask
                mask_file = os.path.splitext(img_file)[0] + '_mask.png'
                mask_path = os.path.join(masks_dir, mask_file)
                
                if not os.path.exists(mask_path):
                    # Try without _mask suffix
                    mask_file = os.path.splitext(img_file)[0] + '.png'
                    mask_path = os.path.join(masks_dir, mask_file)
                
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        training_data.append((image, mask))
                        logger.debug(f"Loaded training pair: {img_file}")
                    else:
                        logger.warning(f"Could not load mask: {mask_path}")
                else:
                    logger.warning(f"No mask found for: {img_file}")
        
        logger.info(f"Loaded {len(training_data)} training samples")
        return training_data
    
    def augment_data(self, training_data: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Simple data augmentation for better generalization
        Uses classic augmentation techniques suitable for vintage ML
        """
        augmented = []
        
        for image, mask in training_data:
            # Original
            augmented.append((image, mask))
            
            # Horizontal flip
            augmented.append((
                cv2.flip(image, 1),
                cv2.flip(mask, 1)
            ))
            
            # Brightness variations
            for factor in [0.8, 1.2]:
                bright = np.clip(image * factor, 0, 255).astype(np.uint8)
                augmented.append((bright, mask))
            
            # Slight rotation
            h, w = image.shape[:2]
            for angle in [-5, 5]:
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                rot_image = cv2.warpAffine(image, M, (w, h))
                rot_mask = cv2.warpAffine(mask, M, (w, h))
                augmented.append((rot_image, rot_mask))
        
        logger.info(f"Augmented data from {len(training_data)} to {len(augmented)} samples")
        return augmented
    
    def train(self, data_dir: str, save_path: str, augment: bool = True):
        """
        Full training pipeline
        """
        # Load data
        training_data = self.prepare_training_data(data_dir)
        
        if augment:
            training_data = self.augment_data(training_data)
        
        # Train detector
        self.detector.train(training_data)
        
        # Save models
        self.detector.save_models(save_path)
        
        # Generate training report
        report = self.evaluate(training_data[:10])  # Evaluate on subset
        self._save_report(report, os.path.splitext(save_path)[0] + '_report.txt')
    
    def evaluate(self, test_data: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
        """
        Evaluate trained models
        """
        logger.info(f"Evaluating on {len(test_data)} samples")
        
        metrics = {
            'total_samples': len(test_data),
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'method_performance': {}
        }
        
        for image, ground_truth_mask in test_data:
            # Detect defects
            results = self.detector.detect_defects(image)
            
            # Combine all detection masks
            combined_mask = np.zeros_like(ground_truth_mask)
            for result in results:
                combined_mask = cv2.bitwise_or(combined_mask, result.defect_mask)
                
                # Track per-method performance
                method = result.method_used
                if method not in metrics['method_performance']:
                    metrics['method_performance'][method] = {
                        'detections': 0,
                        'avg_confidence': 0
                    }
                metrics['method_performance'][method]['detections'] += 1
                metrics['method_performance'][method]['avg_confidence'] += result.confidence
            
            # Calculate pixel-wise metrics
            gt_binary = ground_truth_mask > 128
            pred_binary = combined_mask > 128
            
            tp = np.sum(np.logical_and(gt_binary, pred_binary))
            fp = np.sum(np.logical_and(~gt_binary, pred_binary))
            tn = np.sum(np.logical_and(~gt_binary, ~pred_binary))
            fn = np.sum(np.logical_and(gt_binary, ~pred_binary))
            
            metrics['true_positives'] += tp
            metrics['false_positives'] += fp
            metrics['true_negatives'] += tn
            metrics['false_negatives'] += fn
        
        # Calculate final metrics
        total_pixels = metrics['true_positives'] + metrics['false_positives'] + \
                      metrics['true_negatives'] + metrics['false_negatives']
        
        if total_pixels > 0:
            metrics['accuracy'] = (metrics['true_positives'] + metrics['true_negatives']) / total_pixels
            
            if metrics['true_positives'] + metrics['false_positives'] > 0:
                metrics['precision'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives'])
            else:
                metrics['precision'] = 0
            
            if metrics['true_positives'] + metrics['false_negatives'] > 0:
                metrics['recall'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives'])
            else:
                metrics['recall'] = 0
            
            if metrics['precision'] + metrics['recall'] > 0:
                metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / \
                                     (metrics['precision'] + metrics['recall'])
            else:
                metrics['f1_score'] = 0
        
        # Average method performances
        for method, perf in metrics['method_performance'].items():
            if perf['detections'] > 0:
                perf['avg_confidence'] /= perf['detections']
        
        return metrics
    
    def _save_report(self, metrics: Dict, path: str):
        """Save evaluation report"""
        with open(path, 'w') as f:
            f.write("Vintage ML Training Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Overall Metrics:\n")
            f.write(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}\n")
            f.write(f"  Precision: {metrics.get('precision', 0):.4f}\n")
            f.write(f"  Recall:    {metrics.get('recall', 0):.4f}\n")
            f.write(f"  F1 Score:  {metrics.get('f1_score', 0):.4f}\n\n")
            
            f.write("Per-Method Performance:\n")
            for method, perf in metrics.get('method_performance', {}).items():
                f.write(f"  {method}:\n")
                f.write(f"    Detections: {perf['detections']}\n")
                f.write(f"    Avg Confidence: {perf['avg_confidence']:.3f}\n")
            
            f.write("\nConfusion Matrix:\n")
            f.write(f"  True Positives:  {metrics.get('true_positives', 0):,}\n")
            f.write(f"  False Positives: {metrics.get('false_positives', 0):,}\n")
            f.write(f"  True Negatives:  {metrics.get('true_negatives', 0):,}\n")
            f.write(f"  False Negatives: {metrics.get('false_negatives', 0):,}\n")
        
        logger.info(f"Saved training report to {path}")
    
    def visualize_detections(self, image: np.ndarray, save_path: str):
        """
        Visualize detection results from all vintage ML methods
        """
        results = self.detector.detect_defects(image)
        
        # Create visualization grid
        n_results = len(results) + 1  # +1 for original
        cols = min(3, n_results)
        rows = (n_results + cols - 1) // cols
        
        fig_width = cols * 5
        fig_height = rows * 5
        
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        if n_results == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Show original
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        # Show each detection result
        for i, result in enumerate(results):
            ax = axes[i + 1]
            
            # Overlay mask on image
            overlay = image.copy()
            mask_colored = np.zeros_like(image)
            mask_colored[:, :, 0] = result.defect_mask  # Red channel
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
            
            ax.imshow(overlay)
            ax.set_title(f"{result.method_used}\n{result.defect_type} ({result.confidence:.2f})")
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_results, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved detection visualization to {save_path}")


def main():
    """Command-line interface for training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Vintage ML models for VintageOptics")
    parser.add_argument('data_dir', help='Directory containing training data')
    parser.add_argument('--output', default='vintageml_models.pkl', help='Output model file')
    parser.add_argument('--config', default='config/default.yaml', help='Configuration file')
    parser.add_argument('--no-augment', action='store_true', help='Disable data augmentation')
    parser.add_argument('--visualize', help='Visualize results on test image')
    
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = VintageMLTrainer(config)
    
    # Train models
    trainer.train(args.data_dir, args.output, augment=not args.no_augment)
    
    # Visualize if requested
    if args.visualize:
        test_image = cv2.imread(args.visualize)
        if test_image is not None:
            test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
            viz_path = os.path.splitext(args.visualize)[0] + '_vintageml_detection.png'
            trainer.visualize_detections(test_image, viz_path)


if __name__ == '__main__':
    main()
