#!/usr/bin/env python3
"""
VintageOptics CLI with vintage ML support.
Demonstrates the hybrid physics-ML pipeline.
"""
import argparse
import os
import sys
import json
import logging
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vintageoptics.core.pipeline import VintageOpticsPipeline, ProcessingMode, ProcessingRequest
from vintageoptics.vintageml import DefectPerceptron, VintagePCA
from vintageoptics.detection.vintageml_detector import VintageMLDetector


def setup_logging(verbose: bool):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def train_vintage_ml(args):
    """Train vintage ML models on annotated data."""
    print("Training Vintage ML models...")
    
    # Load configuration
    config_path = args.config or 'config/default.yaml'
    pipeline = VintageOpticsPipeline(config_path)
    
    # Train the models
    pipeline.train_vintage_ml(
        training_data_dir=args.training_dir,
        annotations_file=args.annotations
    )
    
    print(f"Training complete. Models trained on images from {args.training_dir}")
    
    # Save training report if requested
    if args.save_report:
        report = pipeline.hybrid_pipeline.defect_detector.get_explainable_results()
        report_path = args.save_report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Training report saved to {report_path}")


def process_image(args):
    """Process a single image."""
    # Load configuration
    config_path = args.config or 'config/default.yaml'
    pipeline = VintageOpticsPipeline(config_path)
    
    # Determine processing mode
    if args.mode == 'correct':
        mode = ProcessingMode.CORRECT
    elif args.mode == 'synthesize':
        mode = ProcessingMode.SYNTHESIZE
    else:
        mode = ProcessingMode.HYBRID
    
    # Create processing request
    request = ProcessingRequest(
        image_path=args.input,
        mode=mode,
        output_path=args.output,
        source_lens=args.source_lens,
        target_lens=args.target_lens,
        preserve_metadata=not args.no_metadata,
        use_depth=not args.no_depth,
        gpu_acceleration=args.gpu
    )
    
    print(f"Processing {args.input} in {mode.value} mode...")
    
    # Process the image
    result = pipeline.process(request)
    
    # Display results
    if hasattr(result, 'lens_info'):
        print(f"Detected lens: {result.lens_info.get('name', 'Unknown')}")
        
    if hasattr(result, 'ml_confidence'):
        print("\nVintage ML Detection Confidence:")
        for defect_type, confidence in result.ml_confidence.items():
            print(f"  {defect_type}: {confidence:.2%}")
            
    if hasattr(result, 'iterations'):
        print(f"\nHybrid processing completed in {result.iterations} iterations")
        print(f"Converged: {result.converged}")
        
    if hasattr(result, 'performance_metrics'):
        print("\nPerformance Metrics:")
        for metric, value in result.performance_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}s")
                
    print(f"\nOutput saved to {args.output}")


def batch_process(args):
    """Process multiple images in a directory."""
    # Load configuration
    config_path = args.config or 'config/default.yaml'
    pipeline = VintageOpticsPipeline(config_path)
    
    # Determine processing mode
    mode_map = {
        'correct': ProcessingMode.CORRECT,
        'synthesize': ProcessingMode.SYNTHESIZE,
        'hybrid': ProcessingMode.HYBRID
    }
    mode = mode_map.get(args.mode, ProcessingMode.HYBRID)
    
    print(f"Batch processing {args.input_dir} in {mode.value} mode...")
    
    # Run batch processing
    batch_result = pipeline.batch_process(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mode=mode,
        preserve_metadata=not args.no_metadata,
        use_depth=not args.no_depth,
        gpu_acceleration=args.gpu
    )
    
    # Display summary
    report = batch_result.report
    print(f"\nBatch processing complete:")
    print(f"  Total processed: {report['total_processed']}")
    print(f"  Successful: {report['successful']}")
    
    if report.get('detected_lenses'):
        print("\nDetected lenses:")
        for lens, count in report['detected_lenses'].items():
            print(f"  {lens}: {count} images")
            
    print(f"\nFull report saved to {args.output_dir}/batch_report.json")


def analyze_vintage_ml(args):
    """Analyze and explain vintage ML model behavior."""
    print("Analyzing Vintage ML models...")
    
    # Create a simple perceptron for demonstration
    perceptron = DefectPerceptron()
    pca = VintagePCA(n_components=10)
    
    # Generate synthetic data for analysis
    import numpy as np
    
    # Create sample patches with known patterns
    n_samples = 100
    patch_size = 32
    
    # Clean patches (gaussian noise)
    clean_patches = np.random.randn(n_samples//2, patch_size, patch_size) * 0.1 + 0.5
    
    # Defect patches (with artifacts)
    defect_patches = np.random.randn(n_samples//2, patch_size, patch_size) * 0.2 + 0.5
    # Add dust spots
    for i in range(n_samples//2):
        n_spots = np.random.randint(1, 5)
        for _ in range(n_spots):
            y, x = np.random.randint(0, patch_size-3, size=2)
            defect_patches[i, y:y+3, x:x+3] *= 0.3
    
    all_patches = np.vstack([clean_patches, defect_patches])
    labels = {
        'dust': np.concatenate([np.zeros(n_samples//2), np.ones(n_samples//2)]),
        'scratch': np.zeros(n_samples),
        'fungus': np.zeros(n_samples)
    }
    
    # Train perceptron
    print("Training perceptron on synthetic data...")
    perceptron.train(list(all_patches), labels)
    
    # Get and display weights
    weights = perceptron.inspect_weights()
    
    print("\nPerceptron Analysis:")
    print("=" * 50)
    
    for defect_type, weight_info in weights.items():
        print(f"\n{defect_type.upper()} Detector:")
        print(f"  Bias: {weight_info.bias:.4f}")
        print(f"  Top 3 important features:")
        
        # Get top features by absolute weight
        feature_importance = list(zip(weight_info.feature_names, 
                                    np.abs(weight_info.weights)))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        for feat_name, importance in feature_importance[:3]:
            print(f"    {feat_name}: {importance:.4f}")
            
        if weight_info.training_history:
            print(f"  Training converged in {len(weight_info.training_history)} epochs")
            print(f"  Final error rate: {weight_info.training_history[-1]}")
    
    # PCA analysis
    print("\n\nPCA Analysis:")
    print("=" * 50)
    
    patches_flat = all_patches.reshape(n_samples, -1)
    pca.fit(patches_flat)
    
    print(f"Explained variance by component:")
    for i, var_ratio in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var_ratio:.2%}")
        if i >= 4:  # Show first 5
            break
            
    total_var = pca.explained_variance_ratio_.sum()
    print(f"\nTotal variance explained by {pca.n_components} components: {total_var:.2%}")
    
    # Save detailed report if requested
    if args.save_report:
        report = {
            'perceptron_weights': {
                k: v.to_dict() for k, v in weights.items()
            },
            'pca_analysis': {
                'n_components': pca.n_components,
                'explained_variance': pca.explained_variance_.tolist(),
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist()
            }
        }
        
        with open(args.save_report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed report saved to {args.save_report}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='VintageOptics - Character-preserving lens correction with Vintage ML'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process a single image')
    process_parser.add_argument('input', help='Input image path')
    process_parser.add_argument('output', help='Output image path')
    process_parser.add_argument('--mode', choices=['correct', 'synthesize', 'hybrid'],
                              default='hybrid', help='Processing mode')
    process_parser.add_argument('--source-lens', help='Source lens for synthesis')
    process_parser.add_argument('--target-lens', help='Target lens for synthesis')
    process_parser.add_argument('--config', help='Configuration file path')
    process_parser.add_argument('--no-metadata', action='store_true',
                              help='Do not preserve metadata')
    process_parser.add_argument('--no-depth', action='store_true',
                              help='Disable depth-aware processing')
    process_parser.add_argument('--gpu', action='store_true',
                              help='Enable GPU acceleration')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Process multiple images')
    batch_parser.add_argument('input_dir', help='Input directory')
    batch_parser.add_argument('output_dir', help='Output directory')
    batch_parser.add_argument('--mode', choices=['correct', 'synthesize', 'hybrid'],
                            default='correct', help='Processing mode')
    batch_parser.add_argument('--config', help='Configuration file path')
    batch_parser.add_argument('--no-metadata', action='store_true',
                            help='Do not preserve metadata')
    batch_parser.add_argument('--no-depth', action='store_true',
                            help='Disable depth-aware processing')
    batch_parser.add_argument('--gpu', action='store_true',
                            help='Enable GPU acceleration')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train vintage ML models')
    train_parser.add_argument('training_dir', help='Directory with training images')
    train_parser.add_argument('--annotations', help='Annotations JSON file')
    train_parser.add_argument('--config', help='Configuration file path')
    train_parser.add_argument('--save-report', help='Save training report to file')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', 
                                         help='Analyze vintage ML models')
    analyze_parser.add_argument('--save-report', help='Save analysis report to file')
    
    # Global options
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Execute command
    if args.command == 'process':
        process_image(args)
    elif args.command == 'batch':
        batch_process(args)
    elif args.command == 'train':
        train_vintage_ml(args)
    elif args.command == 'analyze':
        analyze_vintage_ml(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
