"""
Enhanced lens correction example using the new constraint-based system.

This shows how the new components provide better results than traditional approaches
by respecting physical constraints and leveraging orthogonal error decomposition.
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

from vintageoptics.analysis import LensCharacterizer
from vintageoptics.constraints import (
    ConstraintSpecification,
    OrthogonalErrorAnalyzer,
    OpticalTaskGraph,
    TaskNode
)
from vintageoptics.synthesis import LensSynthesizer


class EnhancedLensProcessor:
    """
    Enhanced lens processor that uses constraint validation and error decomposition
    for superior results.
    """
    
    def __init__(self):
        self.characterizer = LensCharacterizer(
            use_hd=True,
            use_constraints=True,
            use_uncertainty=True
        )
        self.error_analyzer = OrthogonalErrorAnalyzer()
        self.constraint_spec = ConstraintSpecification()
        self.synthesizer = LensSynthesizer()
    
    def process_image(self, image_path: str, target_lens: str = None, 
                     preservation_strength: float = 0.3) -> dict:
        """
        Process an image with enhanced lens correction/synthesis.
        
        Args:
            image_path: Path to input image
            target_lens: Optional target lens to emulate
            preservation_strength: How much of original character to preserve (0-1)
            
        Returns:
            Dictionary with processed results and analysis
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Processing {Path(image_path).name}...")
        
        # Step 1: Comprehensive analysis with constraints
        print("1. Analyzing lens characteristics...")
        characteristics = self.characterizer.analyze(
            image,
            full_analysis=True,
            validate_constraints=True,
            track_uncertainty=True
        )
        
        # Step 2: Orthogonal error decomposition
        print("2. Decomposing errors...")
        error_decomposition = self.error_analyzer.decompose_errors(
            image.astype(np.float32) / 255.0,
            lens_profile={'type': 'vintage' if characteristics.lens_model else 'unknown'},
            sensor_profile=None
        )
        
        # Step 3: Constraint-aware correction
        print("3. Applying constraint-aware corrections...")
        corrected_image = self._apply_corrections(
            image,
            characteristics,
            error_decomposition,
            preservation_strength
        )
        
        # Step 4: Optional lens synthesis
        if target_lens:
            print(f"4. Synthesizing {target_lens} characteristics...")
            synthesized_image = self._synthesize_lens(
                corrected_image,
                target_lens,
                characteristics
            )
        else:
            synthesized_image = corrected_image
        
        # Step 5: Validate final result
        print("5. Validating results...")
        validation_results = self._validate_results(
            image,
            synthesized_image,
            characteristics
        )
        
        return {
            'original': image,
            'corrected': corrected_image,
            'final': synthesized_image,
            'characteristics': characteristics,
            'error_decomposition': error_decomposition,
            'validation': validation_results,
            'report': self.characterizer.generate_report(characteristics)
        }
    
    def _apply_corrections(self, image: np.ndarray, 
                         characteristics: LensCharacteristics,
                         error_decomposition: dict,
                         preservation_strength: float) -> np.ndarray:
        """Apply physics-constrained corrections."""
        
        # Start with clean signal from error decomposition
        clean_signal = error_decomposition['clean_signal']
        
        # Convert back to uint8 range
        corrected = (clean_signal * 255).astype(np.uint8)
        
        # Selectively reintroduce some analog characteristics for artistic effect
        if preservation_strength > 0:
            analog_errors = error_decomposition['analog_errors']
            
            # Preserve some vignetting (often desirable)
            if 'vignetting' in analog_errors:
                vignetting = analog_errors['vignetting']
                # Apply reduced vignetting
                preservation_factor = preservation_strength * 0.5
                corrected = corrected * (1 - vignetting * preservation_factor)
                corrected = np.clip(corrected, 0, 255).astype(np.uint8)
            
            # Preserve some bokeh characteristics
            if 'blur' in analog_errors:
                # This would require more sophisticated processing
                pass
        
        # Remove all digital errors (always undesirable)
        # Already done by using clean_signal
        
        return corrected
    
    def _synthesize_lens(self, image: np.ndarray, target_lens: str,
                        source_characteristics: LensCharacteristics) -> np.ndarray:
        """Synthesize target lens characteristics."""
        
        # Map target lens names to profiles
        lens_profiles = {
            'helios': {
                'swirly_bokeh': True,
                'vignetting_strength': 0.4,
                'contrast_reduction': 0.2,
                'warmth': 0.1
            },
            'canon_dream': {
                'bokeh_creaminess': 0.9,
                'spherical_aberration': 0.3,
                'vignetting_strength': 0.2
            },
            'leica_summicron': {
                'sharpness': 0.95,
                'micro_contrast': 0.9,
                'vignetting_strength': 0.1
            }
        }
        
        profile = lens_profiles.get(target_lens, {})
        
        # Apply synthesis with physical constraints
        synthesized = image.copy()
        
        # Example: Add vignetting respecting physical constraints
        if 'vignetting_strength' in profile:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
            max_dist = np.sqrt(center[0]**2 + center[1]**2)
            
            # Physically accurate vignetting (cos^4 law)
            angle = np.arctan(dist / max_dist)
            vignetting = np.cos(angle) ** 4
            vignetting = 1 - (1 - vignetting) * profile['vignetting_strength']
            
            synthesized = synthesized * vignetting[:, :, np.newaxis]
            synthesized = np.clip(synthesized, 0, 255).astype(np.uint8)
        
        return synthesized
    
    def _validate_results(self, original: np.ndarray, 
                         processed: np.ndarray,
                         characteristics: LensCharacteristics) -> dict:
        """Validate that processing respects constraints."""
        
        # Prepare metadata
        metadata = {
            'f_stop': characteristics.max_aperture,
            'resolution': max(original.shape[:2])
        }
        
        # Validate constraints
        constraint_results = self.constraint_spec.validate_correction(
            original.astype(np.float32) / 255.0,
            processed.astype(np.float32) / 255.0,
            metadata
        )
        
        # Calculate quality metrics
        quality_metrics = {
            'sharpness_preserved': self._measure_sharpness(processed) / 
                                 (self._measure_sharpness(original) + 1e-6),
            'color_accuracy': self._measure_color_accuracy(processed),
            'noise_reduction': self._estimate_noise_reduction(original, processed)
        }
        
        return {
            'constraints': constraint_results,
            'quality': quality_metrics
        }
    
    def _measure_sharpness(self, image: np.ndarray) -> float:
        """Measure image sharpness using Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def _measure_color_accuracy(self, image: np.ndarray) -> float:
        """Measure color balance accuracy."""
        b, g, r = cv2.split(image)
        means = [b.mean(), g.mean(), r.mean()]
        avg_mean = np.mean(means)
        deviation = np.std(means) / (avg_mean + 1e-6)
        return max(0, 1 - deviation)
    
    def _estimate_noise_reduction(self, original: np.ndarray, 
                                 processed: np.ndarray) -> float:
        """Estimate noise reduction achieved."""
        # Simple estimation using high-frequency content
        orig_high = cv2.Laplacian(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
        proc_high = cv2.Laplacian(cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
        
        orig_noise = np.std(orig_high)
        proc_noise = np.std(proc_high)
        
        reduction = (orig_noise - proc_noise) / (orig_noise + 1e-6)
        return max(0, reduction)
    
    def create_comparison_figure(self, results: dict, save_path: str):
        """Create a comprehensive comparison figure."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original
        axes[0, 0].imshow(cv2.cvtColor(results['original'], cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Error map
        error_map = results['error_decomposition']['error_map']
        if error_map.ndim == 3:
            error_map = error_map.mean(axis=2)
        axes[0, 1].imshow(error_map, cmap='hot')
        axes[0, 1].set_title('Detected Errors (Hot = High Error)')
        axes[0, 1].axis('off')
        
        # Corrected
        axes[0, 2].imshow(cv2.cvtColor(results['corrected'], cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('Constraint-Aware Correction')
        axes[0, 2].axis('off')
        
        # Final result
        axes[1, 0].imshow(cv2.cvtColor(results['final'], cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Final Result')
        axes[1, 0].axis('off')
        
        # Uncertainty map if available
        if results['characteristics'].uncertainty_estimates:
            unc = results['characteristics'].uncertainty_estimates.get('final')
            if unc:
                unc_map = unc.std.mean(axis=2) if unc.std.ndim == 3 else unc.std
                axes[1, 1].imshow(unc_map, cmap='viridis')
                axes[1, 1].set_title('Uncertainty Map')
                axes[1, 1].axis('off')
        else:
            axes[1, 1].axis('off')
        
        # Validation summary
        axes[1, 2].axis('off')
        validation_text = "Validation Results:\n\n"
        
        # Constraints
        validation_text += "Constraints:\n"
        for name, (valid, msg) in results['validation']['constraints'].items():
            status = "✓" if valid else "✗"
            validation_text += f"  {status} {name}\n"
        
        # Quality metrics
        validation_text += "\nQuality Metrics:\n"
        for metric, value in results['validation']['quality'].items():
            validation_text += f"  {metric}: {value:.2f}\n"
        
        axes[1, 2].text(0.1, 0.9, validation_text, transform=axes[1, 2].transAxes,
                       fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Enhanced Lens Processing Results', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison to {save_path}")


def process_example_image():
    """Process an example image with the enhanced system."""
    
    # Create a test image if none exists
    test_image_path = "test_vintage_photo.jpg"
    
    if not Path(test_image_path).exists():
        print("Creating test image...")
        # Create synthetic test image
        size = (800, 600, 3)
        image = np.ones(size, dtype=np.uint8) * 180
        
        # Add some content
        cv2.circle(image, (400, 300), 150, (255, 200, 150), -1)
        cv2.rectangle(image, (100, 100), (300, 250), (100, 150, 200), -1)
        cv2.ellipse(image, (600, 400), (100, 80), 45, 0, 360, (200, 100, 150), -1)
        
        # Add simulated vintage artifacts
        # Vignetting
        h, w = size[:2]
        center = (w // 2, h // 2)
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        vignetting = 1 - 0.5 * (dist / dist.max()) ** 2
        image = (image * vignetting[:, :, np.newaxis]).astype(np.uint8)
        
        # Dust and scratches
        for _ in range(10):
            x, y = np.random.randint(50, 750), np.random.randint(50, 550)
            cv2.circle(image, (x, y), np.random.randint(2, 5), (0, 0, 0), -1)
        
        # Add noise
        noise = np.random.normal(0, 10, size).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        cv2.imwrite(test_image_path, image)
        print(f"Created test image: {test_image_path}")
    
    # Process the image
    processor = EnhancedLensProcessor()
    
    # Example 1: Just correction
    print("\n=== Example 1: Constraint-Aware Correction ===")
    results = processor.process_image(test_image_path, preservation_strength=0.2)
    processor.create_comparison_figure(results, "enhanced_correction_example.png")
    
    # Print report
    print("\nAnalysis Report:")
    print(results['report'])
    
    # Example 2: Correction + Synthesis
    print("\n=== Example 2: Helios Synthesis ===")
    results = processor.process_image(
        test_image_path, 
        target_lens='helios',
        preservation_strength=0.5
    )
    processor.create_comparison_figure(results, "enhanced_synthesis_example.png")
    
    # Save detailed results
    with open("processing_report.txt", "w") as f:
        f.write("=== Enhanced Lens Processing Report ===\n\n")
        f.write(results['report'])
        f.write("\n\n=== Validation Results ===\n")
        f.write(str(results['validation']))
    
    print("\nProcessing complete! Check the output files:")
    print("  - enhanced_correction_example.png")
    print("  - enhanced_synthesis_example.png")
    print("  - processing_report.txt")


if __name__ == "__main__":
    process_example_image()
