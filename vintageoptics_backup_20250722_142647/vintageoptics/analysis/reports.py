"""
Report generation module for VintageOptics analysis results.
"""

import numpy as np
from typing import Dict, Optional, Union, List
from pathlib import Path
import json
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate analysis reports in various formats.
    """
    
    def __init__(self):
        self.report_data = {}
        
    def create_report(self, 
                     original: np.ndarray,
                     processed: np.ndarray,
                     output_path: Optional[Union[str, Path]] = None,
                     metadata: Optional[Dict] = None) -> Dict:
        """
        Create a comprehensive analysis report.
        
        Args:
            original: Original image
            processed: Processed image
            output_path: Optional path to save report
            metadata: Additional metadata
            
        Returns:
            Report data dictionary
        """
        # Import here to avoid circular dependencies
        from . import QualityAnalyzer
        
        # Analyze both images
        analyzer = QualityAnalyzer()
        original_metrics = analyzer.analyze(original, compute_maps=True)
        processed_metrics = analyzer.analyze(processed, compute_maps=True)
        
        # Compare
        comparison = analyzer.compare_quality(original, processed)
        
        # Build report data
        self.report_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'original_metrics': self._metrics_to_dict(original_metrics),
            'processed_metrics': self._metrics_to_dict(processed_metrics),
            'comparison': comparison,
            'metadata': metadata or {}
        }
        
        # Generate visualizations if output path provided
        if output_path:
            output_path = Path(output_path)
            
            if output_path.suffix == '.pdf':
                self._generate_pdf_report(original, processed, output_path)
            elif output_path.suffix == '.json':
                self._save_json_report(output_path)
            elif output_path.suffix in ['.png', '.jpg']:
                self._generate_image_report(original, processed, output_path)
            else:
                # Default to PDF
                pdf_path = output_path.with_suffix('.pdf')
                self._generate_pdf_report(original, processed, pdf_path)
        
        return self.report_data
    
    def _metrics_to_dict(self, metrics) -> Dict:
        """Convert metrics object to dictionary."""
        return {
            'overall_quality': float(metrics.overall_quality),
            'sharpness': float(metrics.sharpness),
            'contrast': float(metrics.contrast),
            'noise_level': float(metrics.noise_level),
            'dynamic_range': float(metrics.dynamic_range),
            'color_accuracy': float(metrics.color_accuracy),
            'local_contrast': float(metrics.local_contrast),
            'edge_sharpness': float(metrics.edge_sharpness),
            'texture_detail': float(metrics.texture_detail),
            'hd_quality_score': float(metrics.hd_quality_score),
            'defect_impact': float(metrics.defect_impact),
            'perceptual_quality': float(metrics.perceptual_quality),
            'aesthetic_score': float(metrics.aesthetic_score)
        }
    
    def _generate_pdf_report(self, original: np.ndarray, processed: np.ndarray, 
                           output_path: Path):
        """Generate PDF report with visualizations."""
        with PdfPages(str(output_path)) as pdf:
            # Page 1: Overview
            fig = plt.figure(figsize=(11, 8.5))
            gs = gridspec.GridSpec(3, 3, figure=fig)
            
            # Title
            fig.suptitle('VintageOptics Analysis Report', fontsize=16, fontweight='bold')
            
            # Original image
            ax1 = fig.add_subplot(gs[0:2, 0])
            ax1.imshow(original[..., ::-1] if original.shape[-1] == 3 else original, cmap='gray')
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # Processed image
            ax2 = fig.add_subplot(gs[0:2, 1])
            ax2.imshow(processed[..., ::-1] if processed.shape[-1] == 3 else processed, cmap='gray')
            ax2.set_title('Processed Image')
            ax2.axis('off')
            
            # Difference
            ax3 = fig.add_subplot(gs[0:2, 2])
            diff = np.abs(original.astype(float) - processed.astype(float))
            if len(diff.shape) == 3:
                diff = np.mean(diff, axis=2)
            im = ax3.imshow(diff, cmap='hot')
            ax3.set_title('Difference Map')
            ax3.axis('off')
            plt.colorbar(im, ax=ax3)
            
            # Metrics summary
            ax4 = fig.add_subplot(gs[2, :])
            ax4.axis('off')
            
            summary_text = self._generate_summary_text()
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 2: Detailed metrics
            fig = self._create_metrics_page()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 3: Quality maps (if available)
            if hasattr(self, '_create_quality_maps_page'):
                fig = self._create_quality_maps_page(original, processed)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
    
    def _generate_image_report(self, original: np.ndarray, processed: np.ndarray,
                             output_path: Path):
        """Generate single image report."""
        fig = plt.figure(figsize=(15, 10))
        
        # Create grid
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Images
        ax1 = fig.add_subplot(gs[0:2, 0])
        ax1.imshow(original[..., ::-1] if original.shape[-1] == 3 else original, cmap='gray')
        ax1.set_title('Original', fontsize=12)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0:2, 1])
        ax2.imshow(processed[..., ::-1] if processed.shape[-1] == 3 else processed, cmap='gray')
        ax2.set_title('Processed', fontsize=12)
        ax2.axis('off')
        
        # Metrics comparison
        ax3 = fig.add_subplot(gs[0:2, 2:])
        self._plot_metrics_comparison(ax3)
        
        # Summary text
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        summary_text = self._generate_summary_text()
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('VintageOptics Analysis Report', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_json_report(self, output_path: Path):
        """Save report data as JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.report_data, f, indent=2)
        
        logger.info(f"Saved JSON report to {output_path}")
    
    def _generate_summary_text(self) -> str:
        """Generate summary text for report."""
        orig = self.report_data['original_metrics']
        proc = self.report_data['processed_metrics']
        comp = self.report_data['comparison']
        
        summary = []
        summary.append("QUALITY ANALYSIS SUMMARY")
        summary.append("=" * 50)
        summary.append(f"Timestamp: {self.report_data['timestamp']}")
        summary.append("")
        
        summary.append("Overall Quality:")
        summary.append(f"  Original:  {orig['overall_quality']:.1%}")
        summary.append(f"  Processed: {proc['overall_quality']:.1%}")
        summary.append(f"  Change:    {comp['overall_diff']:+.1%}")
        summary.append("")
        
        summary.append("Key Metrics Improvement:")
        summary.append(f"  Sharpness:  {comp['sharpness_diff']:+.1%}")
        summary.append(f"  Contrast:   {comp['contrast_diff']:+.1%}")
        summary.append(f"  Noise:      {comp['noise_diff']:+.1%}")
        summary.append(f"  Perceptual: {comp['perceptual_diff']:+.1%}")
        summary.append("")
        
        if comp['overall_diff'] > 0:
            summary.append(f"✓ Image quality improved by {comp['improvement_percent']:.1f}%")
        else:
            summary.append(f"✗ Image quality decreased by {comp['improvement_percent']:.1f}%")
        
        return "\n".join(summary)
    
    def _create_metrics_page(self) -> plt.Figure:
        """Create detailed metrics comparison page."""
        fig = plt.figure(figsize=(11, 8.5))
        
        # Radar chart
        ax1 = fig.add_subplot(121, projection='polar')
        self._plot_radar_chart(ax1)
        
        # Bar chart
        ax2 = fig.add_subplot(122)
        self._plot_metrics_comparison(ax2)
        
        fig.suptitle('Detailed Quality Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def _plot_radar_chart(self, ax):
        """Plot radar chart of metrics."""
        metrics = ['Sharpness', 'Contrast', 'Noise', 'Color', 'Texture', 'Perceptual']
        
        orig = self.report_data['original_metrics']
        proc = self.report_data['processed_metrics']
        
        orig_values = [
            orig['sharpness'],
            orig['contrast'],
            orig['noise_level'],
            orig['color_accuracy'],
            orig['texture_detail'],
            orig['perceptual_quality']
        ]
        
        proc_values = [
            proc['sharpness'],
            proc['contrast'],
            proc['noise_level'],
            proc['color_accuracy'],
            proc['texture_detail'],
            proc['perceptual_quality']
        ]
        
        # Number of variables
        num_vars = len(metrics)
        
        # Compute angle of each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
        # Complete the circle
        orig_values += orig_values[:1]
        proc_values += proc_values[:1]
        angles += angles[:1]
        
        # Plot
        ax.plot(angles, orig_values, 'o-', linewidth=2, label='Original', color='blue')
        ax.fill(angles, orig_values, alpha=0.25, color='blue')
        
        ax.plot(angles, proc_values, 'o-', linewidth=2, label='Processed', color='green')
        ax.fill(angles, proc_values, alpha=0.25, color='green')
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'])
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        ax.set_title('Quality Metrics Comparison', y=1.08)
    
    def _plot_metrics_comparison(self, ax):
        """Plot bar chart comparison of metrics."""
        metrics = ['Overall', 'Sharp', 'Contrast', 'Noise', 'Color', 'Texture', 'Percept']
        
        orig = self.report_data['original_metrics']
        proc = self.report_data['processed_metrics']
        
        orig_values = [
            orig['overall_quality'],
            orig['sharpness'],
            orig['contrast'],
            orig['noise_level'],
            orig['color_accuracy'],
            orig['texture_detail'],
            orig['perceptual_quality']
        ]
        
        proc_values = [
            proc['overall_quality'],
            proc['sharpness'],
            proc['contrast'],
            proc['noise_level'],
            proc['color_accuracy'],
            proc['texture_detail'],
            proc['perceptual_quality']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, orig_values, width, label='Original', color='blue', alpha=0.8)
        bars2 = ax.bar(x + width/2, proc_values, width, label='Processed', color='green', alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('Score')
        ax.set_title('Quality Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
    
    def generate_html_report(self, 
                           original: np.ndarray,
                           processed: np.ndarray,
                           output_path: Union[str, Path]) -> None:
        """Generate interactive HTML report."""
        # This would generate an interactive HTML report
        # For now, just create a simple HTML with embedded images
        pass
