# src/vintageoptics/analysis/__init__.py

"""
Image quality analysis and metrics
"""

from .quality_metrics import QualityAnalyzer, QualityMetrics
from .lens_characterizer import LensCharacterizer, LensCharacteristics
from .comparison import ComparisonAnalyzer
from .reports import ReportGenerator

# Quick access functions
def quick_lens_analysis(image_path):
    """Quick lens characteristic analysis."""
    analyzer = LensCharacterizer()
    return analyzer.analyze(image_path)

def quick_quality_check(original, processed):
    """Quick quality comparison between images."""
    analyzer = QualityAnalyzer()
    return analyzer.analyze(original, processed)

def detailed_quality_report(original, processed, output_path=None):
    """Generate detailed quality report."""
    generator = ReportGenerator()
    return generator.create_report(original, processed, output_path)

__all__ = [
    'QualityAnalyzer',
    'QualityMetrics',
    'LensCharacterizer',
    'LensCharacteristics',
    'ComparisonAnalyzer',
    'ReportGenerator',
    'quick_lens_analysis',
    'quick_quality_check',
    'detailed_quality_report'
]
