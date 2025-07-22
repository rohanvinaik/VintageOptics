# VintageOptics Import Map

## Detection Module
- Main detector: `from vintageoptics.detection import UnifiedLensDetector`
- ML defect detector: `from vintageoptics.vintageml.detector import VintageMLDefectDetector`
- Base classes: `from vintageoptics.detection import BaseLensDetector, VintageDetector, ElectronicDetector`

## Analysis Module
- Quality analysis: `from vintageoptics.analysis import QualityAnalyzer, QualityMetrics`
- Reports: `from vintageoptics.analysis import ReportGenerator`
- Lens characterization: `from vintageoptics.analysis import LensCharacterizer`

## Core Module
- Synthesis: `from vintageoptics.synthesis import LensSynthesizer`
- Calibration: `from vintageoptics.calibration import CalibrationManager`

## API Module
- CLI: `from vintageoptics.api.cli import main`
- REST API: `from vintageoptics.api.rest_api import create_app`
- Batch processing: `from vintageoptics.api.batch_processor import BatchProcessor`

## Quick Functions
- `from vintageoptics.analysis import quick_lens_analysis, quick_quality_check`
- `from vintageoptics.detection import detect_lens`
