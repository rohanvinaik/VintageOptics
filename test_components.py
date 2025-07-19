#!/usr/bin/env python3
"""Test which VintageOptics components are actually available"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'VintageOptics', 'src'))

print("Testing VintageOptics Component Availability")
print("=" * 50)

components = {
    "lens_characterizer": "vintageoptics.analysis.lens_characterizer.LensCharacterizer",
    "quality_analyzer": "vintageoptics.analysis.quality_metrics.QualityAnalyzer",
    "vintage_detector": "vintageoptics.detection.vintage_detector.VintageDetector",
    "lens_fingerprinter": "vintageoptics.detection.lens_fingerprinting.LensFingerprinter",
    "optics_engine": "vintageoptics.physics.optics_engine.OpticsEngine",
    "aberration_sim": "vintageoptics.physics.aberrations.AberrationSimulator",
    "vignetting": "vintageoptics.physics.vignetting.VignettingModel",
    "chromatic": "vintageoptics.physics.chromatic.ChromaticAberration",
    "lens_synthesizer": "vintageoptics.synthesis.lens_synthesizer.LensSynthesizer",
    "bokeh_synth": "vintageoptics.synthesis.bokeh_synthesis.BokehSynthesizer",
    "focus_map": "vintageoptics.depth.focus_map.FocusMapGenerator",
    "adaptive_cleanup": "vintageoptics.statistical.adaptive_cleanup.AdaptiveCleanup",
}

available = []
missing = []

for name, import_path in components.items():
    module_path, class_name = import_path.rsplit('.', 1)
    try:
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name, None)
        if cls:
            available.append(name)
            print(f"✓ {name}: Available")
        else:
            missing.append(name)
            print(f"✗ {name}: Class not found in module")
    except ImportError as e:
        missing.append(name)
        print(f"✗ {name}: Import failed - {e}")
    except Exception as e:
        missing.append(name)
        print(f"✗ {name}: Error - {e}")

print("\n" + "=" * 50)
print(f"Summary: {len(available)}/{len(components)} components available")
print(f"Available: {', '.join(available)}")
print(f"Missing: {', '.join(missing)}")

# Test the enhanced pipeline
print("\n" + "=" * 50)
print("Testing Enhanced Pipeline Integration:")
try:
    from enhanced_pipeline_integration import get_enhanced_pipeline
    pipeline = get_enhanced_pipeline()
    print(f"✓ Enhanced pipeline created")
    print(f"✓ Active components: {list(pipeline.components.keys())}")
except Exception as e:
    print(f"✗ Enhanced pipeline failed: {e}")
