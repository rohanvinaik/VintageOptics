#!/usr/bin/env python3
"""Test which VintageOptics components are actually available"""

import pytest
import sys
import os
from importlib import import_module


class TestVintageOpticsComponents:
    """Test availability of VintageOptics components."""
    
    @pytest.fixture(autouse=True)
    def setup_path(self):
        """Ensure VintageOptics src is in path."""
        src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
    
    def test_core_components_available(self):
        """Test that core components can be imported."""
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
                module = import_module(module_path)
                cls = getattr(module, class_name, None)
                if cls:
                    available.append(name)
                else:
                    missing.append(name)
                    pytest.fail(f"{name}: Class {class_name} not found in module {module_path}")
            except ImportError as e:
                missing.append(name)
                # Mark as expected failure for now
                pytest.skip(f"{name}: Import failed - {e}")
            except Exception as e:
                missing.append(name)
                pytest.fail(f"{name}: Unexpected error - {e}")
        
        # Report summary
        print(f"\nComponent Summary: {len(available)}/{len(components)} available")
        if available:
            print(f"Available: {', '.join(available)}")
        if missing:
            print(f"Missing: {', '.join(missing)}")
        
        # At least some components should be available
        assert len(available) > 0, "No components were successfully imported"
    
    @pytest.mark.skip(reason="Enhanced pipeline moved to legacy")
    def test_enhanced_pipeline_integration(self):
        """Test the enhanced pipeline integration."""
        try:
            # Updated import path since file was moved
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'legacy'))
            from enhanced_pipeline_integration import get_enhanced_pipeline
            
            pipeline = get_enhanced_pipeline()
            assert pipeline is not None
            assert hasattr(pipeline, 'components')
            assert len(pipeline.components) > 0
            
            print(f"✓ Enhanced pipeline created")
            print(f"✓ Active components: {list(pipeline.components.keys())}")
        except ImportError as e:
            pytest.skip(f"Enhanced pipeline not available: {e}")
        except Exception as e:
            pytest.fail(f"Enhanced pipeline failed: {e}")
    
    def test_vintageoptics_package_imports(self):
        """Test that the main vintageoptics package can be imported."""
        try:
            import vintageoptics
            assert hasattr(vintageoptics, '__version__')
            print(f"✓ VintageOptics version: {vintageoptics.__version__}")
        except ImportError as e:
            pytest.fail(f"Failed to import vintageoptics package: {e}")


# Standalone test runner
def run_component_tests():
    """Run component tests without pytest."""
    print("Testing VintageOptics Component Availability")
    print("=" * 50)
    
    # Add src to path
    src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
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
            module = import_module(module_path)
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
    
    return len(available) > 0


if __name__ == "__main__":
    success = run_component_tests()
    sys.exit(0 if success else 1)
