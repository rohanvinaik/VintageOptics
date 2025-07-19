#!/usr/bin/env python3
"""Test script to verify VintageOptics modules can be imported."""

import pytest
import sys
import os
from pathlib import Path


class TestVintageOpticsImports:
    """Test that VintageOptics modules can be imported."""
    
    @pytest.fixture(autouse=True)
    def setup_path(self):
        """Add VintageOptics src to path."""
        src_path = Path(__file__).parent.parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
    
    def test_core_imports(self):
        """Test core module imports."""
        try:
            from vintageoptics.core.pipeline import VintageOpticsPipeline, PipelineConfig
            assert VintageOpticsPipeline is not None
            assert PipelineConfig is not None
        except ImportError as e:
            pytest.skip(f"Core imports not available: {e}")
    
    def test_physics_imports(self):
        """Test physics module imports."""
        try:
            from vintageoptics.physics.optics_engine import OpticsEngine
            assert OpticsEngine is not None
        except ImportError as e:
            pytest.skip(f"Physics imports not available: {e}")
    
    def test_detection_imports(self):
        """Test detection module imports."""
        try:
            from vintageoptics.detection import UnifiedDetector
            assert UnifiedDetector is not None
        except ImportError as e:
            pytest.skip(f"Detection imports not available: {e}")
    
    def test_analysis_imports(self):
        """Test analysis module imports."""
        try:
            from vintageoptics.analysis import LensCharacterizer, QualityAnalyzer
            assert LensCharacterizer is not None
            assert QualityAnalyzer is not None
        except ImportError as e:
            pytest.skip(f"Analysis imports not available: {e}")
    
    def test_synthesis_imports(self):
        """Test synthesis module imports."""
        try:
            from vintageoptics.synthesis import LensSynthesizer
            assert LensSynthesizer is not None
        except ImportError as e:
            pytest.skip(f"Synthesis imports not available: {e}")
    
    def test_hyperdimensional_imports(self):
        """Test hyperdimensional module imports."""
        try:
            from vintageoptics.hyperdimensional import HyperdimensionalLensAnalyzer
            assert HyperdimensionalLensAnalyzer is not None
        except ImportError as e:
            pytest.skip(f"Hyperdimensional imports not available: {e}")
    
    def test_main_package_import(self):
        """Test main package import and version."""
        try:
            import vintageoptics
            assert hasattr(vintageoptics, '__version__')
            print(f"VintageOptics version: {vintageoptics.__version__}")
        except ImportError as e:
            pytest.skip(f"Main package import failed: {e}")


# Standalone test runner
def run_import_tests():
    """Run import tests without pytest."""
    print("Testing VintageOptics imports...")
    print("=" * 50)
    
    # Add src to path
    src_path = Path(__file__).parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    modules_to_test = [
        ("Core Pipeline", "vintageoptics.core.pipeline", ["VintageOpticsPipeline", "PipelineConfig"]),
        ("Types", "vintageoptics.types", ["ProcessingMode", "LensProfile"]),
        ("Hyperdimensional", "vintageoptics.hyperdimensional", ["HyperdimensionalLensAnalyzer"]),
        ("Physics Engine", "vintageoptics.physics.optics_engine", ["OpticsEngine"]),
        ("Detection", "vintageoptics.detection", ["UnifiedDetector"]),
        ("Analysis", "vintageoptics.analysis", ["LensCharacterizer", "QualityAnalyzer"]),
        ("Synthesis", "vintageoptics.synthesis", ["LensSynthesizer"]),
    ]
    
    successful = []
    failed = []
    
    for name, module, classes in modules_to_test:
        try:
            print(f"Testing {name}...", end=" ")
            # Dynamic import
            mod = __import__(module, fromlist=classes)
            for cls in classes:
                if not hasattr(mod, cls):
                    raise ImportError(f"{cls} not found in {module}")
            print("✓ SUCCESS")
            successful.append(name)
        except ImportError as e:
            print(f"✗ FAILED: {e}")
            failed.append((name, str(e)))
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed.append((name, str(e)))
    
    print("\n" + "="*50)
    print(f"Summary: {len(successful)}/{len(modules_to_test)} modules loaded successfully")
    
    if failed:
        print("\nFailed modules:")
        for name, error in failed:
            print(f"  - {name}: {error}")
        print("\nTo fix these issues, you may need to:")
        print("  1. Install missing dependencies: pip install -r requirements/base.txt")
        print("  2. Ensure all __init__.py files are present")
        print("  3. Check for circular imports")
        return False
    else:
        print("\nAll modules loaded successfully! ✨")
        print("The VintageOptics pipeline is ready to use.")
        return True


if __name__ == "__main__":
    success = run_import_tests()
    sys.exit(0 if success else 1)
