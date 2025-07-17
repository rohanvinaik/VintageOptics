#!/usr/bin/env python3
"""
Test script for VintageOptics imports
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("Testing VintageOptics imports...")
    
    # Test core imports
    from vintageoptics.core.pipeline import VintageOpticsPipeline, ProcessingMode, ProcessingRequest
    print("✅ Core pipeline imports successful")
    
    # Test detection imports
    from vintageoptics.detection import UnifiedLensDetector
    print("✅ Detection imports successful")
    
    # Test physics imports
    from vintageoptics.physics import OpticsEngine
    print("✅ Physics imports successful")
    
    # Test depth imports
    from vintageoptics.depth import DepthAwareProcessor
    print("✅ Depth processing imports successful")
    
    # Test synthesis imports
    from vintageoptics.synthesis import LensSynthesizer
    print("✅ Synthesis imports successful")
    
    # Test statistical imports
    from vintageoptics.statistical import AdaptiveCleanup
    print("✅ Statistical cleanup imports successful")
    
    # Test analysis imports
    from vintageoptics.analysis import QualityAnalyzer
    print("✅ Analysis imports successful")
    
    # Test main package import
    import vintageoptics
    print(f"✅ Main package import successful - Version: {vintageoptics.__version__}")
    
    # Test pipeline initialization
    pipeline = VintageOpticsPipeline("config/default.yaml")
    print("✅ Pipeline initialization successful")
    
    print("\n🎉 All imports successful! VintageOptics is ready to use.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
