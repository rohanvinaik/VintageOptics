#!/usr/bin/env python3

import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("🔧 VintageOptics MVP Quick Test")
print("=" * 40)

try:
    # Test 1: Configuration
    print("\n1. Testing Configuration...")
    from vintageoptics.core.config_manager import ConfigManager
    config = ConfigManager.load("config/default.yaml")
    print(f"✅ Config loaded with {len(config)} sections")
    
    # Test 2: Physics Engine
    print("\n2. Testing Physics Engine...")
    from vintageoptics.physics import OpticsEngine
    engine = OpticsEngine(config)
    
    # Create test image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    corrected = engine.apply_corrections(test_image, {
        'distortion_k1': 0.1,
        'chromatic_red': 1.02
    })
    print(f"✅ Physics corrections applied: {corrected.shape}")
    
    # Test 3: Quality Analysis
    print("\n3. Testing Quality Analysis...")
    from vintageoptics.analysis import QualityAnalyzer
    analyzer = QualityAnalyzer()
    
    metrics = analyzer.analyze(test_image, corrected)
    print(f"✅ Quality analysis: score = {metrics['quality_score']:.3f}")
    
    # Test 4: Pipeline Creation
    print("\n4. Testing Pipeline...")
    from vintageoptics.core.pipeline import VintageOpticsPipeline
    pipeline = VintageOpticsPipeline("config/default.yaml")
    print("✅ Pipeline created successfully")
    
    print("\n🎉 MVP Quick Test: ALL PASSED!")
    print("VintageOptics core functionality is working!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
