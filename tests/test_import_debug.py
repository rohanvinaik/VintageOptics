#!/usr/bin/env python3
"""Test imports step by step"""

import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)
print(f"Added to path: {src_path}")

# Try the problematic import
try:
    from vintageoptics.analysis.error_orthogonality import HybridErrorCorrector
    print("Success! HybridErrorCorrector imported")
except Exception as e:
    print(f"Failed to import: {e}")
    
    # Trace the issue
    print("\nDebugging import chain...")
    
    try:
        import vintageoptics
        print("✓ vintageoptics imported")
        
        # Check what triggers the error
        print("\nChecking vintageoptics.__init__.py imports...")
        
    except Exception as e2:
        print(f"✗ vintageoptics import failed: {e2}")
