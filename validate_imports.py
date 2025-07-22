#!/usr/bin/env python3
"""Validate that all imports work correctly."""

import sys
import importlib

modules_to_test = [
    'vintageoptics',
    'vintageoptics.detection',
    'vintageoptics.analysis',
    'vintageoptics.synthesis',
    'vintageoptics.calibration',
    'vintageoptics.api',
    'vintageoptics.vintageml',
]

print("Validating VintageOptics imports...")
errors = []

for module in modules_to_test:
    try:
        importlib.import_module(module)
        print(f"✓ {module}")
    except ImportError as e:
        print(f"✗ {module}: {e}")
        errors.append((module, str(e)))

if errors:
    print(f"\n{len(errors)} import errors found!")
    sys.exit(1)
else:
    print("\nAll imports validated successfully!")
