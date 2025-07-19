#!/usr/bin/env python3

import sys
sys.path.append('src')

try:
    from vintageoptics.core.pipeline import VintageOpticsPipeline
    print("✅ Import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
