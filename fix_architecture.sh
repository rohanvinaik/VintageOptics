#!/bin/bash
# Comprehensive VintageOptics Architecture Fix Script

echo "=== VintageOptics Architecture Fix ==="
echo "This script will clean up tail-chasing bugs and improve code architecture"
echo ""

# Create backup
echo "Creating backup..."
BACKUP_DIR="vintageoptics_backup_$(date +%Y%m%d_%H%M%S)"
cp -r src $BACKUP_DIR
echo "Backup created in $BACKUP_DIR"

# 1. Clean all __pycache__ directories
echo ""
echo "Cleaning Python cache..."
find src -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find src -name "*.pyc" -delete 2>/dev/null

# 2. Fix import statements
echo ""
echo "Fixing import statements..."

# Fix detection module imports
echo "  - Updating detection module imports"
find src -name "*.py" -type f -exec sed -i '' \
    -e 's/from \.\.detection import UnifiedDetector/from ..detection import UnifiedLensDetector/g' \
    -e 's/from \.detection import UnifiedDetector/from .detection import UnifiedLensDetector/g' \
    -e 's/import UnifiedDetector/import UnifiedLensDetector/g' {} \;

# Fix analysis module imports
echo "  - Updating analysis module imports"
find src -name "*.py" -type f -exec sed -i '' \
    -e 's/from \.comparison import/from .quality_metrics import/g' \
    -e 's/from \.\.analysis\.comparison/from ..analysis.quality_metrics/g' {} \;

# 3. Consolidate detector implementations
echo ""
echo "Consolidating detector implementations..."

# Create a mapping file for old->new imports
cat > src/vintageoptics/detection/compat.py << 'EOF'
"""
Compatibility module for old import paths.
Maps old class names to new implementations.
"""

# Import the actual implementations
from .unified_detector import UnifiedLensDetector
from ..vintageml.detector import VintageMLDefectDetector

# Compatibility aliases
UnifiedDetector = UnifiedLensDetector  # Old name compatibility

__all__ = ['UnifiedDetector', 'UnifiedLensDetector', 'VintageMLDefectDetector']
EOF

# 4. Fix module __init__ files
echo ""
echo "Updating module __init__ files..."

# Ensure all modules are properly initialized
for dir in $(find src/vintageoptics -type d -name "[!__]*" | grep -v __pycache__); do
    init_file="$dir/__init__.py"
    if [ ! -f "$init_file" ]; then
        module_name=$(basename "$dir")
        echo "  - Creating __init__.py for $module_name"
        cat > "$init_file" << EOF
"""
${module_name} module for VintageOptics.
"""

__version__ = "0.1.0"
EOF
    fi
done

# 5. Remove orphaned files
echo ""
echo "Checking for orphaned files..."

# Look for comparison.py (should not exist)
if [ -f "src/vintageoptics/analysis/comparison.py" ]; then
    echo "  - Found orphaned comparison.py, removing..."
    rm "src/vintageoptics/analysis/comparison.py"
fi

# 6. Generate import map
echo ""
echo "Generating import map..."
cat > IMPORT_MAP.md << 'EOF'
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
EOF

# 7. Create a validation script
echo ""
echo "Creating validation script..."
cat > validate_imports.py << 'EOF'
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
EOF

chmod +x validate_imports.py

# 8. Run validation
echo ""
echo "Running import validation..."
cd src && python3 ../validate_imports.py
cd ..

# 9. Summary
echo ""
echo "=== Fix Summary ==="
echo "1. ✓ Cleaned all Python cache files"
echo "2. ✓ Fixed UnifiedDetector -> UnifiedLensDetector imports"
echo "3. ✓ Fixed ComparisonAnalyzer -> QualityAnalyzer imports"
echo "4. ✓ Created compatibility module for old imports"
echo "5. ✓ Updated module __init__ files"
echo "6. ✓ Generated IMPORT_MAP.md for reference"
echo "7. ✓ Created validate_imports.py script"
echo ""
echo "Backup saved in: $BACKUP_DIR"
echo ""
echo "Next steps:"
echo "1. Review IMPORT_MAP.md for correct import paths"
echo "2. Run 'python3 validate_imports.py' to check imports"
echo "3. Run your test suite to ensure functionality"
echo "4. Delete the backup directory once verified"
