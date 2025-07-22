#!/bin/bash
# Automated Tail-Chasing Bug Fixer for VintageOptics
# This script fixes common import and naming issues

echo "=== VintageOptics Tail-Chasing Bug Fixer ==="
echo "This script will fix common import and class naming issues"
echo ""

# Backup first
echo "Creating backup..."
cp -r src/vintageoptics src/vintageoptics.backup.$(date +%Y%m%d_%H%M%S)

# 1. Fix UnifiedDetector -> UnifiedLensDetector imports
echo "Fixing UnifiedDetector imports..."
find src/vintageoptics -name "*.py" -type f -exec sed -i '' 's/import UnifiedDetector/import UnifiedLensDetector/g' {} \;
find src/vintageoptics -name "*.py" -type f -exec sed -i '' 's/from .unified_detector import UnifiedDetector/from .unified_detector import UnifiedLensDetector/g' {} \;
find src/vintageoptics -name "*.py" -type f -exec sed -i '' 's/from ..detection.unified_detector import UnifiedDetector/from ..detection.unified_detector import UnifiedLensDetector/g' {} \;
find src/vintageoptics -name "*.py" -type f -exec sed -i '' 's/UnifiedDetector(/UnifiedLensDetector(/g' {} \;

# 2. Fix ComparisonAnalyzer references (redirect to quality_metrics)
echo "Fixing ComparisonAnalyzer imports..."
find src/vintageoptics -name "*.py" -type f -exec sed -i '' 's/from .comparison import ComparisonAnalyzer/from .quality_metrics import QualityAnalyzer/g' {} \;
find src/vintageoptics -name "*.py" -type f -exec sed -i '' 's/from ..analysis.comparison import ComparisonAnalyzer/from ..analysis.quality_metrics import QualityAnalyzer/g' {} \;
find src/vintageoptics -name "*.py" -type f -exec sed -i '' 's/ComparisonAnalyzer(/QualityAnalyzer(/g' {} \;

# 3. Clean up __pycache__ directories
echo "Cleaning __pycache__ directories..."
find src/vintageoptics -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# 4. Remove orphaned .pyc files
echo "Removing orphaned .pyc files..."
find src/vintageoptics -name "*.pyc" -type f -delete

# 5. Check for remaining issues
echo ""
echo "=== Remaining Issues to Check ==="

echo "Checking for UnifiedDetector references..."
grep -r "UnifiedDetector" src/vintageoptics --include="*.py" | grep -v "UnifiedLensDetector" || echo "✓ No UnifiedDetector references found"

echo ""
echo "Checking for ComparisonAnalyzer references..."
grep -r "ComparisonAnalyzer" src/vintageoptics --include="*.py" || echo "✓ No ComparisonAnalyzer references found"

echo ""
echo "Checking for missing imports..."
python3 -c "
import ast
import os

missing_imports = []
for root, dirs, files in os.walk('src/vintageoptics'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r') as f:
                    tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ImportFrom):
                            if node.module and 'comparison' in str(node.module):
                                missing_imports.append(f'{filepath}: {node.module}')
            except:
                pass

if missing_imports:
    print('Found references to missing comparison module:')
    for imp in missing_imports:
        print(f'  - {imp}')
else:
    print('✓ No references to missing modules found')
"

echo ""
echo "=== Fix Complete ==="
echo "Please run your tests to ensure everything works correctly."
echo "A backup was created in src/vintageoptics.backup.*"
