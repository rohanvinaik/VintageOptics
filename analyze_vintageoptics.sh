#!/bin/bash
# Comprehensive VintageOptics codebase analysis

echo "=== VintageOptics Codebase Analysis ==="
echo "Date: $(date)"
echo ""

# Create reports directory
mkdir -p vintageoptics_analysis

# 1. Duplicate Detection
echo "=== DUPLICATE DETECTION ==="
echo "Running custom duplicate detection..."
python detect_duplicates.py > vintageoptics_analysis/function_duplicates.txt 2>&1

# 2. Circular Dependencies
echo -e "\n=== CIRCULAR DEPENDENCIES ==="
echo "Running circular import analysis..."
python analyze_imports.py > vintageoptics_analysis/circular_imports.txt 2>&1

# 3. Placeholder Detection
echo -e "\n=== PLACEHOLDER DETECTION ==="
python detect_placeholders.py > vintageoptics_analysis/placeholders.txt 2>&1

# 4. Quick Statistics
echo -e "\n=== QUICK STATISTICS ==="
echo "Total Python files: $(find src/vintageoptics -name "*.py" | wc -l)"
echo "Total functions: $(grep -r "def " src/vintageoptics --include="*.py" | wc -l)"
echo "TODO/FIXME comments: $(grep -r -i "TODO\|FIXME" src/vintageoptics --include="*.py" | wc -l)"
echo "Pass-only functions: $(grep -r "def.*:\s*pass" src/vintageoptics --include="*.py" | wc -l)"
echo "NotImplementedError: $(grep -r "NotImplementedError" src/vintageoptics --include="*.py" | wc -l)"

# 5. Find specific patterns
echo -e "\n=== SPECIFIC PATTERN SEARCH ==="
echo "Report generation functions..."
grep -r "def.*report" src/vintageoptics --include="*.py" | grep -v "__pycache__" > vintageoptics_analysis/report_functions.txt

echo "Comparison/analysis functions..."
grep -r "def.*\(compare\|analyze\|quality\)" src/vintageoptics --include="*.py" | grep -v "__pycache__" > vintageoptics_analysis/comparison_functions.txt

echo "Detector-related classes..."
grep -r "class.*Detector" src/vintageoptics --include="*.py" | grep -v "__pycache__" > vintageoptics_analysis/detector_classes.txt

echo -e "\nAnalysis complete! Results saved in vintageoptics_analysis/"
echo ""
echo "=== SUMMARY OF KEY FINDINGS ==="
echo "1. Function duplicates: $(cat vintageoptics_analysis/function_duplicates.txt | grep "found in multiple" | wc -l)"
echo "2. Circular imports: $(cat vintageoptics_analysis/circular_imports.txt | grep "Circular:" | wc -l)"
echo "3. Placeholder functions: $(cat vintageoptics_analysis/placeholders.txt | grep "Function contains" | wc -l)"
echo ""
echo "Check the vintageoptics_analysis directory for detailed reports."
