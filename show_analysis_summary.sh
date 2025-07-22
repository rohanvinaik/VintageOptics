#!/bin/bash
# Quick summary of analysis results

echo "=== VINTAGEOPTICS ANALYSIS SUMMARY ==="
echo ""

if [ -d "vintageoptics_analysis" ]; then
    echo "Files generated:"
    ls -la vintageoptics_analysis/
    echo ""
    
    for file in vintageoptics_analysis/*.txt; do
        if [ -f "$file" ]; then
            echo "=== $(basename $file) ==="
            head -n 20 "$file"
            echo "..."
            echo ""
        fi
    done
else
    echo "Analysis directory not found. Running basic checks..."
    
    echo ""
    echo "=== DUPLICATE FUNCTIONS ==="
    python detect_duplicates.py 2>/dev/null | head -20
    
    echo ""
    echo "=== CIRCULAR IMPORTS ==="
    python analyze_imports.py 2>/dev/null | head -20
    
    echo ""
    echo "=== PLACEHOLDERS ==="
    python detect_placeholders.py 2>/dev/null | head -20
fi
