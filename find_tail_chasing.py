#!/usr/bin/env python3
import os
import re
from collections import defaultdict

def find_tail_chasing_bugs():
    base_path = '/Users/rohanvinaik/VintageOptics/src/vintageoptics'
    
    # Pattern matches for common tail-chasing issues
    patterns = {
        'unified_detector': [],
        'comparison_analyzer': [],
        'quality_analyzer': [],
        'report_generation': [],
        'lens_detector': []
    }
    
    # Search for specific patterns
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.py') and '__pycache__' not in root:
                filepath = os.path.join(root, file)
                rel_path = filepath.replace(base_path + '/', '')
                
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        lines = content.split('\n')
                        
                        for i, line in enumerate(lines, 1):
                            # Look for detector classes
                            if re.search(r'class\s+\w*Detector', line):
                                patterns['lens_detector'].append(f"{rel_path}:{i} - {line.strip()}")
                            
                            # Look for UnifiedDetector variations
                            if re.search(r'Unified\w*Detector|UnifiedLens\w*', line):
                                patterns['unified_detector'].append(f"{rel_path}:{i} - {line.strip()}")
                            
                            # Look for comparison/analyzer patterns
                            if re.search(r'Comparison\w*Analyzer|Quality\w*Analyzer', line):
                                patterns['comparison_analyzer'].append(f"{rel_path}:{i} - {line.strip()}")
                            
                            # Look for quality analysis functions
                            if re.search(r'def\s+\w*quality\w*|def\s+\w*compare\w*', line):
                                patterns['quality_analyzer'].append(f"{rel_path}:{i} - {line.strip()}")
                            
                            # Look for report generation
                            if re.search(r'def\s+\w*report|generate.*report|create.*report', line):
                                patterns['report_generation'].append(f"{rel_path}:{i} - {line.strip()}")
                
                except Exception as e:
                    print(f"Error reading {rel_path}: {e}")
    
    # Print findings
    print("=== TAIL-CHASING BUG PATTERNS FOUND ===\n")
    
    print("1. DETECTOR CLASS VARIATIONS:")
    print(f"   Found {len(patterns['lens_detector'])} detector classes")
    for item in patterns['lens_detector'][:10]:
        print(f"   {item}")
    
    print("\n2. UNIFIED DETECTOR VARIANTS (potential circular renaming):")
    for item in patterns['unified_detector']:
        print(f"   {item}")
    
    print("\n3. COMPARISON/QUALITY ANALYZER DUPLICATES:")
    comparison_items = patterns['comparison_analyzer'] + patterns['quality_analyzer']
    for item in comparison_items[:15]:
        print(f"   {item}")
    
    print("\n4. REPORT GENERATION DUPLICATES:")
    for item in patterns['report_generation'][:10]:
        print(f"   {item}")
    
    # Look for specific imports that might be problematic
    print("\n\n5. PROBLEMATIC IMPORT PATTERNS:")
    
    import_issues = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.py') and '__pycache__' not in root:
                filepath = os.path.join(root, file)
                rel_path = filepath.replace(base_path + '/', '')
                
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        
                        # Find imports of potentially renamed classes
                        imports = re.findall(r'from .+ import (.+)', content)
                        for imp in imports:
                            if 'UnifiedDetector' in imp or 'UnifiedLensDetector' in imp:
                                import_issues.append(f"{rel_path} - imports: {imp}")
                            if 'ComparisonAnalyzer' in imp and 'QualityAnalyzer' in content:
                                import_issues.append(f"{rel_path} - imports both ComparisonAnalyzer and uses QualityAnalyzer")
                
                except:
                    pass
    
    for issue in import_issues[:10]:
        print(f"   {issue}")
    
    return patterns

if __name__ == "__main__":
    find_tail_chasing_bugs()
