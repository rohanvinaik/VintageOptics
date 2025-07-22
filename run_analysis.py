#!/usr/bin/env python3
import os
import ast
from collections import defaultdict
import json

def find_all_duplicates():
    base_path = '/Users/rohanvinaik/VintageOptics/src/vintageoptics'
    
    # Track all findings
    findings = {
        'duplicate_functions': defaultdict(list),
        'duplicate_classes': defaultdict(list),
        'placeholder_functions': [],
        'circular_imports': [],
        'similar_names': defaultdict(list)
    }
    
    # First pass: collect all definitions
    all_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.py') and '__pycache__' not in root:
                filepath = os.path.join(root, file)
                all_files.append(filepath)
    
    print(f"Analyzing {len(all_files)} Python files...")
    
    # Analyze each file
    for filepath in all_files:
        rel_path = filepath.replace(base_path + '/', '')
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                tree = ast.parse(content)
                
                # Find functions and classes
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        findings['duplicate_functions'][node.name].append({
                            'file': rel_path,
                            'line': node.lineno,
                            'is_placeholder': len(node.body) == 1 and isinstance(node.body[0], (ast.Pass, ast.Expr))
                        })
                        
                        # Check for placeholder implementations
                        if len(node.body) == 1:
                            if isinstance(node.body[0], ast.Pass):
                                findings['placeholder_functions'].append(f"{rel_path}:{node.lineno} - {node.name}() contains only pass")
                            elif isinstance(node.body[0], ast.Raise) and hasattr(node.body[0].exc, 'func'):
                                if hasattr(node.body[0].exc.func, 'id') and node.body[0].exc.func.id == 'NotImplementedError':
                                    findings['placeholder_functions'].append(f"{rel_path}:{node.lineno} - {node.name}() raises NotImplementedError")
                    
                    elif isinstance(node, ast.ClassDef):
                        findings['duplicate_classes'][node.name].append({
                            'file': rel_path,
                            'line': node.lineno
                        })
        
        except Exception as e:
            print(f"Error analyzing {rel_path}: {e}")
    
    # Generate report
    print("\n=== ANALYSIS RESULTS ===\n")
    
    # Duplicate functions
    print("DUPLICATE FUNCTIONS:")
    dup_count = 0
    for func_name, locations in findings['duplicate_functions'].items():
        if len(locations) > 1:
            dup_count += 1
            print(f"\n'{func_name}' found in {len(locations)} locations:")
            for loc in locations[:5]:  # Show first 5
                placeholder = " [PLACEHOLDER]" if loc.get('is_placeholder') else ""
                print(f"  - {loc['file']}:{loc['line']}{placeholder}")
            if len(locations) > 5:
                print(f"  ... and {len(locations) - 5} more")
    
    print(f"\nTotal duplicate function names: {dup_count}")
    
    # Duplicate classes
    print("\n\nDUPLICATE CLASSES:")
    class_dup_count = 0
    for class_name, locations in findings['duplicate_classes'].items():
        if len(locations) > 1:
            class_dup_count += 1
            print(f"\n'{class_name}' found in {len(locations)} locations:")
            for loc in locations:
                print(f"  - {loc['file']}:{loc['line']}")
    
    print(f"\nTotal duplicate class names: {class_dup_count}")
    
    # Placeholder functions
    print(f"\n\nPLACEHOLDER IMPLEMENTATIONS: {len(findings['placeholder_functions'])}")
    for placeholder in findings['placeholder_functions'][:10]:
        print(f"  - {placeholder}")
    if len(findings['placeholder_functions']) > 10:
        print(f"  ... and {len(findings['placeholder_functions']) - 10} more")
    
    # Look for specific tail-chasing patterns
    print("\n\nTAIL-CHASING PATTERNS:")
    
    # Find similar named functions/classes
    all_names = list(findings['duplicate_functions'].keys()) + list(findings['duplicate_classes'].keys())
    for name1 in all_names:
        for name2 in all_names:
            if name1 != name2 and name1.lower().replace('_', '') == name2.lower().replace('_', ''):
                findings['similar_names'][name1].append(name2)
    
    if findings['similar_names']:
        print("\nSimilar names that might indicate tail-chasing:")
        for name, similar in list(findings['similar_names'].items())[:5]:
            print(f"  - '{name}' similar to: {', '.join(set(similar))}")
    
    return findings

if __name__ == "__main__":
    find_all_duplicates()
