#!/usr/bin/env python3
"""Find tail-chasing issues in VintageOptics codebase."""

import os
import re
from pathlib import Path

def find_pass_only_functions(root_path):
    """Find all pass-only functions."""
    pass_only_pattern = re.compile(r'def\s+(\w+)\s*\([^)]*\)\s*:\s*pass', re.MULTILINE)
    results = []
    
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        matches = pass_only_pattern.findall(content)
                        if matches:
                            rel_path = os.path.relpath(filepath, root_path)
                            results.append((rel_path, matches))
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    
    return results

def find_not_implemented_functions(root_path):
    """Find all NotImplementedError functions."""
    pattern = re.compile(r'def\s+(\w+)\s*\([^)]*\)\s*:\s*raise\s+NotImplementedError', re.MULTILINE)
    results = []
    
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        matches = pattern.findall(content)
                        if matches:
                            rel_path = os.path.relpath(filepath, root_path)
                            results.append((rel_path, matches))
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    
    return results

def find_duplicate_functions(root_path):
    """Find duplicate function definitions."""
    function_locations = {}
    
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        # Find all function definitions
                        pattern = re.compile(r'def\s+(\w+)\s*\(')
                        matches = pattern.findall(content)
                        
                        rel_path = os.path.relpath(filepath, root_path)
                        for func_name in matches:
                            if func_name not in function_locations:
                                function_locations[func_name] = []
                            function_locations[func_name].append(rel_path)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    
    # Find duplicates
    duplicates = {}
    for func_name, locations in function_locations.items():
        if len(locations) > 1 and not func_name.startswith('_'):  # Ignore private methods
            duplicates[func_name] = locations
    
    return duplicates

def main():
    root_path = "/Users/rohanvinaik/VintageOptics/src/vintageoptics"
    
    print("=== VintageOptics Tail-Chasing Issues ===\n")
    
    # Find pass-only functions
    pass_only = find_pass_only_functions(root_path)
    print(f"Found {sum(len(funcs) for _, funcs in pass_only)} pass-only functions in {len(pass_only)} files:")
    for filepath, funcs in pass_only[:10]:
        print(f"\n{filepath}:")
        for func in funcs:
            print(f"  - {func}()")
    if len(pass_only) > 10:
        print(f"\n... and {len(pass_only) - 10} more files")
    
    # Find NotImplementedError functions
    not_impl = find_not_implemented_functions(root_path)
    print(f"\n\nFound {sum(len(funcs) for _, funcs in not_impl)} NotImplementedError functions in {len(not_impl)} files")
    
    # Find duplicate functions
    duplicates = find_duplicate_functions(root_path)
    print(f"\n\nFound {len(duplicates)} duplicate function names:")
    for func_name, locations in list(duplicates.items())[:10]:
        print(f"\n{func_name}() found in:")
        for loc in locations:
            print(f"  - {loc}")
    
    return pass_only, not_impl, duplicates

if __name__ == "__main__":
    main()
