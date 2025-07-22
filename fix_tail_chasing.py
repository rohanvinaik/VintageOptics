#!/usr/bin/env python3
"""
Tail-Chasing Bug Fixer for VintageOptics
Automatically identifies and suggests fixes for common tail-chasing patterns
"""

import ast
import os
import re
from pathlib import Path
from collections import defaultdict

class TailChasingFixer:
    def __init__(self):
        self.fixes = []
        self.import_map = defaultdict(set)  # module -> set of (file, line)
        
    def analyze_and_fix(self, root_dir="src/vintageoptics"):
        # Step 1: Build a complete map of what's actually implemented
        actual_implementations = self.scan_implementations(root_dir)
        
        # Step 2: Find problematic imports
        problematic_imports = self.find_problematic_imports(root_dir, actual_implementations)
        
        # Step 3: Find duplicate/redundant implementations
        duplicates = self.find_duplicates(actual_implementations)
        
        # Step 4: Generate fixes
        self.generate_fixes(problematic_imports, duplicates)
        
        return self.fixes
    
    def scan_implementations(self, root_dir):
        """Build a map of all actual implementations"""
        implementations = {
            'classes': defaultdict(list),  # class_name -> [(file, line, full_module_path)]
            'functions': defaultdict(list),  # func_name -> [(file, line, class, full_module_path)]
            'modules': set()  # all module paths
        }
        
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    module_path = filepath.replace('/', '.').replace('.py', '').replace('src.', '')
                    implementations['modules'].add(module_path)
                    
                    try:
                        with open(filepath, 'r') as f:
                            tree = ast.parse(f.read())
                            self._scan_ast(tree, filepath, module_path, implementations)
                    except SyntaxError:
                        pass
        
        return implementations
    
    def _scan_ast(self, tree, filepath, module_path, implementations, current_class=None):
        """Recursively scan AST for implementations"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                implementations['classes'][node.name].append((filepath, node.lineno, module_path))
                # Scan methods within class
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        implementations['functions'][item.name].append(
                            (filepath, item.lineno, node.name, module_path)
                        )
            elif isinstance(node, ast.FunctionDef) and current_class is None:
                implementations['functions'][node.name].append(
                    (filepath, node.lineno, None, module_path)
                )
    
    def find_problematic_imports(self, root_dir, implementations):
        """Find imports that reference non-existent or moved items"""
        problematic = []
        
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    
                    try:
                        with open(filepath, 'r') as f:
                            tree = ast.parse(f.read())
                            
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ImportFrom):
                                if node.module and 'vintageoptics' in str(node.module):
                                    # Check each imported name
                                    for alias in node.names:
                                        imported_name = alias.name
                                        
                                        # Check if this import is valid
                                        if not self._is_valid_import(
                                            node.module, imported_name, implementations
                                        ):
                                            problematic.append({
                                                'file': filepath,
                                                'line': node.lineno,
                                                'module': node.module,
                                                'name': imported_name,
                                                'type': 'invalid_import'
                                            })
                    except SyntaxError:
                        pass
        
        return problematic
    
    def _is_valid_import(self, module, name, implementations):
        """Check if an import is valid"""
        # Check if it's a class
        if name in implementations['classes']:
            for _, _, impl_module in implementations['classes'][name]:
                if impl_module == module:
                    return True
        
        # Check if it's a function
        if name in implementations['functions']:
            for _, _, _, impl_module in implementations['functions'][name]:
                if impl_module == module:
                    return True
        
        # Check if it's a module
        full_module = f"{module}.{name}"
        if full_module in implementations['modules']:
            return True
        
        return False
    
    def find_duplicates(self, implementations):
        """Find duplicate or very similar implementations"""
        duplicates = []
        
        # Check for duplicate class names
        for class_name, locations in implementations['classes'].items():
            if len(locations) > 1:
                duplicates.append({
                    'type': 'duplicate_class',
                    'name': class_name,
                    'locations': locations
                })
        
        # Check for similar function implementations
        func_groups = defaultdict(list)
        for func_name, locations in implementations['functions'].items():
            # Group by normalized name
            normalized = self._normalize_name(func_name)
            func_groups[normalized].append((func_name, locations))
        
        for normalized, variants in func_groups.items():
            if len(variants) > 1:
                all_locations = []
                for func_name, locations in variants:
                    all_locations.extend([(func_name, loc) for loc in locations])
                
                if len(all_locations) > 1:
                    duplicates.append({
                        'type': 'similar_functions',
                        'normalized': normalized,
                        'variants': all_locations
                    })
        
        return duplicates
    
    def _normalize_name(self, name):
        """Normalize a name for comparison"""
        # Remove common prefixes/suffixes and normalize
        name = name.lower()
        for prefix in ['get_', 'set_', 'compute_', 'calculate_', 'generate_']:
            if name.startswith(prefix):
                name = name[len(prefix):]
        for suffix in ['_report', '_analysis', '_quality']:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        return name.replace('_', '')
    
    def generate_fixes(self, problematic_imports, duplicates):
        """Generate fix suggestions"""
        # Fix problematic imports
        for problem in problematic_imports:
            self.fixes.append({
                'file': problem['file'],
                'line': problem['line'],
                'issue': f"Invalid import: from {problem['module']} import {problem['name']}",
                'fix': f"Remove this import or update to correct module path",
                'severity': 'high'
            })
        
        # Fix duplicates
        for dup in duplicates:
            if dup['type'] == 'duplicate_class':
                self.fixes.append({
                    'issue': f"Duplicate class '{dup['name']}' found in multiple files",
                    'locations': dup['locations'],
                    'fix': f"Consolidate into a single implementation and update imports",
                    'severity': 'high'
                })
            elif dup['type'] == 'similar_functions':
                self.fixes.append({
                    'issue': f"Similar functions found (normalized: '{dup['normalized']}')",
                    'locations': dup['variants'],
                    'fix': f"Consider consolidating these similar implementations",
                    'severity': 'medium'
                })

def main():
    print("=== TAIL-CHASING BUG FIXER ===\n")
    
    fixer = TailChasingFixer()
    fixes = fixer.analyze_and_fix()
    
    # Group fixes by severity
    high_severity = [f for f in fixes if f.get('severity') == 'high']
    medium_severity = [f for f in fixes if f.get('severity') == 'medium']
    
    print(f"Found {len(fixes)} issues to fix:\n")
    
    if high_severity:
        print("HIGH SEVERITY ISSUES:")
        print("-" * 50)
        for fix in high_severity:
            print(f"\n{fix['issue']}")
            if 'file' in fix and 'line' in fix:
                print(f"  Location: {fix['file']}:{fix['line']}")
            elif 'locations' in fix:
                print("  Locations:")
                for loc in fix['locations'][:5]:  # Show first 5
                    if isinstance(loc, tuple) and len(loc) >= 2:
                        print(f"    - {loc[0]} at {loc[1] if len(loc) > 1 else 'unknown'}")
            print(f"  Fix: {fix['fix']}")
    
    if medium_severity:
        print("\n\nMEDIUM SEVERITY ISSUES:")
        print("-" * 50)
        for fix in medium_severity:
            print(f"\n{fix['issue']}")
            if 'locations' in fix:
                print("  Locations:")
                for item in fix['locations'][:3]:  # Show first 3
                    if isinstance(item, tuple) and len(item) >= 2:
                        name, loc = item
                        print(f"    - {name} at {loc[0]}:{loc[1]}")
            print(f"  Fix: {fix['fix']}")
    
    # Generate fix script
    print("\n\nGenerating automated fix script...")
    generate_fix_script(fixes)

def generate_fix_script(fixes):
    """Generate a script to automatically apply fixes"""
    script_content = """#!/bin/bash
# Automated fix script for tail-chasing bugs
# Review each fix before applying!

echo "=== APPLYING TAIL-CHASING FIXES ==="

# Fix invalid imports
"""
    
    # Add commands to fix invalid imports
    import_fixes = [f for f in fixes if 'file' in f and 'line' in f]
    for fix in import_fixes:
        script_content += f"""
# Fix: {fix['issue']}
echo "Fixing {fix['file']}:{fix['line']}"
# sed -i '' '{fix['line']}d' {fix['file']}  # Uncomment to delete the line
"""
    
    # Save the script
    with open('apply_fixes.sh', 'w') as f:
        f.write(script_content)
    
    print("Fix script saved as 'apply_fixes.sh'")
    print("Review the script carefully before running it!")

if __name__ == "__main__":
    main()
