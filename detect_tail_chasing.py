#!/usr/bin/env python3
"""
Tail-Chasing Bug Detector for VintageOptics
Identifies circular dependencies, redundant implementations, and phantom classes
"""

import ast
import os
from collections import defaultdict
from pathlib import Path

class TailChasingDetector:
    def __init__(self):
        self.classes = defaultdict(list)  # class_name -> [(file, line)]
        self.functions = defaultdict(list)  # func_name -> [(file, line, class)]
        self.imports = defaultdict(set)  # file -> set of imports
        self.current_file = ""
        self.current_class = None
        
    def analyze_file(self, filepath):
        self.current_file = filepath
        self.current_class = None
        
        with open(filepath, 'r') as f:
            try:
                tree = ast.parse(f.read())
                self.visit(tree)
            except SyntaxError:
                print(f"Syntax error in {filepath}")
                
    def visit(self, node):
        if isinstance(node, ast.ClassDef):
            self.classes[node.name].append((self.current_file, node.lineno))
            old_class = self.current_class
            self.current_class = node.name
            for child in node.body:
                self.visit(child)
            self.current_class = old_class
            
        elif isinstance(node, ast.FunctionDef):
            self.functions[node.name].append((
                self.current_file, 
                node.lineno, 
                self.current_class
            ))
            for child in node.body:
                self.visit(child)
                
        elif isinstance(node, ast.Import):
            for alias in node.names:
                self.imports[self.current_file].add(alias.name)
                
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                self.imports[self.current_file].add(node.module)
                
        else:
            for child in ast.iter_child_nodes(node):
                self.visit(child)

def detect_tail_chasing_bugs(root_dir="src/vintageoptics"):
    detector = TailChasingDetector()
    
    # Analyze all Python files
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                detector.analyze_file(filepath)
    
    print("=== TAIL-CHASING BUG ANALYSIS ===\n")
    
    # 1. Circular Dependencies (classes with similar names)
    print("1. POTENTIAL CIRCULAR DEPENDENCIES / RENAMED CLASSES:")
    similar_classes = defaultdict(list)
    for class_name, locations in detector.classes.items():
        # Group by similar names (e.g., UnifiedDetector vs UnifiedLensDetector)
        base_name = class_name.replace('Lens', '').replace('Unified', '').replace('Base', '')
        similar_classes[base_name].append((class_name, locations))
    
    for base, variants in similar_classes.items():
        if len(variants) > 1:
            print(f"\n  Similar classes around '{base}':")
            for class_name, locations in variants:
                for file, line in locations:
                    print(f"    - {class_name} at {file}:{line}")
    
    # 2. Redundant Implementations
    print("\n\n2. REDUNDANT FUNCTION IMPLEMENTATIONS:")
    # Look for functions with similar names across different files
    function_groups = defaultdict(list)
    for func_name, locations in detector.functions.items():
        # Group functions by normalized name
        normalized = func_name.lower().replace('_', '')
        function_groups[normalized].append((func_name, locations))
    
    for normalized, variants in function_groups.items():
        if len(variants) > 1 or (len(variants) == 1 and len(variants[0][1]) > 1):
            print(f"\n  Potential redundancy for '{normalized}':")
            for func_name, locations in variants:
                for file, line, class_name in locations:
                    class_info = f" (in class {class_name})" if class_name else ""
                    print(f"    - {func_name}{class_info} at {file}:{line}")
    
    # 3. Phantom Implementations
    print("\n\n3. PHANTOM IMPLEMENTATIONS (Empty/Stub Classes):")
    # Look for very small classes that might be placeholders
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    content = f.read()
                    # Find classes with only pass or very minimal implementation
                    if 'class' in content:
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if line.strip().startswith('class '):
                                # Check next few lines for minimal implementation
                                class_body = []
                                j = i + 1
                                indent_level = None
                                while j < len(lines):
                                    curr_line = lines[j]
                                    if curr_line.strip() == '':
                                        j += 1
                                        continue
                                    
                                    # Determine indent level
                                    if indent_level is None and curr_line != curr_line.lstrip():
                                        indent_level = len(curr_line) - len(curr_line.lstrip())
                                    
                                    # Check if still in class
                                    if indent_level and curr_line.startswith(' ' * indent_level):
                                        class_body.append(curr_line.strip())
                                    else:
                                        break
                                    j += 1
                                
                                # Check if it's a phantom implementation
                                if len(class_body) <= 3 or all(
                                    line in ['pass', 'pass', '...'] or 
                                    line.startswith('def __init__') 
                                    for line in class_body
                                ):
                                    class_match = line.strip()
                                    print(f"    - {filepath}:{i+1} - {class_match}")
    
    # 4. Import-Driven Architecture Issues
    print("\n\n4. IMPORT-DRIVEN ARCHITECTURE ISSUES:")
    # Find imports that don't correspond to actual implementations
    all_defined_classes = set(detector.classes.keys())
    all_defined_functions = set(detector.functions.keys())
    
    for file, imports in detector.imports.items():
        for imp in imports:
            if '.' in imp:
                # Check if it's an internal import
                if 'vintageoptics' in imp:
                    parts = imp.split('.')
                    potential_class = parts[-1]
                    # Check if this import actually exists
                    if potential_class not in all_defined_classes and potential_class not in all_defined_functions:
                        print(f"    - {file} imports non-existent '{imp}'")

if __name__ == "__main__":
    detect_tail_chasing_bugs()
