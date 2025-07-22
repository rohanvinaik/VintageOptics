#!/usr/bin/env python3
import os
import ast

def find_placeholder_functions():
    base_path = '/Users/rohanvinaik/VintageOptics/src/vintageoptics'
    placeholders = []
    
    for root, dirs, files in os.walk(base_path):
        if '__pycache__' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                rel_path = filepath.replace(base_path + '/', '')
                
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        tree = ast.parse(content)
                        
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                # Check for pass-only functions
                                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                                    placeholders.append(f"{rel_path}:{node.lineno} - {node.name}() - pass only")
                                
                                # Check for NotImplementedError
                                elif len(node.body) == 1 and isinstance(node.body[0], ast.Raise):
                                    if hasattr(node.body[0].exc, 'func') and hasattr(node.body[0].exc.func, 'id'):
                                        if node.body[0].exc.func.id == 'NotImplementedError':
                                            placeholders.append(f"{rel_path}:{node.lineno} - {node.name}() - NotImplementedError")
                                
                                # Check for minimal returns
                                elif len(node.body) == 1 and isinstance(node.body[0], ast.Return):
                                    if node.body[0].value is None:
                                        placeholders.append(f"{rel_path}:{node.lineno} - {node.name}() - returns None")
                                    elif isinstance(node.body[0].value, ast.Constant):
                                        if node.body[0].value.value in [None, False, True, 0, "", [], {}]:
                                            placeholders.append(f"{rel_path}:{node.lineno} - {node.name}() - returns {node.body[0].value.value}")
                
                except Exception as e:
                    print(f"Error parsing {rel_path}: {e}")
    
    return placeholders

# Find placeholders
placeholders = find_placeholder_functions()

print(f"Found {len(placeholders)} placeholder functions:\n")
for p in placeholders[:20]:  # Show first 20
    print(p)

if len(placeholders) > 20:
    print(f"\n... and {len(placeholders) - 20} more")
