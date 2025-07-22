import ast
import os
from difflib import SequenceMatcher
from collections import defaultdict

class FunctionAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.functions = defaultdict(list)
        self.current_file = ""
        
    def visit_FunctionDef(self, node):
        self.functions[node.name].append({
            'file': self.current_file,
            'line': node.lineno,
            'args': [arg.arg for arg in node.args.args],
            'body_size': len(node.body)
        })
        self.generic_visit(node)

def analyze_vintageoptics():
    analyzer = FunctionAnalyzer()
    
    for root, dirs, files in os.walk('src/vintageoptics'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                analyzer.current_file = filepath
                
                with open(filepath, 'r') as f:
                    try:
                        tree = ast.parse(f.read())
                        analyzer.visit(tree)
                    except SyntaxError:
                        print(f"Syntax error in {filepath}")
    
    # Find duplicate function names
    print("=== DUPLICATE FUNCTION NAMES ===")
    for func_name, locations in analyzer.functions.items():
        if len(locations) > 1:
            print(f"\nFunction '{func_name}' found in multiple locations:")
            for loc in locations:
                print(f"  - {loc['file']}:{loc['line']} (args: {loc['args']})")

if __name__ == "__main__":
    analyze_vintageoptics()
