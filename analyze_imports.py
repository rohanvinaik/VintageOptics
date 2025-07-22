import ast
import os
from collections import defaultdict

class ImportAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.imports = defaultdict(set)
        self.current_module = ""
        
    def visit_Import(self, node):
        for alias in node.names:
            self.imports[self.current_module].add(alias.name)
            
    def visit_ImportFrom(self, node):
        if node.module:
            self.imports[self.current_module].add(node.module)

def find_circular_imports():
    analyzer = ImportAnalyzer()
    
    # Analyze all files
    for root, dirs, files in os.walk('src/vintageoptics'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                module_path = filepath.replace('/', '.').replace('.py', '')
                analyzer.current_module = module_path
                
                with open(filepath, 'r') as f:
                    try:
                        tree = ast.parse(f.read())
                        analyzer.visit(tree)
                    except SyntaxError:
                        pass
    
    # Detect circular dependencies
    print("=== POTENTIAL CIRCULAR IMPORTS ===")
    for module, imports in analyzer.imports.items():
        for imported in imports:
            if imported in analyzer.imports:
                if module in analyzer.imports[imported]:
                    print(f"Circular: {module} <-> {imported}")

if __name__ == "__main__":
    find_circular_imports()
