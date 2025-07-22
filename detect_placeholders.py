import ast
import os
import re

class PlaceholderDetector(ast.NodeVisitor):
    def __init__(self):
        self.issues = []
        self.current_file = ""
        
    def visit_FunctionDef(self, node):
        # Check for pass-only functions
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            self.add_issue(node.name, node.lineno, "Function contains only 'pass'")
            
        # Check for NotImplementedError
        for stmt in node.body:
            if self.is_not_implemented(stmt):
                self.add_issue(node.name, node.lineno, "Raises NotImplementedError")
                
        # Check for minimal returns
        if self.has_minimal_return(node):
            self.add_issue(node.name, node.lineno, "Returns minimal value (None/False/True/0/empty)")
            
        self.generic_visit(node)
        
    def is_not_implemented(self, stmt):
        if isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call):
            if hasattr(stmt.exc.func, 'id') and stmt.exc.func.id == 'NotImplementedError':
                return True
        return False
        
    def has_minimal_return(self, node):
        if len(node.body) == 1 and isinstance(node.body[0], ast.Return):
            if node.body[0].value is None:
                return True
            if isinstance(node.body[0].value, ast.Constant):
                if node.body[0].value.value in [None, False, True, 0, "", [], {}]:
                    return True
        return False
        
    def add_issue(self, func_name, line, description):
        self.issues.append({
            'file': self.current_file,
            'function': func_name,
            'line': line,
            'issue': description
        })

def scan_for_placeholders():
    detector = PlaceholderDetector()
    
    # Also search for TODO patterns in comments
    todo_pattern = re.compile(r'#.*?(TODO|FIXME|HACK|XXX|IMPLEMENT|PLACEHOLDER)', re.IGNORECASE)
    
    for root, dirs, files in os.walk('src/vintageoptics'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                detector.current_file = filepath
                
                with open(filepath, 'r') as f:
                    content = f.read()
                    
                    # AST analysis
                    try:
                        tree = ast.parse(content)
                        detector.visit(tree)
                    except SyntaxError:
                        print(f"Syntax error in {filepath}")
                    
                    # Comment analysis
                    for i, line in enumerate(content.split('\n'), 1):
                        if todo_pattern.search(line):
                            print(f"{filepath}:{i} - TODO/FIXME comment found")
    
    # Report findings
    print("\n=== PLACEHOLDER IMPLEMENTATIONS ===")
    for issue in detector.issues:
        print(f"{issue['file']}:{issue['line']} - {issue['function']}(): {issue['issue']}")

if __name__ == "__main__":
    scan_for_placeholders()
