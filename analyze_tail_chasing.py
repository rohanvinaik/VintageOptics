#!/usr/bin/env python3
"""
Comprehensive tail-chasing analysis for VintageOptics.
This shows how TailChasingFixer would analyze the codebase.
"""

import os
import re
import ast
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

class TailChasingAnalyzer:
    def __init__(self, root_path):
        self.root_path = Path(root_path)
        self.issues = {
            'pass_only_functions': [],
            'not_implemented_functions': [],
            'duplicate_functions': defaultdict(list),
            'circular_imports': [],
            'phantom_imports': [],
            'wrapper_abstractions': [],
            'semantic_duplicates': [],
            'missing_symbols': []
        }
        self.stats = {
            'total_files': 0,
            'total_functions': 0,
            'total_classes': 0,
            'total_imports': 0
        }
        
    def analyze(self):
        """Run complete analysis."""
        print("🔍 Running TailChasingFixer-style analysis on VintageOptics...\n")
        
        # Scan all Python files
        py_files = list(self.root_path.rglob("*.py"))
        py_files = [f for f in py_files if '__pycache__' not in str(f)]
        self.stats['total_files'] = len(py_files)
        
        print(f"📁 Analyzing {len(py_files)} Python files...\n")
        
        # Run various analyses
        self._analyze_placeholder_functions(py_files)
        self._analyze_imports(py_files)
        self._analyze_duplicates(py_files)
        self._analyze_wrappers(py_files)
        self._calculate_risk_scores()
        
        return self.generate_report()
    
    def _analyze_placeholder_functions(self, py_files):
        """Find pass-only and NotImplementedError functions."""
        pass_pattern = re.compile(r'def\s+(\w+)\s*\([^)]*\)\s*:\s*pass\s*$', re.MULTILINE)
        not_impl_pattern = re.compile(r'def\s+(\w+)\s*\([^)]*\)\s*:\s*raise\s+NotImplementedError', re.MULTILINE)
        return_none_pattern = re.compile(r'def\s+(\w+)\s*\([^)]*\)\s*:\s*return\s+None\s*$', re.MULTILINE)
        
        for py_file in py_files:
            try:
                content = py_file.read_text()
                
                # Find pass-only functions
                for match in pass_pattern.finditer(content):
                    self.issues['pass_only_functions'].append({
                        'file': str(py_file.relative_to(self.root_path)),
                        'function': match.group(1),
                        'line': content[:match.start()].count('\n') + 1,
                        'risk_score': 2  # Phantom function weight
                    })
                
                # Find NotImplementedError functions
                for match in not_impl_pattern.finditer(content):
                    self.issues['not_implemented_functions'].append({
                        'file': str(py_file.relative_to(self.root_path)),
                        'function': match.group(1),
                        'line': content[:match.start()].count('\n') + 1,
                        'risk_score': 2
                    })
                
                # Find return None only functions
                for match in return_none_pattern.finditer(content):
                    self.issues['pass_only_functions'].append({
                        'file': str(py_file.relative_to(self.root_path)),
                        'function': match.group(1),
                        'line': content[:match.start()].count('\n') + 1,
                        'risk_score': 1
                    })
                    
            except Exception as e:
                print(f"Error analyzing {py_file}: {e}")
    
    def _analyze_imports(self, py_files):
        """Analyze imports for circular dependencies and missing symbols."""
        import_graph = defaultdict(set)
        all_defined_symbols = set()
        symbol_usage = defaultdict(list)
        
        # First pass: collect all defined symbols and imports
        for py_file in py_files:
            try:
                content = py_file.read_text()
                module_path = str(py_file.relative_to(self.root_path)).replace('.py', '').replace('/', '.')
                
                # Parse AST to find definitions
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        all_defined_symbols.add(node.name)
                        self.stats['total_classes'] += 1
                    elif isinstance(node, ast.FunctionDef):
                        all_defined_symbols.add(node.name)
                        self.stats['total_functions'] += 1
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            import_graph[module_path].add(alias.name)
                            self.stats['total_imports'] += 1
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            import_graph[module_path].add(node.module)
                            for alias in node.names:
                                symbol_usage[alias.name].append({
                                    'file': str(py_file.relative_to(self.root_path)),
                                    'from_module': node.module
                                })
                                
            except Exception as e:
                print(f"Error parsing {py_file}: {e}")
        
        # Find circular imports
        for module1, imports in import_graph.items():
            for module2 in imports:
                if module2 in import_graph and module1 in import_graph[module2]:
                    if module1 < module2:  # Avoid duplicates
                        self.issues['circular_imports'].append({
                            'module1': module1,
                            'module2': module2,
                            'risk_score': 3
                        })
        
        # Find missing symbols
        for symbol, usages in symbol_usage.items():
            if symbol not in all_defined_symbols and not symbol.startswith('_'):
                for usage in usages:
                    self.issues['missing_symbols'].append({
                        'symbol': symbol,
                        'file': usage['file'],
                        'imported_from': usage['from_module'],
                        'risk_score': 2
                    })
    
    def _analyze_duplicates(self, py_files):
        """Find duplicate function implementations."""
        function_signatures = defaultdict(list)
        
        for py_file in py_files:
            try:
                content = py_file.read_text()
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Create a signature
                        args = [arg.arg for arg in node.args.args]
                        sig = f"{node.name}({','.join(args)})"
                        
                        # Get function body characteristics
                        body_size = len(node.body)
                        has_docstring = (len(node.body) > 0 and 
                                       isinstance(node.body[0], ast.Expr) and 
                                       isinstance(node.body[0].value, ast.Str))
                        
                        function_signatures[node.name].append({
                            'file': str(py_file.relative_to(self.root_path)),
                            'line': node.lineno,
                            'signature': sig,
                            'body_size': body_size,
                            'has_docstring': has_docstring
                        })
                        
            except Exception as e:
                print(f"Error parsing {py_file}: {e}")
        
        # Find duplicates
        for func_name, locations in function_signatures.items():
            if len(locations) > 1:
                self.issues['duplicate_functions'][func_name] = locations
    
    def _analyze_wrappers(self, py_files):
        """Find trivial wrapper functions."""
        wrapper_pattern = re.compile(
            r'def\s+(\w+)\s*\([^)]*\)\s*:\s*\n\s*return\s+\w+\s*\([^)]*\)\s*$',
            re.MULTILINE
        )
        
        for py_file in py_files:
            try:
                content = py_file.read_text()
                
                for match in wrapper_pattern.finditer(content):
                    self.issues['wrapper_abstractions'].append({
                        'file': str(py_file.relative_to(self.root_path)),
                        'function': match.group(1),
                        'line': content[:match.start()].count('\n') + 1,
                        'risk_score': 1
                    })
                    
            except Exception as e:
                print(f"Error analyzing {py_file}: {e}")
    
    def _calculate_risk_scores(self):
        """Calculate overall risk scores."""
        self.total_risk_score = 0
        
        # Sum up all risk scores
        for issue_type, issues in self.issues.items():
            if issue_type == 'duplicate_functions':
                self.total_risk_score += len(issues) * 2
            elif isinstance(issues, list):
                for issue in issues:
                    if isinstance(issue, dict) and 'risk_score' in issue:
                        self.total_risk_score += issue['risk_score']
    
    def generate_report(self):
        """Generate comprehensive report."""
        report = []
        report.append("=" * 70)
        report.append("TAILCHASINGFIXER ANALYSIS REPORT - VintageOptics")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Statistics
        report.append("📊 CODEBASE STATISTICS:")
        report.append(f"  Total files analyzed: {self.stats['total_files']}")
        report.append(f"  Total functions: {self.stats['total_functions']}")
        report.append(f"  Total classes: {self.stats['total_classes']}")
        report.append(f"  Total imports: {self.stats['total_imports']}")
        report.append("")
        
        # Risk Score
        report.append(f"🎯 OVERALL RISK SCORE: {self.total_risk_score}")
        if self.total_risk_score < 15:
            report.append("  ✅ Low risk - Good code quality")
        elif self.total_risk_score < 30:
            report.append("  ⚠️  Medium risk - Some issues to address")
        else:
            report.append("  ❌ High risk - Significant tail-chasing patterns detected")
        report.append("")
        
        # Pass-only functions
        if self.issues['pass_only_functions']:
            report.append(f"🔴 PASS-ONLY FUNCTIONS ({len(self.issues['pass_only_functions'])} found):")
            report.append("  These are phantom implementations that need real code")
            for issue in self.issues['pass_only_functions'][:5]:
                report.append(f"  - {issue['file']}:{issue['line']} - {issue['function']}()")
            if len(self.issues['pass_only_functions']) > 5:
                report.append(f"  ... and {len(self.issues['pass_only_functions']) - 5} more")
            report.append("")
        
        # NotImplementedError functions
        if self.issues['not_implemented_functions']:
            report.append(f"🟡 NOT IMPLEMENTED FUNCTIONS ({len(self.issues['not_implemented_functions'])} found):")
            report.append("  These functions explicitly raise NotImplementedError")
            for issue in self.issues['not_implemented_functions'][:5]:
                report.append(f"  - {issue['file']}:{issue['line']} - {issue['function']}()")
            report.append("")
        
        # Duplicate functions
        if self.issues['duplicate_functions']:
            report.append(f"🔵 DUPLICATE FUNCTIONS ({len(self.issues['duplicate_functions'])} found):")
            report.append("  Multiple implementations of the same function name")
            for func_name, locations in list(self.issues['duplicate_functions'].items())[:5]:
                report.append(f"  - {func_name}() appears in {len(locations)} files:")
                for loc in locations[:3]:
                    report.append(f"    • {loc['file']}:{loc['line']}")
            report.append("")
        
        # Circular imports
        if self.issues['circular_imports']:
            report.append(f"🔄 CIRCULAR IMPORTS ({len(self.issues['circular_imports'])} found):")
            report.append("  These modules import each other, creating dependency cycles")
            for issue in self.issues['circular_imports'][:5]:
                report.append(f"  - {issue['module1']} <-> {issue['module2']}")
            report.append("")
        
        # Missing symbols
        if self.issues['missing_symbols']:
            report.append(f"❓ MISSING SYMBOLS ({len(self.issues['missing_symbols'])} found):")
            report.append("  References to undefined functions/classes")
            unique_symbols = set(issue['symbol'] for issue in self.issues['missing_symbols'])
            for symbol in list(unique_symbols)[:5]:
                report.append(f"  - {symbol} (referenced but not defined)")
            report.append("")
        
        # Wrapper abstractions
        if self.issues['wrapper_abstractions']:
            report.append(f"🎁 TRIVIAL WRAPPERS ({len(self.issues['wrapper_abstractions'])} found):")
            report.append("  Functions that just call another function")
            for issue in self.issues['wrapper_abstractions'][:3]:
                report.append(f"  - {issue['file']}:{issue['line']} - {issue['function']}()")
            report.append("")
        
        # Recommendations
        report.append("📋 RECOMMENDATIONS:")
        report.append("1. Implement all pass-only functions or remove them")
        report.append("2. Consolidate duplicate function implementations")
        report.append("3. Resolve circular imports by refactoring dependencies")
        report.append("4. Define missing symbols or fix incorrect imports")
        report.append("5. Consider removing trivial wrapper functions")
        report.append("")
        
        # Semantic analysis note
        report.append("💡 NOTE: This analysis uses AST-based detection.")
        report.append("   The full TailChasingFixer tool also includes:")
        report.append("   - Semantic hypervector analysis for deep duplicates")
        report.append("   - Temporal pattern analysis across git history")
        report.append("   - ML-based code similarity detection")
        report.append("   - Integration with other linting tools")
        
        return "\n".join(report)

def main():
    analyzer = TailChasingAnalyzer("/Users/rohanvinaik/VintageOptics/src/vintageoptics")
    report = analyzer.analyze()
    
    # Print report
    print(report)
    
    # Save report
    with open("/Users/rohanvinaik/VintageOptics/tailchasing_analysis_report.txt", "w") as f:
        f.write(report)
    
    # Also save JSON data
    json_data = {
        'timestamp': datetime.now().isoformat(),
        'stats': analyzer.stats,
        'total_risk_score': analyzer.total_risk_score,
        'issues': {
            'pass_only_functions': analyzer.issues['pass_only_functions'],
            'not_implemented_functions': analyzer.issues['not_implemented_functions'],
            'duplicate_functions': [
                {'function': name, 'locations': locs} 
                for name, locs in analyzer.issues['duplicate_functions'].items()
            ],
            'circular_imports': analyzer.issues['circular_imports'],
            'missing_symbols': analyzer.issues['missing_symbols'],
            'wrapper_abstractions': analyzer.issues['wrapper_abstractions']
        }
    }
    
    with open("/Users/rohanvinaik/VintageOptics/tailchasing_analysis.json", "w") as f:
        json.dump(json_data, f, indent=2)
    
    print("\n\n📄 Report saved to: tailchasing_analysis_report.txt")
    print("📊 JSON data saved to: tailchasing_analysis.json")

if __name__ == "__main__":
    main()
