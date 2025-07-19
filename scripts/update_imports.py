#!/usr/bin/env python3
"""
Update imports after refactoring.

This script updates import statements in Python files to reflect the new directory structure.
"""

import os
import re
from pathlib import Path

def update_imports_in_file(filepath):
    """Update import statements in a single file."""
    updates = []
    
    # Map old locations to new locations
    import_mappings = {
        # Test file imports
        r'from tests.test_': 'from tests.test_',
        r'import tests.test_': 'import tests.test_',
        
        # Demo imports
        r'from demos.demo_': 'from demos.demo_',
        r'import demos.demo_': 'import demos.demo_',
        
        # Frontend API imports (keep the main one)
        r'from frontend_api_\w+ import': 'from frontend_api import',
        r'import frontend_api_\w+': 'import frontend_api',
    }
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply all mappings
        for pattern, replacement in import_mappings.items():
            content = re.sub(pattern, replacement, content)
        
        # Only write if changes were made
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            updates.append(filepath)
            
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
    
    return updates

def update_all_imports(root_dir):
    """Update imports in all Python files."""
    root_path = Path(root_dir)
    all_updates = []
    
    # Skip these directories
    skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'env'}
    
    for py_file in root_path.rglob('*.py'):
        # Skip files in directories we want to ignore
        if any(skip_dir in py_file.parts for skip_dir in skip_dirs):
            continue
            
        updates = update_imports_in_file(py_file)
        all_updates.extend(updates)
    
    return all_updates

def main():
    """Main function."""
    print("Updating imports after refactoring...")
    
    # Get the VintageOptics root directory
    script_dir = Path(__file__).parent
    
    # Update imports
    updated_files = update_all_imports(script_dir)
    
    if updated_files:
        print(f"\nUpdated {len(updated_files)} files:")
        for f in updated_files:
            print(f"  - {f}")
    else:
        print("\nNo import updates needed.")
    
    print("\nRefactoring complete!")
    print("\nNext steps:")
    print("1. Run tests to ensure everything works: pytest tests/")
    print("2. Update any shell scripts that reference moved files")
    print("3. Commit the changes: git add -A && git commit -m 'Refactor: Clean up project structure'")

if __name__ == "__main__":
    main()
