#!/usr/bin/env python3
"""Simple sanity check tests for VintageOptics."""

import pytest
import sys
import os
from pathlib import Path


class TestSanityCheck:
    """Basic sanity checks that don't require complex imports."""
    
    def test_python_version(self):
        """Test Python version is 3.7 or higher."""
        assert sys.version_info >= (3, 7), "Python 3.7+ required"
    
    def test_project_directories_exist(self):
        """Test that main project directories exist."""
        project_root = Path(__file__).parent.parent
        
        directories = [
            "src",
            "tests", 
            "docs",
            "examples",
            "scripts",
            "requirements",
            "legacy",
            "dev",
            "frontend"
        ]
        
        for dir_name in directories:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"Directory {dir_name} does not exist"
            assert dir_path.is_dir(), f"{dir_name} is not a directory"
    
    def test_main_files_exist(self):
        """Test that main files exist."""
        project_root = Path(__file__).parent.parent
        
        files = [
            "main.py",
            "setup.py",
            "README.md",
            "LICENSE",
            "pyproject.toml",
            "frontend_api.py"
        ]
        
        for file_name in files:
            file_path = project_root / file_name
            assert file_path.exists(), f"File {file_name} does not exist"
            assert file_path.is_file(), f"{file_name} is not a file"
    
    def test_src_structure(self):
        """Test src directory structure."""
        src_path = Path(__file__).parent.parent / "src" / "vintageoptics"
        
        modules = [
            "__init__.py",
            "analysis",
            "api", 
            "calibration",
            "core",
            "database",
            "depth",
            "detection",
            "hyperdimensional",
            "integration",
            "physics",
            "statistical",
            "synthesis",
            "types",
            "utils"
        ]
        
        for module in modules:
            module_path = src_path / module
            assert module_path.exists(), f"Module {module} does not exist in src/vintageoptics"
    
    def test_no_merge_conflicts(self):
        """Test that no files have merge conflict markers."""
        project_root = Path(__file__).parent.parent
        
        conflict_markers = ["<<<<<<<", "=======", ">>>>>>>"]
        files_with_conflicts = []
        
        # Check Python files
        for py_file in project_root.rglob("*.py"):
            # Skip node_modules and other directories
            if any(skip in str(py_file) for skip in ["node_modules", ".git", "__pycache__", ".venv"]):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                for marker in conflict_markers:
                    if marker in content:
                        files_with_conflicts.append(str(py_file))
                        break
            except Exception:
                # Skip files that can't be read
                pass
        
        assert len(files_with_conflicts) == 0, f"Files with merge conflicts: {files_with_conflicts}"
    
    def test_requirements_files_valid(self):
        """Test that requirements files exist and are not empty."""
        req_dir = Path(__file__).parent.parent / "requirements"
        
        req_files = ["base.txt", "dev.txt", "ml.txt", "gpu.txt"]
        
        for req_file in req_files:
            file_path = req_dir / req_file
            if file_path.exists():
                content = file_path.read_text().strip()
                assert len(content) > 0, f"{req_file} is empty"
                # Basic check for valid pip format
                lines = [l.strip() for l in content.split('\n') if l.strip() and not l.startswith('#')]
                assert len(lines) > 0, f"{req_file} has no package entries"
    
    def test_refactoring_complete(self):
        """Test that refactoring was completed successfully."""
        project_root = Path(__file__).parent.parent
        
        # Check files were moved out of root
        root_files = list(project_root.glob("frontend_api_*.py"))
        assert len(root_files) == 0, "frontend_api variants still in root"
        
        root_sh_files = [f for f in project_root.glob("*.sh") if f.name != "setup.sh"]
        assert len(root_sh_files) <= 2, f"Too many shell scripts in root: {[f.name for f in root_sh_files]}"
        
        # Check legacy directory has content
        legacy_dir = project_root / "legacy"
        legacy_files = list(legacy_dir.glob("*.py"))
        assert len(legacy_files) > 0, "No files in legacy directory"
        
        # Check scripts/shell has content
        shell_dir = project_root / "scripts" / "shell"
        shell_scripts = list(shell_dir.glob("*.sh"))
        assert len(shell_scripts) > 0, "No shell scripts in scripts/shell"
    
    def test_duplicate_removed(self):
        """Test that the duplicate VintageOptics directory was removed."""
        project_root = Path(__file__).parent.parent
        duplicate = project_root / "VintageOptics"
        
        assert not duplicate.exists(), "Duplicate VintageOptics directory still exists"
        
        # Check backup exists
        backup = project_root / "VintageOptics_duplicate_backup"
        assert backup.exists(), "Backup of duplicate directory not found"


if __name__ == "__main__":
    # Run with pytest if available, otherwise run directly
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        print("Running tests without pytest...")
        test = TestSanityCheck()
        
        test_methods = [m for m in dir(test) if m.startswith("test_")]
        passed = 0
        failed = 0
        
        for method_name in test_methods:
            try:
                method = getattr(test, method_name)
                method()
                print(f"✓ {method_name}")
                passed += 1
            except AssertionError as e:
                print(f"✗ {method_name}: {e}")
                failed += 1
            except Exception as e:
                print(f"✗ {method_name}: Unexpected error: {e}")
                failed += 1
        
        print(f"\nResults: {passed} passed, {failed} failed")
        sys.exit(0 if failed == 0 else 1)
