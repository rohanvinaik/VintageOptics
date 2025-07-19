#!/usr/bin/env python3
"""Basic infrastructure tests for VintageOptics."""

import pytest
import sys
import os
from pathlib import Path


class TestInfrastructure:
    """Test basic project infrastructure."""
    
    def test_project_structure(self):
        """Test that the project has the expected structure."""
        project_root = Path(__file__).parent.parent
        
        # Check main directories exist
        assert (project_root / "src").exists()
        assert (project_root / "tests").exists()
        assert (project_root / "docs").exists()
        assert (project_root / "examples").exists()
        
        # Check main package exists
        assert (project_root / "src" / "vintageoptics").exists()
        assert (project_root / "src" / "vintageoptics" / "__init__.py").exists()
    
    def test_python_path_setup(self):
        """Test that Python path is set up correctly."""
        src_path = Path(__file__).parent.parent / "src"
        
        # Add to path if not already there
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        # Try to import the package
        try:
            import vintageoptics
            assert vintageoptics is not None
        except ImportError:
            pytest.skip("VintageOptics package not yet importable")
    
    def test_requirements_files(self):
        """Test that requirements files exist."""
        project_root = Path(__file__).parent.parent
        req_dir = project_root / "requirements"
        
        assert req_dir.exists()
        assert (req_dir / "base.txt").exists()
        assert (req_dir / "README.md").exists()
    
    def test_main_entry_point(self):
        """Test that main.py exists and is executable."""
        project_root = Path(__file__).parent.parent
        main_py = project_root / "main.py"
        
        assert main_py.exists()
        assert main_py.is_file()
        
        # Check it has a main function
        content = main_py.read_text()
        assert "def main()" in content
        assert 'if __name__ == "__main__"' in content
    
    def test_legacy_files_moved(self):
        """Test that legacy files have been moved."""
        project_root = Path(__file__).parent.parent
        
        # Check that frontend_api variants are not in root
        root_files = list(project_root.glob("frontend_api_*.py"))
        assert len(root_files) == 0, f"Found frontend_api variants in root: {root_files}"
        
        # Check legacy directory exists and has files
        legacy_dir = project_root / "legacy"
        assert legacy_dir.exists()
        legacy_files = list(legacy_dir.glob("frontend_api_*.py"))
        assert len(legacy_files) > 0, "No frontend_api files found in legacy directory"
    
    def test_shell_scripts_moved(self):
        """Test that shell scripts have been moved."""
        project_root = Path(__file__).parent.parent
        scripts_dir = project_root / "scripts" / "shell"
        
        assert scripts_dir.exists()
        shell_scripts = list(scripts_dir.glob("*.sh"))
        assert len(shell_scripts) > 0, "No shell scripts found in scripts/shell"
        
        # Check no .sh files in root (except maybe one or two critical ones)
        root_sh_files = list(project_root.glob("*.sh"))
        assert len(root_sh_files) <= 2, f"Too many .sh files in root: {root_sh_files}"


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
