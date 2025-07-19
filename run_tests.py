#!/usr/bin/env python3
"""
Test runner for VintageOptics.

This script runs the test suite with appropriate configuration.
"""

import sys
import subprocess
import os
from pathlib import Path

def run_tests():
    """Run the VintageOptics test suite."""
    # Get the project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Ensure src is in PYTHONPATH
    src_path = project_root / "src"
    python_path = os.environ.get("PYTHONPATH", "")
    if str(src_path) not in python_path:
        os.environ["PYTHONPATH"] = f"{src_path}{os.pathsep}{python_path}"
    
    print("Running VintageOptics Test Suite")
    print("=" * 50)
    print(f"Project root: {project_root}")
    print(f"Python path: {os.environ['PYTHONPATH']}")
    print("=" * 50)
    
    # Run pytest with appropriate options
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "-x",  # Stop on first failure
        "--import-mode=importlib",  # Use importlib mode
        "-p", "no:warnings",  # Disable warnings
    ]
    
    # Add coverage if available
    try:
        import pytest_cov
        cmd.extend(["--cov=src/vintageoptics", "--cov-report=term-missing"])
    except ImportError:
        print("Note: Install pytest-cov for coverage reports")
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    # Run the tests
    result = subprocess.run(cmd)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_tests())
