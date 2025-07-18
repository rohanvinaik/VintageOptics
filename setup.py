"""
Setup script for VintageOptics package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read version
version_file = this_directory / "src" / "vintageoptics" / "__version__.py"
version_dict = {}
exec(version_file.read_text(), version_dict)
version = version_dict["__version__"]

setup(
    name="vintageoptics",
    version=version,
    author="VintageOptics Team",
    author_email="info@vintageoptics.dev",
    description="Advanced lens correction and synthesis with hyperdimensional computing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vintageoptics/vintageoptics",
    project_urls={
        "Bug Tracker": "https://github.com/vintageoptics/vintageoptics/issues",
        "Documentation": "https://vintageoptics.readthedocs.io",
        "Source Code": "https://github.com/vintageoptics/vintageoptics",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "scipy>=1.7.0",
        "scikit-image>=0.19.0",
        "Pillow>=9.0.0",
        
        # Machine learning
        "scikit-learn>=1.0.0",
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        
        # API
        "fastapi>=0.85.0",
        "uvicorn>=0.18.0",
        "pydantic>=1.9.0",
        "python-multipart>=0.0.5",
        
        # Database
        "sqlalchemy>=1.4.0",
        "alembic>=1.8.0",
        
        # Utilities
        "click>=8.0.0",
        "tqdm>=4.62.0",
        "colorama>=0.4.4",
        "pyyaml>=6.0",
        
        # Image processing
        "rawpy>=0.17.0",
        "imageio>=2.22.0",
        "colour-science>=0.4.0",
        
        # Optional but recommended
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.20.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "gpu": [
            "cupy>=11.0.0",
            "pycuda>=2022.1",
        ],
        "ml": [
            "tensorflow>=2.10.0",
            "transformers>=4.25.0",
            "timm>=0.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vintageoptics=vintageoptics.api.cli:main",
            "vo=vintageoptics.api.cli:main",  # Short alias
        ],
    },
    include_package_data=True,
    package_data={
        "vintageoptics": [
            "data/*.json",
            "data/*.yaml",
            "models/*.pth",
            "models/*.onnx",
        ],
    },
    zip_safe=False,
)
