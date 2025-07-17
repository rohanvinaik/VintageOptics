# setup.py
setup(
    name="vintageoptics",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "numba>=0.53.0",
        "scipy>=1.7.0",
        # ... other dependencies
    ],
    entry_points={
        "console_scripts": [
            "vintageoptics=vintageoptics.interface.cli:main",
        ],
    },
)