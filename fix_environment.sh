#!/bin/bash

echo "🔧 VintageOptics Environment Fix Script"
echo "======================================"

# Fix numpy version conflict
echo "📦 Fixing NumPy version conflict..."
pip install "numpy>=1.21.0,<2.0.0" --force-reinstall

echo "📦 Installing compatible versions..."
pip install "scipy>=1.7.0,<1.14.0" --force-reinstall
pip install "numba>=0.53.0,<0.61.0" --force-reinstall

echo "✅ Dependencies fixed! Testing imports..."
python debug_imports.py

echo "🎉 Setup complete!"
