#!/bin/bash

# VintageOptics GitHub Push Script
# This script will help you push your project to GitHub

echo "========================================="
echo "VintageOptics GitHub Setup and Push"
echo "========================================="
echo ""

# Navigate to the project directory
cd /Users/rohanvinaik/VintageOptics/VintageOptics

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
    echo "✓ Git repository initialized"
else
    echo "✓ Git repository already initialized"
fi

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo "Creating .gitignore file..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
bin/
include/
pip-selfcheck.json

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Project specific
data/cache/
data/databases/*.db
data/models/*.pth
*.log
.env
.env.local

# Jupyter Notebook
.ipynb_checkpoints

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/
.hypothesis/

# Documentation builds
docs/_build/

# macOS
.DS_Store
.AppleDouble
.LSOverride

# Temporary files
*.tmp
*.bak
*.swp
*~.nib
EOF
    echo "✓ .gitignore created"
else
    echo "✓ .gitignore already exists"
fi

# Show current status
echo ""
echo "Current Git status:"
echo "-------------------"
git status

# Add remote if not already added
echo ""
echo "Checking remote configuration..."
if ! git remote | grep -q "origin"; then
    echo "Adding GitHub remote..."
    git remote add origin https://github.com/rohanvinaik/VintageOptics.git
    echo "✓ Remote 'origin' added"
else
    echo "✓ Remote 'origin' already configured"
    echo "Current remote URL:"
    git remote -v
fi

# Create initial commit if needed
echo ""
echo "Checking for commits..."
if ! git rev-parse HEAD >/dev/null 2>&1; then
    echo "No commits found. Creating initial commit..."
    echo ""
    echo "Adding all files to staging..."
    git add .
    echo ""
    echo "Creating initial commit..."
    git commit -m "Initial commit: VintageOptics - Advanced lens correction with character preservation

- Comprehensive lens detection system with EXIF, Lensfun, and optical fingerprinting
- Enhanced statistical cleanup for vintage lens defects (dust, scratches, fungus, etc.)
- Advanced character preservation system maintaining lens personality
- Integration with free-use libraries (Lensfun, ExifTool, etc.)
- Instance profiling for individual lens copies
- Physics-based correction with multiple distortion models
- Depth-aware processing capabilities"
    echo "✓ Initial commit created"
else
    echo "✓ Repository already has commits"
    echo ""
    echo "Recent commits:"
    git log --oneline -5
fi

# Instructions for pushing
echo ""
echo "========================================="
echo "READY TO PUSH TO GITHUB"
echo "========================================="
echo ""
echo "To push your code to GitHub, run these commands:"
echo ""
echo "1. If this is your first push:"
echo "   git push -u origin main"
echo ""
echo "   (Note: If your default branch is 'master' instead of 'main', use:"
echo "    git branch -M main"
echo "    git push -u origin main)"
echo ""
echo "2. For subsequent pushes:"
echo "   git push"
echo ""
echo "3. If you made changes after running this script:"
echo "   git add ."
echo "   git commit -m \"Your commit message\""
echo "   git push"
echo ""
echo "========================================="
echo ""
echo "Current branch:"
git branch --show-current
echo ""
echo "Files ready to be committed (if any):"
git status --short
