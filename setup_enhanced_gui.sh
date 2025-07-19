#!/bin/bash

# VintageOptics Enhanced GUI Setup Script
# Installs dependencies for the enhanced GUI with equipment context support

echo "VintageOptics Enhanced GUI Setup"
echo "================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    else
        echo "unknown"
    fi
}

OS=$(detect_os)
echo "Detected OS: $OS"
echo ""

# Check Python
echo "Checking Python installation..."
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo "✓ Python $PYTHON_VERSION found"
else
    echo "✗ Python 3 not found. Please install Python 3.7 or later."
    exit 1
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install --upgrade pip >/dev/null 2>&1

PACKAGES=(
    "flask"
    "flask-cors"
    "numpy"
    "opencv-python"
    "pillow"
    "requests"  # For testing
)

for package in "${PACKAGES[@]}"; do
    echo -n "Installing $package... "
    if pip3 install -q "$package" 2>/dev/null; then
        echo "✓"
    else
        echo "✗ (may already be installed)"
    fi
done

# Check/Install ExifTool
echo ""
echo "Checking ExifTool installation..."
if command_exists exiftool; then
    EXIFTOOL_VERSION=$(exiftool -ver 2>/dev/null || echo "unknown")
    echo "✓ ExifTool $EXIFTOOL_VERSION found"
else
    echo "✗ ExifTool not found"
    echo ""
    echo "ExifTool is recommended for full metadata extraction."
    echo "Would you like to install it? (y/n)"
    read -r INSTALL_EXIFTOOL
    
    if [[ "$INSTALL_EXIFTOOL" == "y" ]]; then
        case $OS in
            macos)
                if command_exists brew; then
                    echo "Installing ExifTool via Homebrew..."
                    brew install exiftool
                else
                    echo "Homebrew not found. Install manually from: https://exiftool.org"
                fi
                ;;
            linux)
                echo "Installing ExifTool via apt..."
                sudo apt-get update && sudo apt-get install -y libimage-exiftool-perl
                ;;
            *)
                echo "Please install ExifTool manually from: https://exiftool.org"
                ;;
        esac
    fi
fi

# Create directories if needed
echo ""
echo "Setting up directory structure..."
mkdir -p frontend 2>/dev/null && echo "✓ Created frontend directory"
mkdir -p data/samples 2>/dev/null && echo "✓ Created data/samples directory"

# Check for VintageOptics installation
echo ""
echo "Checking VintageOptics installation..."
if python3 -c "import vintageoptics" 2>/dev/null; then
    echo "✓ VintageOptics modules found"
else
    echo "⚠ VintageOptics modules not found in Python path"
    echo "  The enhanced GUI will run in demo mode"
    echo ""
    echo "To enable full functionality:"
    echo "  1. Ensure VintageOptics is properly installed"
    echo "  2. Add VintageOptics/src to your PYTHONPATH"
fi

# Make scripts executable
echo ""
echo "Making scripts executable..."
chmod +x run_enhanced_gui.sh 2>/dev/null && echo "✓ run_enhanced_gui.sh"
chmod +x test_enhanced_gui.py 2>/dev/null && echo "✓ test_enhanced_gui.py"

# Summary
echo ""
echo "================================"
echo "Setup Complete!"
echo ""
echo "Next steps:"
echo "1. Run the enhanced GUI: ./run_enhanced_gui.sh"
echo "2. Test the installation: ./test_enhanced_gui.py [image_path]"
echo "3. Open frontend/index_enhanced.html in your browser"
echo ""
echo "Features available:"
if command_exists exiftool; then
    echo "✓ Full metadata extraction (ExifTool installed)"
else
    echo "⚠ Limited metadata extraction (install ExifTool for full support)"
fi

if python3 -c "import vintageoptics" 2>/dev/null; then
    echo "✓ Full VintageOptics processing"
else
    echo "⚠ Demo mode only (VintageOptics not found)"
fi

echo ""
echo "For more information, see ENHANCED_GUI_README.md"
