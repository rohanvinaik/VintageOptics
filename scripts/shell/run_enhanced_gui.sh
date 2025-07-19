#!/bin/bash

# VintageOptics Enhanced GUI Startup Script
# This script starts the enhanced GUI with equipment context support

echo "Starting VintageOptics Enhanced GUI..."
echo "=================================="

# Change to the VintageOptics directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "No virtual environment found. Using system Python."
fi

# Check for ExifTool
if command -v exiftool &> /dev/null; then
    echo "✓ ExifTool found: $(exiftool -ver)"
else
    echo "⚠ ExifTool not found. Metadata extraction will be limited."
    echo "  Install from: https://exiftool.org"
fi

# Install/update Flask and CORS if needed
echo "Checking dependencies..."
pip install -q flask flask-cors pillow numpy opencv-python 2>/dev/null

# Start the enhanced GUI in the background
echo "Starting backend API on port 8000..."
python frontend_api_enhanced.py &
API_PID=$!

# Wait for API to start
sleep 2

# Open the enhanced frontend
echo "Opening enhanced GUI in browser..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open frontend/index_enhanced.html
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open frontend/index_enhanced.html 2>/dev/null || sensible-browser frontend/index_enhanced.html
else
    echo "Please open frontend/index_enhanced.html in your browser"
fi

echo ""
echo "VintageOptics Enhanced GUI is running!"
echo "=================================="
echo "Backend API: http://localhost:8000"
echo "Frontend: file://$(pwd)/frontend/index_enhanced.html"
echo ""
echo "Features:"
echo "  ✓ Automatic metadata extraction from images"
echo "  ✓ Equipment context input (camera & lens)"
echo "  ✓ Manual parameter override"
echo "  ✓ Custom lens profile creation"
echo ""
echo "Press Ctrl+C to stop the server"

# Wait for user to stop
trap "echo 'Shutting down...'; kill $API_PID 2>/dev/null; exit" INT
wait $API_PID
