#!/bin/bash

# Install frontend dependencies for VintageOptics

echo "Installing VintageOptics Frontend dependencies..."

# Navigate to frontend directory
cd /Users/rohanvinaik/VintageOptics/frontend

# Create package-lock.json if it doesn't exist
if [ ! -f "package-lock.json" ]; then
    echo "Creating package-lock.json..."
    npm i --package-lock-only
fi

# Install dependencies
echo "Installing dependencies..."
npm install

# Fix any vulnerabilities
echo "Checking for vulnerabilities..."
npm audit fix || true

echo ""
echo "Installation complete!"
echo "To start the frontend, run: npm start"
