#!/bin/bash

# Frontend setup script for VintageOptics

echo "Setting up VintageOptics Frontend..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Node.js is not installed. Please install Node.js first."
    exit 1
fi

# Navigate to frontend directory
cd frontend

# Install dependencies
echo "Installing frontend dependencies..."
npm install

echo "Frontend setup complete!"
echo ""
echo "To start the frontend development server, run:"
echo "  cd frontend && npm start"
echo ""
echo "The frontend will be available at http://localhost:3000"
