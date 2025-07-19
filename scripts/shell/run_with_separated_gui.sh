#!/bin/bash

# Run script for VintageOptics with GUI

echo "Starting VintageOptics with GUI (Separated Workflows)..."

# Function to cleanup on exit
cleanup() {
    echo "Shutting down services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit
}

trap cleanup EXIT INT TERM

# Start backend API with separated endpoints
echo "Starting backend API..."
python frontend_api_separated.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "Backend failed to start!"
    exit 1
fi

echo "Backend API running on http://localhost:8000"

# Start frontend
echo "Starting frontend..."
cd frontend
npm start &
FRONTEND_PID=$!

echo ""
echo "VintageOptics is running!"
echo "Frontend: http://localhost:3000"
echo "Backend API: http://localhost:8000"
echo ""
echo "Two separate workflows:"
echo "  - Restore & Enhance: Clean up vintage lens photos"
echo "  - Add Vintage Effects: Apply vintage character to modern photos"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for processes
wait