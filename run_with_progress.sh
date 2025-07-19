#!/bin/bash

# Run script for VintageOptics with Enhanced Progress Tracking

echo "Starting VintageOptics with Enhanced Progress Tracking..."

# Function to cleanup on exit
cleanup() {
    echo "Shutting down services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit
}

trap cleanup EXIT INT TERM

# Start backend API with progress tracking
echo "Starting backend API with progress tracking..."
python frontend_api_with_progress.py &
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
echo "VintageOptics is running with enhanced progress tracking!"
echo "Frontend: http://localhost:3000"
echo "Backend API: http://localhost:8000"
echo ""
echo "Features:"
echo "  ✓ Real-time progress updates"
echo "  ✓ Detailed processing stages"
echo "  ✓ Progress history"
echo "  ✓ Visual progress bar"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for processes
wait