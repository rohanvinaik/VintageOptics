#!/bin/bash

# Run VintageOptics with GUI using actual processing

echo "Starting VintageOptics with REAL processing..."

# Set Python path to include VintageOptics source
export PYTHONPATH="/Users/rohanvinaik/VintageOptics/VintageOptics/src:$PYTHONPATH"

# Function to cleanup on exit
cleanup() {
    echo "Shutting down services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit
}

trap cleanup EXIT INT TERM

# Start backend API with actual VintageOptics
echo "Starting backend API with VintageOptics integration..."
cd /Users/rohanvinaik/VintageOptics
python frontend_api_real.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "Backend failed to start!"
    echo "Check if VintageOptics modules are properly installed"
    exit 1
fi

# Check API status
echo "Checking API status..."
curl -s http://localhost:8000/api/status | python -m json.tool

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
echo "Press Ctrl+C to stop all services"

# Wait for processes
wait
