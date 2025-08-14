#!/bin/bash

# Script to start both Flask backend and Next.js frontend servers

echo "Starting Video Generation Servers..."
echo "=================================="

# Check if Python dependencies are installed
echo "Checking Python dependencies..."
if ! python3 -c "import flask" 2>/dev/null; then
    echo "Installing Flask and dependencies..."
    pip install flask flask-cors
fi

# Start Flask backend server
echo "Starting Flask API server on port 5000..."
python3 api_server.py &
FLASK_PID=$!
echo "Flask server PID: $FLASK_PID"

# Wait for Flask to start
sleep 3

# Start Next.js frontend
echo "Starting Next.js frontend on port 3001..."
cd frontend
npm run dev &
NEXT_PID=$!
echo "Next.js server PID: $NEXT_PID"

echo ""
echo "=================================="
echo "Servers are running!"
echo "Flask API: http://localhost:5000"
echo "Frontend: http://localhost:3001"
echo "Upload page: http://localhost:3001/upload"
echo ""
echo "Press Ctrl+C to stop both servers"
echo "=================================="

# Function to handle cleanup on exit
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $FLASK_PID 2>/dev/null
    kill $NEXT_PID 2>/dev/null
    echo "Servers stopped."
    exit 0
}

# Set up trap to call cleanup on Ctrl+C
trap cleanup INT

# Wait for both processes
wait $FLASK_PID $NEXT_PID