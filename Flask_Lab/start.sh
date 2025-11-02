#!/bin/bash

# Simple script to start the Iris Flower Classifier application

echo "🌸 Starting Iris Flower Classifier..."
echo ""

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please run this script from the Flask_Lab directory."
    exit 1
fi

# Check if model files exist
if [ ! -f "model.pkl" ] || [ ! -f "scaler.pkl" ]; then
    echo "⚠️  Model files not found. Running prepare_model.py first..."
    python prepare_model.py
    echo ""
fi

# Check if dependencies are installed
echo "📦 Checking dependencies..."
pip show Flask > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    echo ""
fi

echo "✅ All checks passed!"
echo ""
echo "🚀 Starting Flask application..."
echo ""
echo "📱 Access the application at:"
echo "   → http://localhost:5000"
echo "   → http://127.0.0.1:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "----------------------------------------"
echo ""

# Start the application
python app.py
