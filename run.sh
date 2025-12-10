#!/bin/bash
# Weather Prediction App - Quick Start Script

echo "================================"
echo "Weather Prediction AI Application"
echo "================================"
echo ""

# Check if model files exist
if [ ! -f "weather_model.pkl" ] || [ ! -f "weather_scaler.pkl" ]; then
    echo "⚠️  Model files not found!"
    echo "Training the model first..."
    echo ""
    python train_models.py
    if [ $? -ne 0 ]; then
        echo "❌ Error: Model training failed!"
        exit 1
    fi
    echo ""
    echo "✅ Model trained successfully!"
    echo ""
fi

# Start the Flask app
echo "🚀 Starting Weather Prediction Web Application..."
echo ""
echo "📍 The app will be available at: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Set Flask environment to development for local testing
export FLASK_ENV=development

python app.py
