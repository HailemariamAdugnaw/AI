from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import joblib
import os
import numpy as np

# Initialize app
app = Flask(__name__, static_folder='.')

# Load trained model and scaler
model = joblib.load('weather_model.pkl')
_, _, _, _, scaler = joblib.load('weather_data.pkl')

# Get weather classes
weather_classes = sorted(model.classes_)
print(f"Loaded model can predict: {weather_classes}")

# Serve the main HTML file
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from JSON request
        data = request.get_json()
        
        precipitation = float(data['precipitation'])
        temp_max = float(data['temp_max'])
        temp_min = float(data['temp_min'])
        wind = float(data['wind'])

        # Prepare dataframe
        new_data = pd.DataFrame([[precipitation, temp_max, temp_min, wind]],
                                columns=['precipitation', 'temp_max', 'temp_min', 'wind'])
        
        # Scale features
        X_scaled = scaler.transform(new_data)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        
        # Create probability dictionary
        prob_dict = {weather_classes[i]: float(probabilities[i]) for i in range(len(weather_classes))}
        
        # Get confidence (probability of predicted class)
        confidence = float(probabilities[model.classes_.tolist().index(prediction)])
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': prob_dict
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# Run app
if __name__ == '__main__':
    app.run(debug=True)

