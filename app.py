from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

# Initialize app
app = Flask(__name__)

# Load trained model and scaler
try:
    model = joblib.load('weather_model.pkl')
    scaler = joblib.load('weather_scaler.pkl')
    print("Model and scaler loaded successfully!")
except FileNotFoundError:
    print("ERROR: Model files not found. Please run train_models.py first!")
    model = None
    scaler = None

# Home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Prediction API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first by running train_models.py'
            }), 500
        
        # Get data from JSON request
        data = request.get_json()
        
        # Extract features
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
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0]

        # Find the confidence (probability of predicted class)
        max_prob_idx = prob.argmax()
        confidence = float(prob[max_prob_idx]) * 100

        # Format prediction with probabilities for each class
        prob_dict = {cls: float(p*100) for cls, p in zip(model.classes_, prob)}
        
        # Map weather to icons and recommendations
        weather_info = get_weather_info(pred, confidence, precipitation, temp_max, temp_min, wind)
        
        return jsonify({
            'prediction': pred,
            'confidence': round(confidence, 2),
            'probabilities': prob_dict,
            'icon': weather_info['icon'],
            'recommendation': weather_info['recommendation']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def get_weather_info(weather, confidence, precip, temp_max, temp_min, wind):
    """Map weather prediction to icon and recommendation"""
    
    weather_lower = weather.lower()
    
    # Icon mapping
    icon_map = {
        'rain': 'fas fa-cloud-showers-heavy',
        'drizzle': 'fas fa-cloud-rain',
        'snow': 'fas fa-snowflake',
        'fog': 'fas fa-smog',
        'sun': 'fas fa-sun'
    }
    
    # Recommendation mapping
    recommendation_map = {
        'rain': 'Carry an umbrella and avoid outdoor plans. Stay dry!',
        'drizzle': 'Take a light jacket and umbrella just in case.',
        'snow': 'Dress warmly and watch for icy conditions. Drive carefully!',
        'fog': 'Drive carefully with reduced visibility. Use fog lights.',
        'sun': 'Perfect day for outdoor activities! Don\'t forget your sunscreen.'
    }
    
    # Get icon and recommendation, with defaults
    icon = icon_map.get(weather_lower, 'fas fa-cloud-sun')
    recommendation = recommendation_map.get(weather_lower, 'Good day for most outdoor activities.')
    
    return {
        'icon': icon,
        'recommendation': recommendation
    }

# Run app
if __name__ == '__main__':
    import os
    
    print("Starting Weather Prediction Flask App...")
    print("Make sure you have trained the model first by running: python train_models.py")
    
    # Use debug mode only in development, not in production
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
