# Using the Trained Weather Prediction Model

After training the model, you can use it in your own applications. This guide shows you how.

## Loading the Saved Model

```python
import pickle
import numpy as np

# Load the trained model
with open('weather_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

print("Model loaded successfully!")
```

## Making Predictions

### Simple Prediction Function

```python
import datetime

def predict_weather(precipitation, temp_max, temp_min, wind, month, day):
    """
    Predict weather based on meteorological features
    
    Args:
        precipitation: Amount of precipitation in mm
        temp_max: Maximum temperature in °C
        temp_min: Minimum temperature in °C
        wind: Wind speed in m/s
        month: Month (1-12)
        day: Day of month (1-31)
    
    Returns:
        predicted_weather: The predicted weather type
        probabilities: Probability for each weather type
    """
    # Create a date for feature extraction
    year = 2023  # arbitrary year
    date = datetime.datetime(year, month, day)
    
    # Calculate derived features
    day_of_week = date.weekday()
    day_of_year = date.timetuple().tm_yday
    temp_range = temp_max - temp_min
    is_freezing = 1 if temp_min <= 0 else 0
    
    # Create feature array (must match training order)
    features = np.array([[
        precipitation, temp_max, temp_min, wind,
        year, month, day, day_of_week, day_of_year,
        temp_range, is_freezing
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)
    predicted_weather = label_encoder.inverse_transform(prediction)[0]
    
    # Get probabilities
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features_scaled)[0]
        prob_dict = {label_encoder.classes_[i]: prob 
                    for i, prob in enumerate(probabilities)}
        return predicted_weather, prob_dict
    
    return predicted_weather, None
```

### Example Usage

```python
# Predict weather for a rainy day
weather, probs = predict_weather(
    precipitation=15.0,
    temp_max=10.0,
    temp_min=5.0,
    wind=5.0,
    month=11,
    day=15
)

print(f"Predicted Weather: {weather}")
print("\nProbabilities:")
for weather_type, probability in sorted(probs.items(), key=lambda x: x[1], reverse=True):
    print(f"  {weather_type}: {probability*100:.2f}%")
```

Output:
```
Predicted Weather: rain

Probabilities:
  rain: 99.00%
  snow: 1.00%
  drizzle: 0.00%
  fog: 0.00%
  sun: 0.00%
```

## Batch Predictions

```python
import pandas as pd

def predict_batch(data_df):
    """
    Make predictions for multiple records
    
    Args:
        data_df: DataFrame with columns: 
                 precipitation, temp_max, temp_min, wind, month, day
    
    Returns:
        DataFrame with predictions and probabilities
    """
    predictions = []
    
    for _, row in data_df.iterrows():
        weather, probs = predict_weather(
            row['precipitation'],
            row['temp_max'],
            row['temp_min'],
            row['wind'],
            row['month'],
            row['day']
        )
        
        result = {
            'predicted_weather': weather,
            **probs
        }
        predictions.append(result)
    
    return pd.DataFrame(predictions)

# Example usage
test_data = pd.DataFrame({
    'precipitation': [0.0, 15.0, 5.0],
    'temp_max': [25.0, 10.0, 5.0],
    'temp_min': [15.0, 5.0, -2.0],
    'wind': [2.0, 5.0, 3.0],
    'month': [7, 11, 1],
    'day': [20, 15, 10]
})

results = predict_batch(test_data)
print(results)
```

## Building a Simple Web API

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load models once at startup
with open('weather_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for weather prediction
    
    Expected JSON:
    {
        "precipitation": 15.0,
        "temp_max": 10.0,
        "temp_min": 5.0,
        "wind": 5.0,
        "month": 11,
        "day": 15
    }
    """
    data = request.get_json()
    
    weather, probs = predict_weather(
        data['precipitation'],
        data['temp_max'],
        data['temp_min'],
        data['wind'],
        data['month'],
        data['day']
    )
    
    return jsonify({
        'predicted_weather': weather,
        'probabilities': probs
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### Testing the API

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "precipitation": 15.0,
    "temp_max": 10.0,
    "temp_min": 5.0,
    "wind": 5.0,
    "month": 11,
    "day": 15
  }'
```

## Integration with Streamlit

```python
import streamlit as st
import pickle
import numpy as np
import datetime

# Load models
@st.cache_resource
def load_models():
    with open('weather_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, scaler, label_encoder

model, scaler, label_encoder = load_models()

# Streamlit UI
st.title('🌤️ Weather Prediction App')

st.sidebar.header('Input Weather Features')

precipitation = st.sidebar.slider('Precipitation (mm)', 0.0, 50.0, 5.0)
temp_max = st.sidebar.slider('Max Temperature (°C)', -10.0, 40.0, 20.0)
temp_min = st.sidebar.slider('Min Temperature (°C)', -15.0, 30.0, 10.0)
wind = st.sidebar.slider('Wind Speed (m/s)', 0.0, 15.0, 3.0)
month = st.sidebar.slider('Month', 1, 12, 6)
day = st.sidebar.slider('Day', 1, 31, 15)

if st.sidebar.button('Predict Weather'):
    weather, probs = predict_weather(precipitation, temp_max, temp_min, wind, month, day)
    
    st.success(f'Predicted Weather: **{weather.upper()}**')
    
    st.subheader('Prediction Probabilities')
    st.bar_chart(probs)
```

Run with:
```bash
streamlit run app.py
```

## Important Notes

1. **Feature Order**: The features must be in the exact same order as training:
   - precipitation, temp_max, temp_min, wind, year, month, day, day_of_week, day_of_year, temp_range, is_freezing

2. **Scaling**: Always scale features using the saved scaler before prediction

3. **Date Features**: The year value (2023) is arbitrary since the model learns patterns based on month/day/day_of_year

4. **Weather Types**: The model can predict:
   - drizzle
   - fog
   - rain
   - snow
   - sun

5. **Model Performance**: Expected accuracy is around 84% based on testing

## Troubleshooting

**Issue**: "ValueError: X has different number of features"
- Make sure you're passing all 11 features in the correct order
- Check that you're calculating derived features (temp_range, is_freezing, etc.)

**Issue**: "Predictions don't make sense"
- Verify that features are scaled using the loaded scaler
- Check that input values are in the correct units (°C, mm, m/s)

**Issue**: "Model file not found"
- Ensure the .pkl files are in the same directory as your script
- Use absolute paths if needed

## Further Development

Ideas for extending the model:

1. **Add more features**: humidity, pressure, cloud cover
2. **Use time series**: Include previous day's weather
3. **Ensemble methods**: Combine multiple models
4. **Deep learning**: Try LSTM or other neural networks
5. **Real-time data**: Integrate with weather APIs
6. **Mobile app**: Create a mobile interface
7. **Alerts**: Send notifications for specific weather predictions

Happy predicting! 🌈
