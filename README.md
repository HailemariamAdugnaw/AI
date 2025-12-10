# Weather Prediction AI Application

A Flask-based web application that predicts weather conditions using machine learning.

## Features
- Predicts 5 weather types: Rain, Drizzle, Snow, Fog, Sun
- Uses Random Forest Classifier for accurate predictions
- Beautiful, responsive web interface with animated gradients
- Real-time predictions based on weather parameters
- 80.5% model accuracy

## Quick Start

### Option 1: Use the run script (Recommended)
```bash
# Make the script executable (first time only)
chmod +x run.sh

# Run the application
./run.sh
```

The script will automatically:
- Train the model if not already trained
- Start the Flask web server
- Open the app at http://localhost:5000

### Option 2: Manual steps

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Train the Model
Before running the application, you need to train the machine learning model:

```bash
python train_models.py
```

This will:
- Load the Seattle weather dataset
- Train a Random Forest model
- Save the model to `weather_model.pkl`
- Save the scaler to `weather_scaler.pkl`
- Display model accuracy and metrics
- Generate a confusion matrix

#### 3. Run the Application
Start the Flask web server:

```bash
python app.py
```

The application will be available at: `http://localhost:5000`

## Usage

1. Open your web browser and navigate to `http://localhost:5000`
2. Enter the weather parameters:
   - **Precipitation (mm)**: Amount of rainfall/precipitation
   - **Max Temperature (°C)**: Maximum temperature
   - **Min Temperature (°C)**: Minimum temperature  
   - **Wind Speed (km/h)**: Wind speed
3. Click "Generate Prediction" to get the weather forecast
4. View the prediction with confidence level and recommendations

### Example Inputs

**Sunny Day:**
- Precipitation: 0.0 mm
- Max Temp: 30°C
- Min Temp: 20°C
- Wind: 10 km/h

**Rainy Day:**
- Precipitation: 15.5 mm
- Max Temp: 18°C
- Min Temp: 12°C
- Wind: 25 km/h

**Snowy Day:**
- Precipitation: 5.0 mm
- Max Temp: -2°C
- Min Temp: -8°C
- Wind: 15 km/h

## Project Structure

```
.
├── app.py                  # Flask web application
├── train_models.py         # Model training script
├── index.html              # Original HTML (for reference)
├── templates/
│   └── index.html         # Flask template
├── requirements.txt        # Python dependencies
├── seattle-weather.csv     # Weather dataset
├── weather_model.pkl       # Trained model (generated)
├── weather_scaler.pkl      # Feature scaler (generated)
└── confusion_matrix.png    # Model performance visualization (generated)
```

## Files Overview

### app.py
Flask web application that:
- Serves the web interface
- Provides `/predict` API endpoint
- Loads the trained model
- Makes predictions based on user input

### train_models.py
Training script that:
- Loads and preprocesses the Seattle weather data
- Trains a Random Forest classifier
- Evaluates model performance
- Saves the trained model and scaler

### templates/index.html
Beautiful, responsive web interface with:
- Animated gradient background
- Real-time form validation
- Weather icons and predictions
- Confidence meter visualization

## API Endpoint

### POST /predict
Predicts weather based on input parameters.

**Request Body (JSON):**
```json
{
  "precipitation": 5.2,
  "temp_max": 28.5,
  "temp_min": 15.3,
  "wind": 12.4
}
```

**Response (JSON):**
```json
{
  "prediction": "sun",
  "confidence": 85.67,
  "probabilities": {
    "drizzle": 2.5,
    "fog": 1.2,
    "rain": 5.3,
    "snow": 0.3,
    "sun": 85.67
  },
  "icon": "fas fa-sun",
  "recommendation": "Perfect day for outdoor activities! Don't forget your sunscreen."
}
```

## Troubleshooting

### Error: Model files not found
If you see this error, make sure to run `python train_models.py` first to generate the model files.

### Port already in use
If port 5000 is already in use, you can modify the port in `app.py`:
```python
app.run(debug=debug_mode, host='0.0.0.0', port=5001)  # Change port to 5001
```

### Running in Production
For production deployment, make sure to:
1. Do NOT set `FLASK_ENV=development`
2. Use a production WSGI server like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```
3. Never run with `debug=True` in production

## Model Performance

The Random Forest model achieves:
- High accuracy on the Seattle weather dataset
- Balanced predictions across all weather types
- Reliable confidence scores

Check `confusion_matrix.png` after training to see detailed performance metrics.

## Credits

Dataset: Seattle Weather Data
Model: Random Forest Classifier (scikit-learn)
Web Framework: Flask
