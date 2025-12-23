# Weather Prediction Model

A simple and accurate machine learning model to predict weather conditions based on meteorological features.

## Overview

This project implements a weather prediction model that classifies weather into categories (sun, rain, drizzle, snow, fog) based on four key features:
- **Precipitation**: Amount of rainfall/snowfall
- **Maximum Temperature**: Daily high temperature
- **Minimum Temperature**: Daily low temperature  
- **Wind Speed**: Wind speed measurement

## Features

✅ **Simple**: Uses only 4 input features (no complex date-based features)  
✅ **Single Algorithm**: Uses Random Forest Classifier for simplicity and accuracy  
✅ **Machine Learning**: Proper ML implementation with train/test split  
✅ **Accurate**: Achieves ~83% accuracy on test data  
✅ **Easy to Use**: Simple API for making predictions  

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the main script to train the model:

```bash
python weather_prediction.py
```

This will:
1. Load the weather data from `seattle-weather.csv`
2. Prepare features (precipitation, temp_max, temp_min, wind)
3. Split data into training and test sets
4. Train a Random Forest Classifier
5. Evaluate model performance
6. Save the trained model to `weather_model.pkl`

### Making Predictions

```python
from weather_prediction import load_model, predict_weather

# Load the trained model
model = load_model('weather_model.pkl')

# Make a prediction
weather = predict_weather(
    model,
    precipitation=10.0,  # mm
    temp_max=10.0,       # °C
    temp_min=5.0,        # °C
    wind=4.5             # m/s
)

print(f"Predicted weather: {weather}")  # Output: rain
```

## Model Performance

- **Accuracy**: ~83%
- **Most Important Feature**: Precipitation (63.4% importance)
- **Algorithm**: Random Forest Classifier with 100 trees

The model performs particularly well at predicting:
- Rain (97% precision, 91% recall)
- Sun (75% precision, 95% recall)

## Data

The model is trained on Seattle weather data from 2012-2015 containing:
- 1,461 daily weather records
- 5 weather categories: drizzle, rain, sun, snow, fog

## Project Structure

```
AI/
├── weather_prediction.py    # Main prediction script
├── requirements.txt          # Python dependencies
├── seattle-weather.csv       # Training data
├── weather_model.pkl         # Trained model (generated)
└── README.md                 # This file
```

## Why This Approach?

This implementation follows the requirement for a **simple, correct machine learning model**:

1. **No unnecessary features**: Removed day/month features that add complexity without significant accuracy improvement
2. **Single algorithm**: Uses only Random Forest (no ensemble of multiple algorithms)
3. **Pure ML**: Proper machine learning workflow with training, validation, and persistence
4. **Practical**: Easy to understand, maintain, and use

## License

Open source - feel free to use and modify for your projects.
