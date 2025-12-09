"""
Weather Prediction Model
Predicts weather based on precipitation, wind, max temperature, and min temperature.
Uses a single machine learning algorithm for simplicity and accuracy.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

def load_data(filepath='seattle-weather.csv'):
    """Load weather data from CSV file."""
    df = pd.read_csv(filepath)
    return df

def prepare_features(df):
    """
    Prepare features for prediction.
    Uses only: precipitation, wind, temp_max, temp_min
    No date-based features (day, month) as requested.
    """
    # Select only the required features
    features = ['precipitation', 'temp_max', 'temp_min', 'wind']
    X = df[features]
    y = df['weather']
    
    return X, y

def train_model(X_train, y_train):
    """
    Train a Random Forest Classifier.
    Using a single algorithm for simplicity as requested.
    Random Forest is robust and provides good accuracy.
    """
    # Initialize the model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print evaluation metrics
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return accuracy, y_pred

def save_model(model, filename='weather_model.pkl'):
    """Save the trained model to a file."""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {filename}")

def load_model(filename='weather_model.pkl'):
    """Load a trained model from a file."""
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_weather(model, precipitation, temp_max, temp_min, wind):
    """
    Predict weather based on input parameters.
    
    Parameters:
    - precipitation: precipitation amount
    - temp_max: maximum temperature
    - temp_min: minimum temperature
    - wind: wind speed
    
    Returns:
    - predicted weather condition
    """
    # Create input DataFrame with correct feature names to avoid warning
    input_data = pd.DataFrame([[precipitation, temp_max, temp_min, wind]], 
                              columns=['precipitation', 'temp_max', 'temp_min', 'wind'])
    prediction = model.predict(input_data)
    return prediction[0]

def main():
    """Main function to train and evaluate the model."""
    print("Weather Prediction Model")
    print("=" * 50)
    
    # Load data
    print("\n1. Loading data...")
    df = load_data()
    print(f"   Loaded {len(df)} records")
    print(f"   Weather categories: {df['weather'].unique()}")
    
    # Prepare features
    print("\n2. Preparing features...")
    X, y = prepare_features(df)
    print(f"   Features used: {list(X.columns)}")
    print(f"   Feature shape: {X.shape}")
    
    # Split data
    print("\n3. Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Train model
    print("\n4. Training Random Forest Classifier...")
    model = train_model(X_train, y_train)
    print("   Model trained successfully!")
    
    # Evaluate model
    print("\n5. Evaluating model...")
    accuracy, y_pred = evaluate_model(model, X_test, y_test)
    
    # Feature importance
    print("\n6. Feature Importance:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.to_string(index=False))
    
    # Save model
    print("\n7. Saving model...")
    save_model(model)
    
    # Example predictions
    print("\n8. Example Predictions:")
    print("-" * 50)
    examples = [
        (0.0, 15.0, 8.0, 3.0, "Clear day (no precipitation, moderate temp)"),
        (10.0, 10.0, 5.0, 4.5, "Rainy day (high precipitation)"),
        (5.0, 1.0, -3.0, 3.0, "Snowy day (precipitation + cold)")
    ]
    
    for precip, tmax, tmin, wind_speed, description in examples:
        prediction = predict_weather(model, precip, tmax, tmin, wind_speed)
        print(f"{description}")
        print(f"  Input: precipitation={precip}, temp_max={tmax}, temp_min={tmin}, wind={wind_speed}")
        print(f"  Predicted weather: {prediction}")
        print()
    
    print("=" * 50)
    print("Model training complete!")
    print(f"Final accuracy: {accuracy:.4f}")
    
    return model

if __name__ == "__main__":
    model = main()
