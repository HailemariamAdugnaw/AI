"""
Seattle Weather Prediction Model
This script trains multiple machine learning models to predict weather types
based on meteorological features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import pickle
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*70)
print("Seattle Weather Prediction Model")
print("="*70)

# Load the dataset
print("\n[1/10] Loading dataset...")
df = pd.read_csv('seattle-weather.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nDataset Info:")
print(df.info())

print(f"\nMissing Values:")
print(df.isnull().sum())

# Explore the target variable
print("\n[2/10] Exploring target variable (weather)...")
print("\nWeather Types Distribution:")
print(df['weather'].value_counts())

# Visualize weather distribution
plt.figure(figsize=(10, 6))
df['weather'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Weather Types', fontsize=16, fontweight='bold')
plt.xlabel('Weather Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('weather_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: weather_distribution.png")
plt.close()

# Feature engineering
print("\n[3/10] Engineering features...")
df_processed = df.copy()

# Convert date to datetime and extract useful features
df_processed['date'] = pd.to_datetime(df_processed['date'])
df_processed['year'] = df_processed['date'].dt.year
df_processed['month'] = df_processed['date'].dt.month
df_processed['day'] = df_processed['date'].dt.day
df_processed['day_of_week'] = df_processed['date'].dt.dayofweek
df_processed['day_of_year'] = df_processed['date'].dt.dayofyear

# Create additional features
df_processed['temp_range'] = df_processed['temp_max'] - df_processed['temp_min']
df_processed['is_freezing'] = (df_processed['temp_min'] <= 0).astype(int)

# Drop the original date column
df_processed = df_processed.drop('date', axis=1)

print(f"Features engineered. New shape: {df_processed.shape}")
print(f"Feature columns: {df_processed.drop('weather', axis=1).columns.tolist()}")

# Prepare data
print("\n[4/10] Preparing data for training...")
X = df_processed.drop('weather', axis=1)
y = df_processed['weather']

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\nWeather type encoding:")
for i, weather_type in enumerate(label_encoder.classes_):
    print(f"  {weather_type}: {i}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled successfully!")

# Define models
print("\n[5/10] Defining models...")
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(kernel='rbf', random_state=42, probability=True)
}

print("Models defined:")
for model_name in models.keys():
    print(f"  - {model_name}")

# Train models
print("\n[6/10] Training models...")
print("="*70)

results = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy
    
    print(f"{model_name} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("-"*70)

print("\n" + "="*70)
print("All models trained successfully!")

# Find the best model
print("\n[7/10] Comparing models...")
best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]
best_model = models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"🎯 Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

# Visualize model comparison
plt.figure(figsize=(12, 6))
model_names = list(results.keys())
accuracies = list(results.values())

colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
bars = plt.bar(model_names, accuracies, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height*100:.2f}%',
             ha='center', va='bottom', fontweight='bold')

plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
plt.xlabel('Model', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.1)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("\nSaved: model_comparison.png")
plt.close()

# Detailed evaluation
print("\n[8/10] Evaluating best model...")
y_pred_best = best_model.predict(X_test_scaled)

print(f"\n{'='*70}")
print(f"Detailed Evaluation: {best_model_name}")
print(f"{'='*70}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred_best, 
                          target_names=label_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Weather', fontsize=12)
plt.ylabel('Actual Weather', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\nSaved: confusion_matrix.png")
plt.close()

# Feature importance
print("\n[9/10] Analyzing feature importance...")
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
    plt.title(f'Feature Importance - {best_model_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("Saved: feature_importance.png")
    plt.close()
    
    print("\nTop 5 Most Important Features:")
    print(feature_importance.head())
else:
    print(f"Feature importance not available for {best_model_name}")

# Save the model
print("\n[10/10] Saving the trained model...")

with open('weather_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Model and preprocessing objects saved:")
print("  - weather_model.pkl")
print("  - scaler.pkl")
print("  - label_encoder.pkl")

# Example predictions
print("\n" + "="*70)
print("Example Predictions")
print("="*70)

def predict_weather(precipitation, temp_max, temp_min, wind, month, day):
    """
    Predict weather based on input parameters
    """
    import datetime
    
    # Create a sample date for the given month and day
    year = 2023  # arbitrary year
    date = datetime.datetime(year, month, day)
    
    # Calculate derived features
    day_of_week = date.weekday()
    day_of_year = date.timetuple().tm_yday
    temp_range = temp_max - temp_min
    is_freezing = 1 if temp_min <= 0 else 0
    
    # Create feature array in the same order as training data
    features = np.array([[
        precipitation, temp_max, temp_min, wind,
        year, month, day, day_of_week, day_of_year,
        temp_range, is_freezing
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = best_model.predict(features_scaled)
    predicted_weather = label_encoder.inverse_transform(prediction)[0]
    
    # Get prediction probabilities if available
    if hasattr(best_model, 'predict_proba'):
        probabilities = best_model.predict_proba(features_scaled)[0]
        prob_dict = {label_encoder.classes_[i]: prob 
                    for i, prob in enumerate(probabilities)}
        return predicted_weather, prob_dict
    else:
        return predicted_weather, None

# Example 1: Rainy conditions
print("\n1. Rainy conditions:")
print("   Input: precipitation=15.0, temp_max=10.0, temp_min=5.0, wind=5.0, month=11, day=15")
weather, probs = predict_weather(15.0, 10.0, 5.0, 5.0, 11, 15)
print(f"   Predicted Weather: {weather}")
if probs:
    print("   Probabilities:")
    for w, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        print(f"     {w}: {p*100:.2f}%")

# Example 2: Sunny conditions
print("\n2. Sunny conditions:")
print("   Input: precipitation=0.0, temp_max=25.0, temp_min=15.0, wind=2.0, month=7, day=20")
weather, probs = predict_weather(0.0, 25.0, 15.0, 2.0, 7, 20)
print(f"   Predicted Weather: {weather}")
if probs:
    print("   Probabilities:")
    for w, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        print(f"     {w}: {p*100:.2f}%")

# Example 3: Snowy conditions
print("\n3. Snowy conditions:")
print("   Input: precipitation=10.0, temp_max=0.0, temp_min=-5.0, wind=3.0, month=1, day=10")
weather, probs = predict_weather(10.0, 0.0, -5.0, 3.0, 1, 10)
print(f"   Predicted Weather: {weather}")
if probs:
    print("   Probabilities:")
    for w, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        print(f"     {w}: {p*100:.2f}%")

print("\n" + "="*70)
print("Training Complete! 🎉")
print("="*70)
print("\nGenerated files:")
print("  - weather_distribution.png")
print("  - model_comparison.png")
print("  - confusion_matrix.png")
if hasattr(best_model, 'feature_importances_'):
    print("  - feature_importance.png")
print("  - weather_model.pkl")
print("  - scaler.pkl")
print("  - label_encoder.pkl")
print("\nYou can now use the saved model for predictions!")
