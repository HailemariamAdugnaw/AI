import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # <-- for saving preprocessed data

# --- Load CSV ---
filename = "seattle-weather.csv"

print(f"Loading data from: {filename}")
df = pd.read_csv(filename)
print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

# --- Preprocessing ---

# 1. Create binary target 'rain' from 'weather' column
if 'weather' in df.columns:
    df['rain'] = df['weather'].apply(lambda x: 1 if 'rain' in str(x).lower() else 0)
    df = df.drop('weather', axis=1)

# 2. Drop 'date' column
if 'date' in df.columns:
    df = df.drop('date', axis=1)

# 3. Handle missing values
df = df.dropna()

# 4. Split features and target
X = df.drop('rain', axis=1)
y = df['rain']

# 5. Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Save preprocessed data
joblib.dump((X_train, X_test, y_train, y_test, scaler), 'weather_data.pkl')
print("Preprocessed data saved to 'weather_data.pkl'")

print("Preprocessing done!")
print(f"Features shape: {X_train.shape}, Target shape: {y_train.shape}")
print(f"Number of rainy days in train set: {y_train.sum()} / {y_train.shape[0]}")
print(f"Number of rainy days in test set: {y_test.sum()} / {y_test.shape[0]}")
