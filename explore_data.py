import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load CSV ---
filename = "seattle-weather.csv"
print(f"Loading data from: {filename}")
df = pd.read_csv(filename)
print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns\n")

# --- Basic Exploration ---
print("First 5 rows:")
print(df.head(), "\n")

print("Data types:")
print(df.dtypes, "\n")

print("Missing values:")
print(df.isnull().sum(), "\n")

# --- Create binary target column ---
# 1 = rain, 0 = no rain
if 'weather' in df.columns:
    df['rain'] = df['weather'].apply(lambda x: 1 if 'rain' in str(x).lower() else 0)

print("Target distribution (rain):")
print(df['rain'].value_counts())
print(f"Percentage of rainy days: {df['rain'].mean()*100:.2f}%\n")

# --- Visualizations ---

# Convert date column to datetime
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month

# 1. Rain probability by month
plt.figure(figsize=(12, 6))
monthly_rain = df.groupby('month')['rain'].mean()
monthly_rain.plot(kind='bar', color='skyblue')
plt.title('Rain Probability by Month')
plt.xlabel('Month')
plt.ylabel('Probability of Rain')
plt.tight_layout()
plt.savefig('monthly_rain.png')
plt.show()

# 2. Correlation heatmap (numeric features only)
numeric_df = df.select_dtypes(include='number')
plt.figure(figsize=(6, 5))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix (Numeric Features)')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.show()
