import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load preprocessed data
X_train, X_test, y_train, y_test, scaler = joblib.load('weather_data.pkl')

# Train multi-class model
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Get unique weather types for confusion matrix labels
weather_types = sorted(pd.Series(y_train).unique())

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=weather_types)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=weather_types, yticklabels=weather_types)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Weather Types')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Save model
joblib.dump(model, 'weather_model.pkl')
print("\nModel saved as weather_model.pkl")
print(f"Model can predict: {weather_types}")

