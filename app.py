from flask import Flask, render_template, request
import pandas as pd
import joblib

# Initialize app
app = Flask(__name__)

# Load trained model and scaler
model = joblib.load('weather_model.pkl')
_, _, _, _, scaler = joblib.load('weather_data.pkl')

# Home page
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get data from form
        precipitation = float(request.form['precipitation'])
        temp_max = float(request.form['temp_max'])
        temp_min = float(request.form['temp_min'])
        wind = float(request.form['wind'])

        # Prepare dataframe
        new_data = pd.DataFrame([[precipitation, temp_max, temp_min, wind]],
                                columns=['precipitation', 'temp_max', 'temp_min', 'wind'])
        
        # Scale features
        X_scaled = scaler.transform(new_data)
        
        # Predict
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1]

        prediction = f"Rain: {'Yes' if pred==1 else 'No'}, Probability: {prob*100:.2f}%"

    return render_template('index.html', prediction=prediction)

# Run app
if __name__ == '__main__':
    app.run(debug=True)

