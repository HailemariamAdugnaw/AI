# Seattle Weather Prediction Model

This project provides a complete machine learning solution for predicting weather conditions based on meteorological data from Seattle.

## 📊 Dataset Overview

The `seattle-weather.csv` file contains daily weather observations with the following features:

- **date**: Date of observation
- **precipitation**: Amount of precipitation in mm
- **temp_max**: Maximum temperature in °C
- **temp_min**: Minimum temperature in °C
- **wind**: Wind speed in m/s
- **weather**: Weather type (drizzle, rain, sun, snow, fog) - **TARGET VARIABLE**

The dataset contains approximately 4 years of daily weather data (1,461 records).

## 🚀 Quick Start with Google Colab

### Method 1: Use the Jupyter Notebook (Recommended)

1. **Open Google Colab**: Go to [https://colab.research.google.com/](https://colab.research.google.com/)

2. **Upload the Notebook**:
   - Click on `File` → `Upload notebook`
   - Select the `weather_prediction.ipynb` file from this repository

3. **Upload the Dataset**:
   - When you run the first code cell (Step 1 in the notebook), it will prompt you to upload a file
   - Click "Choose Files" and select `seattle-weather.csv`

4. **Run All Cells**:
   - Click `Runtime` → `Run all`
   - The notebook will automatically install dependencies, train models, and display results

5. **Explore the Results**:
   - View data visualizations
   - Compare different machine learning models
   - See prediction examples
   - Download the trained model

### Method 2: Use the Python Script

1. **Open Google Colab**: Go to [https://colab.research.google.com/](https://colab.research.google.com/)

2. **Create a New Notebook**:
   - Click on `File` → `New notebook`

3. **Upload Files**:
   ```python
   from google.colab import files
   uploaded = files.upload()
   # Upload both 'weather_prediction.py' and 'seattle-weather.csv'
   ```

4. **Run the Script**:
   ```python
   !python weather_prediction.py
   ```

## 📦 Project Files

```
AI/
├── seattle-weather.csv          # The weather dataset
├── weather_prediction.ipynb     # Complete Jupyter notebook for Google Colab
├── weather_prediction.py        # Python script version
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🔧 Dependencies

The project uses the following Python libraries:

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning models and tools
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization

All dependencies are automatically installed when you run the notebook in Google Colab.

## 🎯 What the Model Does

1. **Data Loading & Exploration**:
   - Loads the weather dataset
   - Displays statistical summaries
   - Visualizes data distributions

2. **Feature Engineering**:
   - Extracts date features (year, month, day, day of week, day of year)
   - Creates derived features (temperature range, freezing indicator)
   - Handles categorical encoding

3. **Model Training**:
   - Trains 6 different machine learning models:
     - Random Forest
     - Gradient Boosting
     - Logistic Regression
     - Decision Tree
     - K-Nearest Neighbors
     - Support Vector Machine

4. **Model Evaluation**:
   - Compares model accuracies
   - Displays confusion matrix
   - Shows classification report
   - Analyzes feature importance

5. **Prediction**:
   - Provides a function to predict weather for new data
   - Shows example predictions with probabilities

6. **Model Saving**:
   - Saves the best-performing model
   - Allows downloading for future use

## 📈 Expected Results

The models typically achieve the following accuracy ranges:

- **Random Forest**: 82-85%
- **Gradient Boosting**: 81-84%
- **Decision Tree**: 75-78%
- **Logistic Regression**: 78-80%
- **K-Nearest Neighbors**: 77-80%
- **Support Vector Machine**: 79-82%

*Note: Actual results may vary slightly due to random initialization.*

## 🔍 Key Features

### 1. Comprehensive Visualizations
- Weather distribution charts
- Feature correlation heatmaps
- Box plots for each feature
- Confusion matrix
- Model comparison charts
- Feature importance plots

### 2. Multiple Models
- Compare 6 different algorithms
- Automatic selection of the best model
- Detailed performance metrics

### 3. Easy Predictions
- Simple prediction function
- Probability estimates for each weather type
- Example predictions included

### 4. Model Persistence
- Save trained models
- Download for deployment
- Reuse in other applications

## 📝 How to Make Predictions

After training the model, you can make predictions using the `predict_weather()` function:

```python
# Example: Predict weather for given conditions
precipitation = 15.0  # mm
temp_max = 10.0      # °C
temp_min = 5.0       # °C
wind = 5.0           # m/s
month = 11           # November
day = 15

weather, probabilities = predict_weather(precipitation, temp_max, temp_min, wind, month, day)

print(f"Predicted Weather: {weather}")
print(f"Probabilities: {probabilities}")
```

## 🎓 Learning Outcomes

By using this project, you will learn:

- How to load and explore datasets in Python
- Data visualization techniques
- Feature engineering strategies
- Training multiple ML models
- Model evaluation and comparison
- Making predictions with trained models
- Saving and loading models

## 🛠️ Customization Ideas

You can extend this project by:

1. **Adding More Features**:
   - Include season information
   - Add rolling averages
   - Include historical weather patterns

2. **Trying Different Models**:
   - Neural Networks (using TensorFlow/Keras)
   - XGBoost
   - LightGBM
   - Ensemble methods

3. **Hyperparameter Tuning**:
   - Use GridSearchCV
   - Try RandomizedSearchCV
   - Implement cross-validation

4. **Deployment**:
   - Create a web app with Streamlit
   - Build a REST API with Flask
   - Deploy to cloud platforms

## 🐛 Troubleshooting

### Issue: "File not found" error
**Solution**: Make sure you've uploaded the CSV file when prompted in Step 1.

### Issue: "Module not found" error
**Solution**: Run the installation cell (Step 2) and wait for it to complete.

### Issue: Model accuracy is low
**Solution**: This is normal for weather prediction. Try:
- Adjusting model hyperparameters
- Adding more features
- Using ensemble methods

### Issue: Notebook runs slowly
**Solution**: 
- Use Google Colab's GPU/TPU (Runtime → Change runtime type → GPU)
- Reduce the number of models trained
- Use a smaller subset of data for testing

## 📚 Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Google Colab Guide](https://colab.research.google.com/notebooks/intro.ipynb)
- [Machine Learning Mastery](https://machinelearningmastery.com/)

## 📄 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Feel free to fork this project and add your own improvements!

## 📧 Support

If you encounter any issues or have questions, please open an issue in the repository.

---

**Happy Weather Predicting! 🌤️ 🌧️ ❄️**
