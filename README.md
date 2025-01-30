# Airplane-Price-Prediction

# Airplane Price Prediction

## Overview
This project aims to predict the price of an airplane based on various features using machine learning techniques. The dataset used is `airplane_price_dataset.csv`, containing crucial information affecting airplane prices.

## Dataset Information
The dataset consists of the following columns:
- **Model**: The model of the airplane.
- **Manufacturer**: The company that manufactured the airplane.
- **Year**: The year of manufacture.
- **Engine Type**: The type of engine used.
- **Horsepower**: The power output of the engine.
- **Max Speed**: Maximum speed the airplane can achieve.
- **Range**: Maximum distance the airplane can travel.
- **Seating Capacity**: Number of passengers it can accommodate.
- **Price**: The target variable representing the price of the airplane.

## Step-by-Step Guide

### 1. Data Preprocessing
- Load the dataset using Pandas.
- Handle missing values (if any).
- Encode categorical variables such as `Model`, `Manufacturer`, and `Engine Type` using Label Encoding or One-Hot Encoding.
- Feature scaling using StandardScaler or MinMaxScaler.

### 2. Exploratory Data Analysis (EDA)
- Analyze data distribution using histograms and box plots.
- Compute correlations between features and the target variable.
- Visualize key features affecting airplane prices using scatter plots and bar charts.

### 3. Model Building
- Split the dataset into training and testing sets (e.g., 80-20 split).
- Train machine learning models such as:
  - Linear Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting (XGBoost, LightGBM)
- Optimize hyperparameters using GridSearchCV or RandomizedSearchCV.

### 4. Model Evaluation
- Use performance metrics:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R-squared (R²) Score
- Compare different models to choose the best-performing one.

### 5. Deployment
- Save the trained model using joblib or pickle.
- Create a Flask or FastAPI application for model inference.
- Deploy the application on cloud platforms such as AWS, GCP, or Heroku.

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/airplane-price-prediction.git
cd airplane-price-prediction
pip install -r requirements.txt
```

## Usage
Run the Jupyter Notebook for training:
```bash
jupyter notebook Airplane_Price_Prediction.ipynb
```

For model inference:
```python
import joblib
model = joblib.load('airplane_price_model.pkl')
prediction = model.predict([[2020, 2, 1500, 800, 5000, 10]])
print(prediction)
```

## Contributing
Feel free to contribute by submitting issues or pull requests.

## License
This project is licensed under the MIT License.
