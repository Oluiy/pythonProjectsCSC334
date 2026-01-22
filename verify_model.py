import joblib
import pandas as pd
import numpy as np
import os

model_path = 'HousePrice_Project_IYANUOLUWA_23CG034029/model/house_price_model.pkl'

try:
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        exit(1)

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    print("Model loaded successfully.")

    # Create dummy input data matching the features
    # 'OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt', 'Neighborhood'
    input_data = pd.DataFrame({
        'OverallQual': [7],
        'GrLivArea': [1500],
        'GarageCars': [2],
        'FullBath': [2],
        'YearBuilt': [2000],
        'Neighborhood': ['CollgCr']
    })

    print("Predicting...")
    prediction = model.predict(input_data)
    print(f"Prediction: {prediction}")

except Exception as e:
    print(f"Error caught: {e}")
