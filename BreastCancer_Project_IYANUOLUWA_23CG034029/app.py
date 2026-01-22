from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
import numpy as np

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'breast_cancer_model.pkl')
model = joblib.load(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Get data from form
            radius_mean = float(request.form['radius_mean'])
            texture_mean = float(request.form['texture_mean'])
            perimeter_mean = float(request.form['perimeter_mean'])
            area_mean = float(request.form['area_mean'])
            smoothness_mean = float(request.form['smoothness_mean'])
            
            # Create DataFrame for model
            # Note: Features must match training names: 'mean radius', etc.
            input_data = pd.DataFrame({
                'mean radius': [radius_mean],
                'mean texture': [texture_mean],
                'mean perimeter': [perimeter_mean],
                'mean area': [area_mean],
                'mean smoothness': [smoothness_mean]
            })
            
            # Predict
            # 0 = Malignant, 1 = Benign in this dataset (sklearn default)
            # Use data.target_names or check docs. 
            # In breast_cancer: 0 = 'malignant', 1 = 'benign'
            pred_val = model.predict(input_data)[0]
            
            if pred_val == 0:
                prediction = "Malignant"
            else:
                prediction = "Benign"
            
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
