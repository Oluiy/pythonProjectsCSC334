from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'house_price_model.pkl')
model = joblib.load(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Get data from form
            overall_qual = int(request.form['overall_qual'])
            gr_liv_area = float(request.form['gr_liv_area'])
            garage_cars = int(request.form['garage_cars'])
            full_bath = int(request.form['full_bath'])
            year_built = int(request.form['year_built'])
            neighborhood = request.form['neighborhood']
            
            # Create DataFrame for model
            input_data = pd.DataFrame({
                'OverallQual': [overall_qual],
                'GrLivArea': [gr_liv_area],
                'GarageCars': [garage_cars],
                'FullBath': [full_bath],
                'YearBuilt': [year_built],
                'Neighborhood': [neighborhood]
            })
            
            # Predict
            prediction = model.predict(input_data)[0]
            prediction = f"{prediction:,.2f}"
            
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
