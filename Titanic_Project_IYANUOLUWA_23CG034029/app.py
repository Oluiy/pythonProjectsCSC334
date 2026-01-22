from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'titanic_survival_model.pkl')
model = joblib.load(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Get data from form
            pclass = int(request.form['pclass'])
            sex = request.form['sex']
            age = float(request.form['age'])
            fare = float(request.form['fare'])
            embarked = request.form['embarked']
            
            # Create DataFrame for model
            input_data = pd.DataFrame({
                'Pclass': [pclass],
                'Sex': [sex],
                'Age': [age],
                'Fare': [fare],
                'Embarked': [embarked]
            })
            
            # Predict
            pred = model.predict(input_data)[0]
            prediction = "Survived" if pred == 1 else "Did Not Survive"
            
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
