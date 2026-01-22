from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'wine_cultivar_model.pkl')
model = joblib.load(model_path)

# Mapping target classification (0, 1, 2) to readable cultivars (1, 2, 3)
cultivar_map = {
    0: "Cultivar 1",
    1: "Cultivar 2",
    2: "Cultivar 3"
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Get data from form
            alcohol = float(request.form['alcohol'])
            magnesium = float(request.form['magnesium'])
            flavanoids = float(request.form['flavanoids'])
            color_intensity = float(request.form['color_intensity'])
            hue = float(request.form['hue'])
            proline = float(request.form['proline'])
            
            # Create DataFrame for model
            input_data = pd.DataFrame({
                'alcohol': [alcohol],
                'magnesium': [magnesium],
                'flavanoids': [flavanoids],
                'color_intensity': [color_intensity],
                'hue': [hue],
                'proline': [proline]
            })
            
            # Predict
            pred_idx = model.predict(input_data)[0]
            prediction = cultivar_map.get(pred_idx, f"Unknown ({pred_idx})")
            
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
