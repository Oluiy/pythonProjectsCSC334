import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# 1. Load the dataset
# Using the sklearn Wine dataset which matches the UCI Machine Learning Repository dataset associated with the project description.
wine_data = load_wine()
df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
df['target'] = wine_data.target

# Target Mapping: 0->1, 1->2, 2->3 (Matches Cultivar 1, 2, 3)
# (Though commonly referred to as Class 0, 1, 2 in sklearn)

# Selected features (6 selected from recommended list)
# Recommended: alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols, flavanoids, color_intensity, hue, od280/od315_of_diluted_wines, proline
selected_features = ['alcohol', 'magnesium', 'flavanoids', 'color_intensity', 'hue', 'proline']
target = 'target'

print("Data Preview:")
print(df[selected_features].head())

# 2. Data Preprocessing setup
X = df[selected_features]
y = df[target]

# Pipeline: Scale then model
# Scaling is mandatory for SVM and generally good for chemical data with different units
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf', probability=True, random_state=42)) 
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Train the model
print("Training model...")
pipeline.fit(X_train, y_train)

# 4. Evaluate the model
print("Evaluating model...")
y_pred = pipeline.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=wine_data.target_names))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 5. Save the trained model
model_filename = 'wine_cultivar_model.pkl'
joblib.dump(pipeline, model_filename)
print(f"Model saved to {model_filename}")

# 6. Reload demonstration
print("Reloading model to verify...")
loaded_model = joblib.load(model_filename)
sample_pred = loaded_model.predict(X_test.iloc[[0]])
print(f"Prediction for first test sample (Class index): {sample_pred[0]}")
