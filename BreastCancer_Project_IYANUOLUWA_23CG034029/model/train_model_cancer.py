import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# 1. Load the dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target

# Corrected features based on sklearn dataset names
# 'radius_mean' -> 'mean radius'
# 'texture_mean' -> 'mean texture'
# 'perimeter_mean' -> 'mean perimeter'
# 'area_mean' -> 'mean area'
# 'smoothness_mean' -> 'mean smoothness'
selected_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
target = 'diagnosis'

print("Data Preview:")
print(df[selected_features].head())

# 2. Data Preprocessing setup
X = df[selected_features]
y = df[target]

# Pipeline: Scale then model
# StandardScaler is important for Logistic Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(random_state=42))
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
print(classification_report(y_test, y_pred, target_names=data.target_names))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 5. Save the trained model
model_filename = 'breast_cancer_model.pkl'
joblib.dump(pipeline, model_filename)
print(f"Model saved to {model_filename}")

# 6. Reload demonstration
print("Reloading model to verify...")
loaded_model = joblib.load(model_filename)
sample_pred = loaded_model.predict(X_test.iloc[[0]])
# target_names[0] = 'malignant', target_names[1] = 'benign'
print(f"Prediction for first test sample: {data.target_names[sample_pred[0]]}")
