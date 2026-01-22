import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Selected features (6 out of recommended 9)
selected_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt', 'Neighborhood']
target = 'SalePrice'

# Load the dataset
# Look for train.csv in the project root (one level up from this script in 'model/')
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, 'train.csv')

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print(f"Dataset loaded from {data_path}")
else:
    print(f"train.csv not found at {data_path}. Generating synthetic dataset for demonstration...")
    # Generate synthetic data

    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'OverallQual': np.random.randint(1, 10, n_samples),
        'GrLivArea': np.random.randint(500, 4000, n_samples),
        'GarageCars': np.random.randint(0, 4, n_samples),
        'FullBath': np.random.randint(1, 4, n_samples),
        'YearBuilt': np.random.randint(1950, 2023, n_samples),
        'Neighborhood': np.random.choice(['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel'], n_samples),
        'SalePrice': np.random.randint(100000, 500000, n_samples)
    })

# Filter specific features
X = df[selected_features]
y = df[target]

# Preprocessing for numerical data
numerical_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_features = ['Neighborhood']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf.fit(X_train, y_train)
print("Model trained.")

# Save the model
output_path = 'house_price_model.pkl'
joblib.dump(clf, output_path)
print(f"Model saved to {output_path}")
