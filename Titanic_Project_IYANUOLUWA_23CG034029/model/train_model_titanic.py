import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# 1. Load the dataset
data_path = 'titanic.csv'

# Selected features (5 selected from recommended list)
# Recommended: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
selected_features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
target = 'Survived'

if os.path.exists(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
else:
    print("titanic.csv not found. Generating synthetic dataset for demonstration...")
    # Generate synthetic data mimicking Titanic dataset
    np.random.seed(42)
    n_samples = 891
    df = pd.DataFrame({
        'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
        'Sex': np.random.choice(['male', 'female'], n_samples),
        'Age': np.random.normal(30, 14, n_samples).astype(int),
        'Fare': np.random.exponential(32, n_samples),
        'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.7, 0.2, 0.1]),
        'Survived': np.random.randint(0, 2, n_samples)
    })
    # Ensure Age is usually positive (some noise might make it negative)
    df['Age'] = df['Age'].apply(lambda x: max(1, x))

print("Data Preview:")
print(df[selected_features].head())

# 2. Data Preprocessing setup
X = df[selected_features]
y = df[target]

# Preprocessing for numerical data
numerical_features = ['Age', 'Fare']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_features = ['Pclass', 'Sex', 'Embarked']
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

# 3. Implement Logistic Regression
model = LogisticRegression(random_state=42, max_iter=1000)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the model
print("Training model...")
clf.fit(X_train, y_train)

# 5. Evaluate the model
print("Evaluating model...")
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 6. Save the trained model
model_filename = 'titanic_survival_model.pkl'
joblib.dump(clf, model_filename)
print(f"Model saved to {model_filename}")

# 7. Reload demonstration
print("Reloading model to verify...")
loaded_model = joblib.load(model_filename)
sample_pred = loaded_model.predict(X_test.iloc[[0]])
print(f"Prediction for first test sample: {'Survived' if sample_pred[0] == 1 else 'Did Not Survive'}")
