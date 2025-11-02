"""
Data preparation and model training script for Iris flower classification
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import json

# Load the Iris dataset
print("Loading Iris dataset...")
iris = load_iris()
X = iris.data
y = iris.target

# Create a DataFrame for better data exploration
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
df['species'] = df['target'].map({0: iris.target_names[0], 
                                   1: iris.target_names[1], 
                                   2: iris.target_names[2]})

print("\n=== Dataset Information ===")
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nDataset statistics:\n{df.describe()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nClass distribution:\n{df['species'].value_counts()}")

# Check for missing values (there are none in Iris dataset)
if df.isnull().sum().sum() > 0:
    print("\nHandling missing values...")
    df = df.fillna(df.mean())
else:
    print("\nNo missing values found!")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n=== Model Comparison ===")

# Define models to compare
models = {
    'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

results = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"\n--- {name} ---")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Cross-Validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=iris.target_names)}")
    
    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

# Find the best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
print(f"\n=== Best Model: {best_model_name} ===")
print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")

# Train the best model on full training data and save it
best_model = models[best_model_name]
best_model.fit(X_train_scaled, y_train)

# Save the model and scaler
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save feature names and target names
metadata = {
    'feature_names': iris.feature_names,
    'target_names': iris.target_names.tolist(),
    'best_model': best_model_name,
    'accuracy': results[best_model_name]['accuracy']
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\nModel, scaler, and metadata saved successfully!")
print(f"Files created: model.pkl, scaler.pkl, model_metadata.json")
