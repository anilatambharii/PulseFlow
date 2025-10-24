import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

def generate_sample_data():
    """Generate synthetic sample data"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'feature1': np.random.randn(n_samples) * 10 + 50,
        'feature2': np.random.randn(n_samples) * 15 + 75,
        'target': None
    }
    
    # Create target as a function of features with some noise
    data['target'] = (
        data['feature1'] * 2.5 + 
        data['feature2'] * 1.8 + 
        np.random.randn(n_samples) * 5
    )
    
    return pd.DataFrame(data)

def train_and_save_model(output_path='models/saved_model.pkl'):
    """Train a simple model and save it"""
    
    print("Generating sample dataset...")
    df = generate_sample_data()
    
    X = df[['feature1', 'feature2']]
    y = df['target']
    
    print(f"Dataset shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Train R² Score: {train_score:.4f}")
    print(f"Test R² Score: {test_score:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    
    print(f"\nModel saved to: {output_path}")
    print(f"Model file size: {os.path.getsize(output_path) / 1024:.2f} KB")
    
    return model

if __name__ == "__main__":
    train_and_save_model()
