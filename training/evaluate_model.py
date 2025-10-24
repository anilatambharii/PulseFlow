import os
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

def evaluate_model(model_path: str, data_path: str, output_metrics_path: str = 'models/metrics.json'):
    """
    Evaluate a trained model and save metrics
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    print(f"Loading evaluation data from {data_path}...")
    df = pd.read_parquet(data_path)

    if 'target' not in df.columns:
        raise ValueError("Target column 'target' not found in dataset")

    X = df.drop('target', axis=1)
    y = df['target']

    print(f"Evaluating model on {len(X)} samples...")
    y_pred = model.predict(X)

    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    rmse = mse ** 0.5

    metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2_score': float(r2)
    }

    print("\nEvaluation Results:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2 Score: {r2:.4f}")

    # Save metrics to JSON
    os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
    with open(output_metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"\nMetrics saved to {output_metrics_path}")

    return metrics

if __name__ == "__main__":
    MODEL_PATH = 'models/saved_model.pkl'
    DATA_PATH = 'data/processed.parquet'
    METRICS_OUTPUT_PATH = 'models/metrics.json'

    evaluate_model(MODEL_PATH, DATA_PATH, METRICS_OUTPUT_PATH)
