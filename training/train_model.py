import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# MLflow tracking configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("enterprise_mlops_training")

def train_model(data_path: str, model_output_path: str):
    """
    Train a Random Forest model with MLflow tracking
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found at {data_path}")

    print(f"Loading processed data from {data_path}...")
    df = pd.read_parquet(data_path)

    # Separate features and target
    if 'target' not in df.columns:
        raise ValueError("Target column 'target' not found in dataset")

    X = df.drop('target', axis=1)
    y = df['target']

    print(f"Dataset shape: {X.shape}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Start MLflow run
    with mlflow.start_run(run_name="random_forest_training"):
        # Model parameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'random_state': 42
        }

        print("Training Random Forest model...")
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Model Performance:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R2 Score: {r2:.4f}")

        # Log parameters and metrics to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

        # Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="enterprise_mlops_rf_model"
        )

        # Save model locally
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump(model, model_output_path)
        print(f"Model saved to {model_output_path}")

        return model, mse, mae, r2

if __name__ == "__main__":
    DATA_PATH = 'data/processed.parquet'
    MODEL_OUTPUT_PATH = 'models/saved_model.pkl'

    train_model(DATA_PATH, MODEL_OUTPUT_PATH)
