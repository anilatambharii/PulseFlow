# End-to-End Execution Steps

| Step | Script / Command | Location / Purpose | Description of What It Does |
|------|------------------|--------------------|-----------------------------|
| 1 | `python data_ingestion/data_loader.py` | Reads raw data (`data/sample.csv`) | Loads data, performs cleaning, encoding, and splits into train/test datasets. Outputs `data/processed.parquet`. |
| 2 | `python models/train_model.py` | Trains model and saves as artifact | Reads processed data, trains ML model (e.g., linear regression), and stores `models/saved_model.pkl`. |
| 3 | `python evaluation/evaluate_model.py` | Evaluates trained model | Loads model + test data, calculates metrics (MSE, MAE, RMSE, R²), and saves to `models/metrics.json`. |
| 4 | `mlflow ui &` | From main project directory | Launches MLflow tracking server via local web app at [http://127.0.0.1:5000](http://127.0.0.1:5000) for metrics and artifact tracking. *(Keep this terminal open.)* |
| 5 | `python mlflow/mlflow_setup.py` | Logs results to MLflow | Connects active model, metrics, and params to MLflow tracking server, versioning each run. |
| 6 | `uvicorn app.main:app --reload --port 8000` | Deploys FastAPI service | Spins up REST API with endpoints `/predict` and `/predict/batch`. This serves the model for real-time predictions. |
| 7 | `curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{\"features\": {\"feature1\": 50.0, \"feature2\": 75.0}}"` | FastAPI test | Sends real input payload to `/predict` endpoint to return a single prediction and the model version ID. |
| 8 | `curl -X POST "http://localhost:8000/predict/batch" -H "Content-Type: application/json" -d "{\"data\": [{\"feature1\": 50.0, \"feature2\": 75.0}, {\"feature1\": 30.0, \"feature2\": 45.0}]}"` | FastAPI test | Sends input payload as a batch to `/predict/batch` endpoint to return multiple predictions and the model version ID. |
| 9 | `pytest ci_cd/tests/ -v` | CI/CD testing | Runs end-to-end automated tests covering pipeline integrity, data validation, and API predictions. |