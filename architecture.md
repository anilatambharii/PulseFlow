# Enterprise MLOps Pipeline Overview

![Enterprise MLOps Architecture](mlops_architecture.png)

## Pipeline Summary

| Step | Component | Description | Tools |
|------|------------|-------------|--------|
| 1 | Data Loader | Reads and preprocesses raw data | pandas, pyarrow |
| 2 | Model Training | Trains and serializes model | scikit-learn |
| 3 | Evaluation | Computes metrics and saves JSON | numpy, sklearn.metrics |
| 4 | Tracking | Registers experiments | MLflow |
| 5 | Deployment | Serves predictions | FastAPI, Uvicorn |
| 6 | CI/CD | Tests and validates build | pytest |


# MLOps pipeline Flow — from data ingestion to CI/CD testing and deployment

| Step | Component / Folder | File(s) | Description / Function | Tools & Frameworks | Execution Sequence in |
|------|--------------------|---------|------------------------------------------------------------------------------------------------------|---------------------------|-------------------------------------------|
| 1 | data/ | sample.csv, processed.parquet | Raw and processed datasets used for training and evaluation. Data ingestion scripts convert CSV → Parquet format using PyArrow. | Pandas, PyArrow | Load data → preprocess for model |
| 2 | data_ingestion/ | data_loader.py, data_preprocessor.py | Reads raw data, handles missing values, encodes features, and splits data into train-test sets. | pandas, NumPy, sklearn.preprocessing | Step 1: Clean and split dataset |
| 3 | models/ | train_model.py, saved_model.pkl, metrics.json | Trains regression/classification model, saves trained model artifact and performance metrics for traceability. | scikit-learn, joblib, json | Step 2: Train and store model |
| 4 | evaluation/ | evaluate_model.py | Loads saved model and test dataset (processed.parquet), computes MSE, MAE, RMSE, R². Writes results to metrics.json. | pandas, numpy, sklearn.metrics | Step 3: Evaluate trained model |
| 5 | mlflow/ | mlflow_setup.py, MLflow tracking directory | Sets up and connects MLflow tracking server, logs metrics, artifacts, and model versions. | MLflow (2.15.0+) | Step 4: Run mlflow ui & python mlflow_setup.py |
| 6 | app/ | main.py, routes/predict.py | Defines REST API (FastAPI) endpoints (/predict, /predict/batch) for real-time and batch predictions. Responds with model results + version info. | FastAPI, Uvicorn | Step 5: Start API → Test predictions via curl |
| 7 | ci_cd/ | tests/test_data_pipeline.py, tests/test_api.py | Automated unit and integration test suite verifying data pipeline, model training, and endpoint responses. | pytest | Step 6: Run tests — pytest ci_cd/tests/ -v |
| 8 | requirements.txt | Dependencies file | Version-pinned dependencies ensuring reproducible builds and compatibility (e.g., pandas 2.2.2, scikit-learn 1.5.1). | pip | Auto-installed in .venv |
| 9 | .venv/ | Virtual environment scripts | Local isolated environment housing all dependencies for model training and inference. | Python 3.11 | Activated before all scripts |
| 10 | Dockerfile (if included) | Environment specification | Containerizes project for production deployment ensuring identical runtime. | Docker, Linux Alpine | Optional prod step before CI/CD |
| 11 | .github/workflows/ or ci_cd/pipeline.yml | CI/CD pipeline configuration | Automates testing, packaging, and pushing of model updates or API builds to GitHub Actions. | GitHub Actions, Pytest, MLflow | Step 7: Automated GitOps or CI/CD build trigger |
| 12 | Root Directory | README.md | High-level project documentation explaining architecture, commands, and workflow setup. | — | Reference for setup and usage |
