# MLflow Directory

This directory contains MLflow configuration and setup scripts for experiment tracking and model registry.

## Quick Start

### 1. Start MLflow Server

mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

Or simply:

mlflow ui

### 2. Run Setup Script

python mlflow/mlflow_setup.py

This will:
- Verify MLflow server connection
- Create the experiment
- Initialize model registry
- Log a sample run

### 3. Access MLflow UI

Open your browser to: [**http://localhost:5000**](http://localhost:5000)

## Configuration

Edit `mlflow_config.env` to customize:

- Tracking URI
- Backend store location
- Artifact storage path
- Experiment name
- Model registry name

## Usage in Training Pipeline

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("enterprise_mlops_training")

with mlflow.start_run():
mlflow.log_param("n_estimators", 100)
mlflow.log_metric("mse", 0.05)
mlflow.sklearn.log_model(model, "model")

## Directory Structure

mlflow/
├── mlflow_setup.py # Setup and configuration script
├── mlflow_config.env # Environment variables
└── README.md # This file

## Troubleshooting

**Connection Error:**
Make sure MLflow server is running:
mlflow ui

**Port Already in Use:**
mlflow server --port 5001

**Database Issues:**
rm mlflow.db
python mlflow/mlflow_setup.py

How to Use
Step 1: Start MLflow Server
bash
# Simple method (uses default settings)
mlflow ui

# Or with custom configuration
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
Step 2: Run Setup
bash
python mlflow/mlflow_setup.py
Output:

text
Starting MLflow setup...

Step 1: Verifying MLflow connection...
✓ Successfully connected to MLflow at http://localhost:5000
  Found 1 experiment(s)

Step 2: Setting up experiment...
============================================================
MLflow Setup Configuration
============================================================
✓ Tracking URI set to: http://localhost:5000
✓ Experiment already exists: enterprise_mlops_training
  Experiment ID: 0
✓ Artifact location: ./mlruns
============================================================
MLflow setup completed successfully!
============================================================

Step 3: Creating model registry...
✓ Created new registered model: enterprise_mlops_rf_model

Step 4: Logging sample run...
✓ Sample run logged successfully
Step 3: Access MLflow UI
Open http://localhost:5000 in your browser to view:

All experiment runs

Model parameters and metrics

Registered models

Artifacts and visualizations
