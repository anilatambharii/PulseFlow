# Enterprise MLOps Pipeline

This repository provides a production-ready MLOps template for building and deploying machine learning pipelines in enterprise environments.

## Architecture Overview

Components included:
- ETL Pipeline: Data ingestion and preprocessing.
- Training Pipeline: Model training with MLflow tracking.
- Deployment Service: FastAPI microservice for real-time inference.
- Airflow Orchestration: Workflow automation for end-to-end pipelines.
- Dockerized Stack: Easily deployable with Docker Compose.

## Run Locally

### Prerequisites
- Python 3.10+
- Docker & Docker Compose

### 1. Install dependencies
python -m venv .venv
source .venv/bin/activate # or .venv\Scripts\activate on Windows
pip install -r requirements.txt


### 2. Run the pipeline manually

python etl/data_ingestion.py
python etl/data_preprocessing.py
python training/train_model.py
uvicorn deployment.app.main:app --reload


### 3. Start MLflow and Airflow (optional)

mlflow ui &
airflow db init && airflow webserver -p 8080 & airflow scheduler &


### 4. Run full stack with Docker
docker-compose up --build
