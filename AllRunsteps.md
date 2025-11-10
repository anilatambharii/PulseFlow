
# PulseFlow Platform - Comprehensive End-to-End Run Steps

This document combines platform setup, pipeline execution, model serving, orchestration, and end-to-end commands for a complete open source MLOps demo.

---

## Repository Setup
```bash
git clone https://github.com/anilatambharii/PulseFlow.git
cd PulseFlow
```

## Python Environment & Dependencies
```bash
python -m venv .venv
# On Unix/Mac:
source .venv/bin/activate
# On Windows:
.venv\Scripts\Activate
pip install -r requirements.txt
```

---

## ETL & Modeling Pipeline
| Step | Script / Command | Location / Purpose | Description of What It Does |
|------|------------------|--------------------|-----------------------------|
| 1 | `python data_ingestion/data_loader.py` | Reads raw data (`data/sample.csv`) | Loads data, cleans, encodes, splits to train/test, outputs `data/processed.parquet`. |
| 2 | `python models/train_model.py` | Model training | Reads processed data, trains ML (e.g., regression), saves to `models/saved_model.pkl`. |
| 3 | `python evaluation/evaluate_model.py` | Model evaluation | Loads model & test data, outputs metrics (`models/metrics.json`). |

---

## MLflow Tracking & Versioning
| Step | Command | Purpose | Description |
|------|---------|---------|-------------|
| 4 | `mlflow ui &` | Tracking server | Launches MLflow web UI at [http://127.0.0.1:5000](http://127.0.0.1:5000) for model/metric tracking. |
| 5 | `python mlflow/mlflow_setup.py` | MLflow pipeline | Connects models, metrics, parameters to MLflow with run versioning. |

---

## FastAPI Real-Time & Batch Prediction
| Step | Command | Purpose | Description |
|------|---------|---------|-------------|
| 6 | `uvicorn app.main:app --reload --port 8000` | Deployment | Spins up REST API with `/predict` and `/predict/batch` endpoints. |
| 7 | `curl -X POST "http://localhost:8000/predict" ...` | Single prediction | Test API endpoint with input payload. |
| 8 | `curl -X POST "http://localhost:8000/predict/batch" ...` | Batch prediction | Test batch API endpoint with multi-row input. |

---

## Airflow Orchestration & CI/CD Testing
| Step | Command | Purpose | Description |
|------|---------|---------|-------------|
| 9 | `airflow db init`<br>`airflow webserver -p 8080`<br>`airflow scheduler` | Airflow DAGs | Starts orchestration UI at [http://localhost:8080](http://localhost:8080). |
| 10 | `pytest ci_cd/tests/ -v` | CI/CD | Runs automated tests for pipeline and API integrity. |

---

## Docker Compose (All services together)
| Step | Command | Purpose | Description |
|------|---------|---------|-------------|
| 11 | `docker-compose up --build` | Platform | Runs API, MLflow, Airflow, dependencies in containers for full-stack demo. |
