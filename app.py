# app.py — PulseFlow MLOps Demo Space
import gradio as gr
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Simulate the PulseFlow pipeline stages
def run_etl(uploaded_file):
    if uploaded_file is None:
        df = pd.DataFrame({
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "feature_3": np.random.rand(100) * 100,
            "target":    np.random.randint(0, 2, 100)
        })
        source = "generated sample data (100 rows)"
    else:
        df = pd.read_csv(uploaded_file.name)
        source = f"uploaded file ({len(df)} rows)"

    nulls_before = df.isnull().sum().sum()
    df = df.dropna()
    nulls_after  = df.isnull().sum().sum()

    report = {
        "source":        source,
        "rows_loaded":   len(df),
        "columns":       list(df.columns),
        "nulls_removed": int(nulls_before - nulls_after),
        "dtypes":        {c: str(t) for c, t in df.dtypes.items()},
        "timestamp":     datetime.utcnow().isoformat() + "Z"
    }
    return df.head(10), json.dumps(report, indent=2)

def run_training(n_estimators, max_depth, test_size):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score

    np.random.seed(42)
    X = np.random.randn(500, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "experiment":    "enterprise_mlops_training",
        "model":         "RandomForestClassifier",
        "n_estimators":  int(n_estimators),
        "max_depth":     int(max_depth),
        "test_size_pct": test_size,
        "accuracy":      round(accuracy_score(y_test, preds), 4),
        "f1_score":      round(f1_score(y_test, preds), 4),
        "train_samples": len(X_train),
        "test_samples":  len(X_test),
        "status":        "completed",
        "mlflow_uri":    "See GitHub repo to connect your MLflow instance",
        "timestamp":     datetime.utcnow().isoformat() + "Z"
    }
    return json.dumps(metrics, indent=2)

def run_inference(f1, f2, f3, f4, f5):
    features   = [f1, f2, f3, f4, f5]
    score      = 1 / (1 + np.exp(-sum(features[:2])))
    prediction = int(score > 0.5)
    result = {
        "endpoint":    "/predict",
        "input":       {"features": features},
        "prediction":  prediction,
        "confidence":  round(float(score if prediction == 1 else 1 - score), 4),
        "model":       "RandomForestClassifier v0.1.0",
        "latency_ms":  round(np.random.uniform(2, 8), 2),
        "status":      "200 OK",
        "timestamp":   datetime.utcnow().isoformat() + "Z"
    }
    return json.dumps(result, indent=2)

# UI
with gr.Blocks(title="PulseFlow MLOps", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# PulseFlow MLOps Pipeline
**Production-grade open source MLOps** — ETL → Training → FastAPI Inference

[![PyPI](https://img.shields.io/pypi/v/pulseflow-mlops)](https://pypi.org/project/pulseflow-mlops/)
[![GitHub](https://img.shields.io/badge/GitHub-PulseFlow-black)](https://github.com/anilatambharii/PulseFlow)
[![AmbhariiLabs](https://img.shields.io/badge/Org-AmbhariiLabs-blue)](https://huggingface.co/AmbhariiLabs)

Built by [Anil Prasad](https://www.linkedin.com/in/anilsprasad) — Head of Engineering & Product, Duke Energy | Founder, Ambharii Labs
    """)

    with gr.Tabs():
        with gr.Tab("Stage 1 — ETL"):
            gr.Markdown("Upload a CSV or use generated sample data to simulate the ingestion and preprocessing pipeline.")
            with gr.Row():
                file_input = gr.File(label="Upload CSV (optional)", file_types=[".csv"])
                etl_btn    = gr.Button("Run ETL Pipeline", variant="primary")
            data_preview  = gr.Dataframe(label="Processed Data Preview (first 10 rows)")
            etl_report    = gr.Code(label="ETL Report (JSON)", language="json")
            etl_btn.click(run_etl, inputs=[file_input], outputs=[data_preview, etl_report])

        with gr.Tab("Stage 2 — Training"):
            gr.Markdown("Configure hyperparameters and run the training pipeline. Metrics mirror what MLflow captures in production.")
            with gr.Row():
                n_est      = gr.Slider(10, 200, value=100, step=10, label="n_estimators")
                max_d      = gr.Slider(2,   20, value=5,   step=1,  label="max_depth")
                test_s     = gr.Slider(10,  40, value=20,  step=5,  label="Test size %")
            train_btn      = gr.Button("Run Training", variant="primary")
            train_output   = gr.Code(label="MLflow Experiment Results (JSON)", language="json")
            train_btn.click(run_training, inputs=[n_est, max_d, test_s], outputs=[train_output])

        with gr.Tab("Stage 3 — Inference"):
            gr.Markdown("Simulate the FastAPI `/predict` endpoint. In production this runs via `uvicorn deployment.app.main:app`.")
            with gr.Row():
                f1 = gr.Number(value=0.5,  label="Feature 1")
                f2 = gr.Number(value=-0.3, label="Feature 2")
                f3 = gr.Number(value=1.2,  label="Feature 3")
                f4 = gr.Number(value=0.0,  label="Feature 4")
                f5 = gr.Number(value=0.8,  label="Feature 5")
            infer_btn    = gr.Button("Run Inference", variant="primary")
            infer_output = gr.Code(label="API Response (JSON)", language="json")
            infer_btn.click(run_inference, inputs=[f1, f2, f3, f4, f5], outputs=[infer_output])

    gr.Markdown("""
---
**Install locally:** `pip install pulseflow-mlops` | 
**Full stack:** `docker-compose up --build` | 
**Docs:** [GitHub README](https://github.com/anilatambharii/PulseFlow)
    """)

demo.launch()