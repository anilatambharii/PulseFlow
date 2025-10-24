from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess


def run_etl():
    subprocess.run(['python', 'etl/data_ingestion.py'], check=True)
    subprocess.run(['python', 'etl/data_preprocessing.py'], check=True)


def run_training():
    subprocess.run(['python', 'training/train_model.py'], check=True)


def run_deployment():
    subprocess.run(['uvicorn', 'deployment.app.main:app', '--host', '0.0.0.0', '--port', '8000'], check=True)


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

dag = DAG(
    'mlops_pipeline_dag',
    default_args=default_args,
    description='End-to-end Enterprise MLOps workflow',
    schedule_interval='@daily',
    start_date=datetime(2025, 1, 1),
    catchup=False,
)

etl_task = PythonOperator(
    task_id='run_etl',
    python_callable=run_etl,
    dag=dag,
)

train_task = PythonOperator(
    task_id='run_training',
    python_callable=run_training,
    dag=dag,
)

deploy_task = PythonOperator(
    task_id='run_deployment',
    python_callable=run_deployment,
    dag=dag,
)

etl_task >> train_task >> deploy_task
