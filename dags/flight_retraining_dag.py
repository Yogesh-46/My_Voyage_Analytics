from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# ==========================================
# 🎛️ AIRFLOW DAG CONFIGURATION
# ==========================================
default_args = {
    'owner': 'Yogesh',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1), # Synchronized to keep production logs clean
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the master automated schedule loop
dag = DAG(
    'voyage_flight_model_retraining',
    default_args=default_args,
    description='Automated pipeline to preprocess new travel logs and retrain the XGBoost engine.',
    schedule_interval='@weekly', # Automatically runs once a week at midnight
    catchup=False,
)

# ==========================================
# 🏃‍♂️ DEFINE THE AUTOMATED PIPELINE TASKS
# ==========================================

# Task 1: Ingest and preprocess any fresh flight transaction data
preprocess_task = BashOperator(
    task_id='run_data_preprocessing',
    bash_command='python /app/src/flight_prediction/preprocessing.py',
    dag=dag,
)

# Task 2: Retrain the XGBoost model and log results to the MLflow registry
retrain_task = BashOperator(
    task_id='run_model_retraining',
    bash_command='python /app/src/flight_prediction/train.py',
    dag=dag,
)

# ==========================================
# ⛓️ SET TASK EXECUTION ORDER (PIPELINE DEPENDENCY)
# ==========================================
# Data must be completely preprocessed before the model can begin training
preprocess_task >> retrain_task