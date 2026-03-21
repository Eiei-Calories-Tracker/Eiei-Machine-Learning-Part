from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
import os
import mlflow
from mlflow.tracking import MlflowClient
from src.main_train import run_training_task
from src.main_eval import run_eval_task
import requests

default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'start_date': datetime(2026, 3, 14),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'initial_train_dag',
    default_args=default_args,
    description='Initial training DAG for GoogLeNet Thai Food Classification',
    schedule_interval=None,
    catchup=False,
)

def register_and_promote_model_func(**context):
    client = MlflowClient()
    model_name = "googlenet-thai-food"
    ti = context['ti']
    
    # Get the run_id from the training task using XCom
    latest_run_id = ti.xcom_pull(task_ids='train_v1', key='return_value')
    if not latest_run_id:
        raise Exception("Run ID not found in XCom from 'train_v1'.")
        
    model_uri = f"runs:/{latest_run_id}/model"
    
    # Check if model artifact exists in MLflow, if not, try to log it from local fallback
    local_fallback = "/opt/airflow/src/best_model_local.pth"
    if os.path.exists(local_fallback):
        print(f"Checking for artifacts in run {latest_run_id}...")
        artifacts = client.list_artifacts(latest_run_id)
        if not any(a.path == "model" for a in artifacts):
            print("Model artifact missing in MLflow. Attempting manual upload from local fallback...")
            import torch
            from src.model import create_model
            # Reconstruct model and load state_dict
            checkpoint = torch.load(local_fallback)
            model = create_model(num_classes=checkpoint['num_classes'])
            model.load_state_dict(checkpoint['model_state_dict'])
            with mlflow.start_run(run_id=latest_run_id):
                mlflow.pytorch.log_model(model, "model")
            print("Manual upload successful.")

    # Register model
    result = mlflow.register_model(model_uri, model_name)
    version = result.version
    
    # Promote to Production
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"Model version {version} promoted to Production.")

train_task = PythonOperator(
    task_id='train_v1',
    python_callable=run_training_task,
    op_kwargs={
        'data_dir': '/opt/airflow/data/v1',
        'epochs': 1,
        'batch_size': 32,
        'experiment_name': 'ThaiFood_Initial',
        'run_name': 'initial_v1_run'
    },
    dag=dag,
)

eval_task = PythonOperator(
    task_id='evaluate_v1',
    python_callable=run_eval_task,
    op_kwargs={
        'data_dir': '/opt/airflow/data/v1',
        'model_uri': 'runs:/{{ ti.xcom_pull(task_ids="train_v1", key="return_value") }}/model', 
        'experiment_name': 'ThaiFood_Initial'
    },
    dag=dag,
)

register_promote_task = PythonOperator(
    task_id='register_and_promote',
    python_callable=register_and_promote_model_func,
    dag=dag,
)

# FastAPI update task (just a placeholder since FastAPI container loads from Production stage on startup/restart)
# We could trigger a restart of the fastapi container here if needed via docker socket, 
# but for "Test Flow" we assume it polls or is restarted manually.
# In a real setup, we'd use a signal or an API call to tell FastAPI to reload.

def reload_fastapi_func():
    try:
        response = requests.post("http://fastapi:7860/reload")
        response.raise_for_status()
        print("FastAPI model reload triggered successfully.")
    except Exception as e:
        print(f"Failed to trigger FastAPI reload: {e}")

reload_task = PythonOperator(
    task_id='reload_fastapi',
    python_callable=reload_fastapi_func,
    dag=dag,
)

train_task >> eval_task >> register_promote_task >> reload_task
