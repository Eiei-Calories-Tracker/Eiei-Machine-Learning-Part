from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
import os
import mlflow
from mlflow.tracking import MlflowClient
from src.data_utils import prepare_new_version_data, get_latest_version
from src.main_train import run_training_task
from src.main_eval import run_eval_task
from src.drift_check_main import run_drift_check_task
import requests

default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'start_date': datetime(2026, 3, 14),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'retrain_drift_dag',
    default_args=default_args,
    description='Retraining DAG triggered by data drift',
    schedule_interval='@weekly',
    catchup=False,
)

def check_drift_branch_func():
    if os.path.exists("drift_result.txt"):
        with open("drift_result.txt", "r") as f:
            result = f.read().strip()
        if result == "drift":
            return "prepare_new_data"
    return "no_drift_detected"

def prepare_data_func(**context):
    base_dir = "/opt/airflow/data"
    mock_dir = "/opt/airflow/mockData"
    
    latest_v = get_latest_version(base_dir)
    v_num = int(latest_v[1:]) if latest_v else 0
    new_v = f"v{v_num + 1}"
    
    new_path = prepare_new_version_data(mock_dir, base_dir, new_v)
    context['ti'].xcom_push(key='new_version', value=new_v)
    return new_path

def compare_and_promote_func(**context):
    ti = context['ti']
    new_acc = float(open("eval_acc.txt").read().strip())
    
    client = MlflowClient()
    model_name = "googlenet-thai-food"
    
    # Get current Production accuracy if possible
    try:
        prod_version = client.get_latest_versions(model_name, stages=["Production"])[0]
        # In a real setup, we'd store accuracy in model tags/metadata
        # For this test, we assume if we reached this task, the new one might be better or we just promote it.
        # Let's just compare against a fixed threshold or the previous run's metric
        prev_run_id = prod_version.run_id
        prev_acc = client.get_run(prev_run_id).data.metrics.get("val_acc", 0)
    except:
        prev_acc = 0
        
    if new_acc >= prev_acc:
        latest_run_id = ti.xcom_pull(task_ids="fine_tune_new_version", key="return_value")
        
        result = mlflow.register_model(f"runs:/{latest_run_id}/model", model_name)
        client.transition_model_version_stage(model_name, result.version, "Production", archive_existing_versions=True)
        print(f"New model version {result.version} promoted to Production (Acc: {new_acc} >= {prev_acc})")
    else:
        print(f"New model (Acc: {new_acc}) not better than Production (Acc: {prev_acc}).")

drift_check_task = PythonOperator(
    task_id='check_drift',
    python_callable=run_drift_check_task,
    op_kwargs={
        'base_data_dir': '/opt/airflow/data',
        'mock_data_dir': '/opt/airflow/mockData'
    },
    dag=dag,
)

branch_task = BranchPythonOperator(
    task_id='drift_branch',
    python_callable=check_drift_branch_func,
    dag=dag,
)

no_drift_task = PythonOperator(
    task_id='no_drift_detected',
    python_callable=lambda: print("No drift detected, stopping pipeline."),
    dag=dag,
)

prepare_data_task = PythonOperator(
    task_id='prepare_new_data',
    python_callable=prepare_data_func,
    dag=dag,
)

train_task = PythonOperator(
    task_id='fine_tune_new_version',
    python_callable=run_training_task,
    op_kwargs={
        'data_dir': '/opt/airflow/data/{{ ti.xcom_pull(task_ids="prepare_new_data", key="new_version") }}',
        'epochs': 1,
        'experiment_name': 'ThaiFood_Initial',
        'run_name': 'retrain_{{ ti.xcom_pull(task_ids="prepare_new_data", key="new_version") }}',
        'base_model_uri': 'models:/googlenet-thai-food/Production'
    },
    dag=dag,
)

eval_task = PythonOperator(
    task_id='evaluate_new_version',
    python_callable=run_eval_task,
    op_kwargs={
        'data_dir': '/opt/airflow/data/test',
        'model_uri': 'runs:/{{ ti.xcom_pull(task_ids="fine_tune_new_version", key="return_value") }}/model',
        'experiment_name': 'ThaiFood_Initial'
    },
    dag=dag,
)

compare_promote_task = PythonOperator(
    task_id='compare_and_promote',
    python_callable=compare_and_promote_func,
    dag=dag,
)

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

drift_check_task >> branch_task >> [no_drift_task, prepare_data_task]
prepare_data_task >> train_task >> eval_task >> compare_promote_task >> reload_task
