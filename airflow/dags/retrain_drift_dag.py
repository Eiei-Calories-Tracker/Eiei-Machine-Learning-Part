from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import os
import subprocess
import time
import mlflow
from mlflow.tracking import MlflowClient
import docker
from src.data_utils import prepare_new_version_from_latest_with_reservoir, get_latest_version
from src.main_train import run_training_task
from src.main_eval import evaluate_model_uri
from src.drift_check_main import run_drift_check_task
from src.mlflow_metadata import CANONICAL_EXPERIMENT_NAME, build_model_version_description, init_mlflow
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
    render_template_as_native_obj=True,
    params={
        "batch_size": 16,
    },
)

def _resolve_latest_test_data_dir(base_data_dir="/opt/airflow/data"):
    latest_version = get_latest_version(base_data_dir)
    if latest_version:
        return f"{base_data_dir}/{latest_version}"
    return f"{base_data_dir}/v1"


def check_drift_branch_func(**context):
    ti = context['ti']
    is_drift = ti.xcom_pull(task_ids='check_drift', key='return_value')
    if bool(is_drift):
        return "prepare_new_data"
    return "no_drift_detected"

def prepare_data_func(**context):
    base_dir = "/opt/airflow/data"
    init_mlflow("http://mlflow:5000")

    latest_v = get_latest_version(base_dir)
    v_num = int(latest_v[1:]) if latest_v else 0
    new_v = f"v{v_num + 1}"

    prepare_summary = prepare_new_version_from_latest_with_reservoir(
        base_data_dir=base_dir,
        target_version=new_v,
        new_data_dir="/opt/airflow/mockData",
        sample_ratio=0.7,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42,
    )

    # subprocess.run(["dvc", "add", f"data/{new_v}"], cwd="/opt/airflow", check=True)
    # subprocess.run(["dvc", "push", "-r", "s3remote"], cwd="/opt/airflow", check=True)

    context['ti'].xcom_push(key='new_version', value=new_v)
    context['ti'].xcom_push(key='prepare_summary', value=prepare_summary)
    return prepare_summary['new_data_dir']


def train_new_version_func(**context):
    ti = context['ti']
    new_version = ti.xcom_pull(task_ids='prepare_new_data', key='new_version')
    if not new_version:
        raise ValueError("Missing new_version from prepare_new_data")

    run_id = run_training_task(
        data_dir=f"/opt/airflow/data/{new_version}",
        epochs=1,
        experiment_name=CANONICAL_EXPERIMENT_NAME,
        run_name=f'retrain_{new_version}',
        base_model_uri='models:/googlenet-thai-food/Production',
        tracking_uri='http://mlflow:5000',
        batch_size=context['params']['batch_size'],
        phase='retrain',
        trigger_source='airflow',
        dag_id=context['dag'].dag_id,
        task_id=context['task'].task_id,
        airflow_run_id=context['dag_run'].run_id if context.get('dag_run') else None,
        drift_triggered=True,
        base_model='models:/googlenet-thai-food/Production',
        data_version=new_version,
        run_description=f"retrain with drift trigger for {new_version}",
    )
    return run_id


def evaluate_candidate_func(**context):
    ti = context['ti']
    init_mlflow("http://mlflow:5000")

    run_id = ti.xcom_pull(task_ids='fine_tune_new_version', key='return_value')
    if not run_id:
        raise ValueError("Missing run_id from fine_tune_new_version")

    data_dir = _resolve_latest_test_data_dir()
    latest_version = get_latest_version("/opt/airflow/data")
    candidate_uri = f"runs:/{run_id}/model"
    mlflow.set_experiment(CANONICAL_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"eval_candidate_{latest_version or 'unknown'}"):
        mlflow.set_tags(
            {
                "trigger_source": "airflow",
                "dag_id": context['dag'].dag_id,
                "task_id": context['task'].task_id,
                "airflow_run_id": context['dag_run'].run_id if context.get('dag_run') else "unknown",
                "phase": "evaluation",
                "data_version": latest_version or "unknown",
                "drift_triggered": "true",
                "base_model": "models:/googlenet-thai-food/Production",
                "eval_purpose": "candidate_selection",
                "eval_split": "test",
                "parent_training_run": run_id,
            }
        )
        mlflow.set_tag("mlflow.note.content", "Evaluate drift-retrained candidate model for promotion")
        _, candidate_acc = evaluate_model_uri(data_dir=data_dir, model_uri=candidate_uri, split='test')
        mlflow.log_metrics({"test_acc": float(candidate_acc)})
    ti.xcom_push(key='candidate_acc', value=float(candidate_acc))
    return float(candidate_acc)

def compare_and_promote_func(**context):
    ti = context['ti']
    init_mlflow("http://mlflow:5000")

    new_acc = float(ti.xcom_pull(task_ids='evaluate_new_version', key='candidate_acc'))
    
    client = MlflowClient()
    model_name = "googlenet-thai-food"
    data_dir = _resolve_latest_test_data_dir()

    prev_acc = None
    should_promote = True
    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if prod_versions:
            _, prev_acc = evaluate_model_uri(data_dir=data_dir, model_uri=f"models:/{model_name}/Production", split='test')
            should_promote = new_acc >= prev_acc
    except Exception as error:
        print(f"No active Production baseline for comparison: {error}")

    if not should_promote:
        print(f"New model not promoted: candidate_acc={new_acc:.4f} < production_acc={prev_acc:.4f}")
        ti.xcom_push(key='promoted', value=False)
        return False

    latest_run_id = ti.xcom_pull(task_ids="fine_tune_new_version", key="return_value")
    result = mlflow.register_model(f"runs:/{latest_run_id}/model", model_name)
    description = build_model_version_description(
        {
            "phase": "retrain",
            "source_run_id": latest_run_id,
            "data_version": get_latest_version("/opt/airflow/data"),
            "trigger_source": "airflow",
            "dag_id": context['dag'].dag_id,
            "task_id": context['task'].task_id,
            "airflow_run_id": context['dag_run'].run_id if context.get('dag_run') else None,
            "drift_triggered": True,
            "base_model": "models:/googlenet-thai-food/Production",
            "candidate_acc": new_acc,
            "production_acc": prev_acc,
            "note": "Retrain DAG promotion decision after drift branch",
        }
    )
    client.update_model_version(name=model_name, version=result.version, description=description)
    client.transition_model_version_stage(model_name, result.version, "Production", archive_existing_versions=True)
    print(f"New model version {result.version} promoted to Production")
    ti.xcom_push(key='promoted', value=True)
    return True


def restart_fastapi_container_func(**context):
    ti = context['ti']
    promoted = ti.xcom_pull(task_ids='compare_and_promote', key='promoted')
    if not promoted:
        print("No promotion occurred. Skipping FastAPI restart.")
        return "skipped"

    docker_client = docker.DockerClient(base_url='unix://var/run/docker.sock')
    container = docker_client.containers.get("fastapi")
    container.restart(timeout=20)
    print("FastAPI container restarted.")

    health_url = "http://fastapi:7860/health"
    last_error = None
    for _ in range(60):
        try:
            response = requests.get(health_url, timeout=3)
            if response.status_code == 200:
                print("FastAPI health check passed.")
                return "restarted"
        except Exception as error:
            last_error = error
        time.sleep(2)

    raise RuntimeError(f"FastAPI did not become healthy after restart. Last error: {last_error}")

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

no_drift_task = EmptyOperator(
    task_id='no_drift_detected',
    dag=dag,
)

prepare_data_task = PythonOperator(
    task_id='prepare_new_data',
    python_callable=prepare_data_func,
    dag=dag,
)

train_task = PythonOperator(
    task_id='fine_tune_new_version',
    python_callable=train_new_version_func,
    dag=dag,
)

eval_task = PythonOperator(
    task_id='evaluate_new_version',
    python_callable=evaluate_candidate_func,
    dag=dag,
)

compare_promote_task = PythonOperator(
    task_id='compare_and_promote',
    python_callable=compare_and_promote_func,
    dag=dag,
)

reload_task = PythonOperator(
    task_id='restart_fastapi',
    python_callable=restart_fastapi_container_func,
    dag=dag,
)

drift_check_task >> branch_task >> [no_drift_task, prepare_data_task]
prepare_data_task >> train_task >> eval_task >> compare_promote_task >> reload_task
