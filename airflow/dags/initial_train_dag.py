from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import time
import os
import requests
import mlflow
from mlflow.tracking import MlflowClient
import docker
from src.data_utils import get_latest_version, prepare_new_version_data
from src.main_train import run_training_task
from src.main_eval import evaluate_model_uri
from src.mlflow_metadata import CANONICAL_EXPERIMENT_NAME, build_model_version_description, init_mlflow

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
    render_template_as_native_obj=True,
    params={
        "batch_size": 16,
    },
)

def _resolve_latest_test_data_dir(base_data_dir="/opt/airflow/data"):
    latest_version = get_latest_version(base_data_dir)
    if latest_version:
        latest_test_dir = f"{base_data_dir}/{latest_version}"
        return latest_test_dir
    return f"{base_data_dir}/v1"


def preprocess_v1_func():
    target_dir = "/opt/airflow/data/v1"
    train_dir = f"{target_dir}/train"
    val_dir = f"{target_dir}/val"

    has_train = os.path.isdir(train_dir) and any(os.scandir(train_dir))
    has_val = os.path.isdir(val_dir) and any(os.scandir(val_dir))
    if has_train and has_val:
        print("data/v1 already prepared. Skipping preprocess.")
        return target_dir

    print("Preparing data/v1 from /opt/airflow/mockData ...")
    prepared_dir = prepare_new_version_data(
        mock_data_dir="/opt/airflow/mockData",
        base_data_dir="/opt/airflow/data",
        target_version="v1",
    )
    print(f"Prepared dataset at {prepared_dir}")
    return prepared_dir


def evaluate_candidate_func(**context):
    ti = context['ti']
    init_mlflow("http://mlflow:5000")

    run_id = ti.xcom_pull(task_ids='train_v1', key='return_value')
    if not run_id:
        raise ValueError("Missing run_id from train_v1")

    data_dir = _resolve_latest_test_data_dir()
    candidate_uri = f"runs:/{run_id}/model"
    mlflow.set_experiment(CANONICAL_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="eval_candidate_v1"):
        mlflow.set_tags(
            {
                "trigger_source": "airflow",
                "dag_id": context['dag'].dag_id,
                "task_id": context['task'].task_id,
                "airflow_run_id": context['dag_run'].run_id if context.get('dag_run') else "unknown",
                "phase": "evaluation",
                "data_version": "v1",
                "drift_triggered": "false",
                "base_model": "none",
                "eval_purpose": "candidate_selection",
                "eval_split": "test",
                "parent_training_run": run_id,
            }
        )
        mlflow.set_tag("mlflow.note.content", "Evaluate initial training candidate model for promotion")
        _, candidate_acc = evaluate_model_uri(data_dir=data_dir, model_uri=candidate_uri, split='test')
        mlflow.log_metrics({"test_acc": float(candidate_acc)})

    ti.xcom_push(key='candidate_acc', value=float(candidate_acc))
    return float(candidate_acc)


def compare_and_promote_model_func(**context):
    ti = context['ti']
    init_mlflow("http://mlflow:5000")

    client = MlflowClient()
    model_name = "googlenet-thai-food"
    run_id = ti.xcom_pull(task_ids='train_v1', key='return_value')
    candidate_acc = float(ti.xcom_pull(task_ids='evaluate_candidate', key='candidate_acc'))
    if not run_id:
        raise ValueError("Missing run_id from train_v1")

    data_dir = _resolve_latest_test_data_dir()
    production_acc = None
    should_promote = True

    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if prod_versions:
            production_uri = f"models:/{model_name}/Production"
            _, production_acc = evaluate_model_uri(data_dir=data_dir, model_uri=production_uri, split='test')
            should_promote = candidate_acc >= production_acc
    except Exception as error:
        print(f"No active Production baseline for comparison: {error}")

    if not should_promote:
        print(f"Skip promotion. Candidate acc={candidate_acc:.4f} < Production acc={production_acc:.4f}")
        ti.xcom_push(key='promoted', value=False)
        return False

    result = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    description = build_model_version_description(
        {
            "phase": "initial_train",
            "source_run_id": run_id,
            "data_version": "v1",
            "trigger_source": "airflow",
            "dag_id": context['dag'].dag_id,
            "task_id": context['task'].task_id,
            "airflow_run_id": context['dag_run'].run_id if context.get('dag_run') else None,
            "drift_triggered": False,
            "base_model": "none",
            "candidate_acc": candidate_acc,
            "production_acc": production_acc,
            "note": "Initial DAG promotion decision",
        }
    )
    client.update_model_version(name=model_name, version=result.version, description=description)
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Production",
        archive_existing_versions=True,
    )

    print(
        f"Promoted model version {result.version} with candidate acc={candidate_acc:.4f}"
        + (f" vs Production acc={production_acc:.4f}" if production_acc is not None else " (no prior Production)")
    )
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

preprocess_task = PythonOperator(
    task_id='preprocess_v1',
    python_callable=preprocess_v1_func,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_v1',
    python_callable=run_training_task,
    op_kwargs={
        'data_dir': '/opt/airflow/data/v1',
        'epochs': 1,
        'experiment_name': CANONICAL_EXPERIMENT_NAME,
        'run_name': 'initial_v1_run',
        'tracking_uri': 'http://mlflow:5000',
        'phase': 'initial_train',
        'trigger_source': 'airflow',
        'dag_id': 'initial_train_dag',
        'task_id': 'train_v1',
        'drift_triggered': False,
        'base_model': 'none',
        'data_version': 'v1',
        'batch_size': "{{ params.batch_size }}",
        'airflow_run_id': "{{ dag_run.run_id }}",
    },
    dag=dag,
)

evaluate_candidate_task = PythonOperator(
    task_id='evaluate_candidate',
    python_callable=evaluate_candidate_func,
    dag=dag,
)

compare_promote_task = PythonOperator(
    task_id='compare_and_promote',
    python_callable=compare_and_promote_model_func,
    dag=dag,
)

restart_fastapi_task = PythonOperator(
    task_id='restart_fastapi',
    python_callable=restart_fastapi_container_func,
    dag=dag,
)

preprocess_task >> train_task >> evaluate_candidate_task >> compare_promote_task >> restart_fastapi_task
