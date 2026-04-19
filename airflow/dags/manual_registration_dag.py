from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import mlflow
from mlflow.tracking import MlflowClient
from src.mlflow_metadata import build_model_version_description, init_mlflow

default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'start_date': datetime(2026, 3, 14),
    'retries': 0,
}

dag = DAG(
    'manual_registration_dag',
    default_args=default_args,
    description='Manually register and promote a run_id to Production',
    schedule_interval=None,
    catchup=False,
)

def manual_register_func(**context):
    init_mlflow("http://mlflow:5000")
    # Try to get run_id from configuration (Trigger DAG w/ config)
    dag_run = context.get('dag_run')
    run_id = None
    if dag_run and dag_run.conf:
        run_id = dag_run.conf.get('run_id')
    
    # Fallback to a default if you want, but better to require it
    if not run_id:
        # For backward compatibility with your last successful run if you just hit 'Trigger'
        run_id = 'ba3e618906d04263a1bd083811b66a0f'
        print(f"No run_id provided in config, falling back to: {run_id}")
    
    client = MlflowClient()
    model_name = "googlenet-thai-food"
    model_uri = f"runs:/{run_id}/model"
    
    print(f"Registering model from run {run_id} as '{model_name}'...")
    
    try:
        # Register the model
        result = mlflow.register_model(model_uri, model_name)
        version = result.version

        description = build_model_version_description(
            {
                "phase": "manual_registration",
                "source_run_id": run_id,
                "data_version": "unknown",
                "trigger_source": "manual",
                "dag_id": context['dag'].dag_id,
                "task_id": context['task'].task_id,
                "airflow_run_id": dag_run.run_id if dag_run else None,
                "drift_triggered": False,
                "base_model": "user-selected-run",
                "note": "Manual registration workflow promoted supplied run_id",
            }
        )
        client.update_model_version(name=model_name, version=version, description=description)
        
        # Promote to Production
        print(f"Promoting version {version} to Production stage...")
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Success! Model version {version} is now in Production.")
    except Exception as e:
        print(f"Error during registration: {e}")
        raise e

register_task = PythonOperator(
    task_id='manual_register_and_promote',
    python_callable=manual_register_func,
    params={"run_id": ""}, # Default empty, user must provide
    dag=dag,
)
