from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import mlflow
from mlflow.tracking import MlflowClient

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
    # Try to get run_id from configuration
    run_id = '1227aa489b1c4eee8cf5cbb73ec9d4a6'
    if not run_id:
        raise Exception("Please provide 'run_id' in DAG configuration. Example: {\"run_id\": \"your-run-id-here\"}")
    
    client = MlflowClient()
    model_name = "googlenet-thai-food"
    model_uri = f"runs:/{run_id}/model"
    
    print(f"Registering model from run {run_id} as '{model_name}'...")
    
    try:
        # Register the model
        result = mlflow.register_model(model_uri, model_name)
        version = result.version
        
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
