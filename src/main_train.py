import argparse
import torch
import mlflow
from src.model import create_model
from src.data_utils import get_dataloaders
from src.train_engine import run_training
from src.mlflow_metadata import CANONICAL_EXPERIMENT_NAME, infer_data_version, init_mlflow

def run_training_task(data_dir, epochs=1, lr=0.0005, batch_size=16, experiment_name=CANONICAL_EXPERIMENT_NAME, run_name="train_run", base_model_uri=None, tracking_uri=None, **kwargs):
    batch_size = int(batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved_tracking_uri = init_mlflow(tracking_uri)

    dag = kwargs.get("dag")
    task_instance = kwargs.get("task_instance")
    dag_run = kwargs.get("dag_run")

    run_tags = {
        "trigger_source": kwargs.get("trigger_source", "airflow" if dag_run else "manual"),
        "dag_id": kwargs.get("dag_id") or (dag.dag_id if dag else None),
        "task_id": kwargs.get("task_id") or (task_instance.task_id if task_instance else None),
        "airflow_run_id": kwargs.get("airflow_run_id") or (dag_run.run_id if dag_run else None),
        "phase": kwargs.get("phase", "retrain" if base_model_uri else "initial_train"),
        "data_version": kwargs.get("data_version") or infer_data_version(data_dir),
        "drift_triggered": kwargs.get("drift_triggered"),
        "base_model": base_model_uri or "none",
    }
    run_description = kwargs.get(
        "run_description",
        f"phase={run_tags['phase']} data_version={run_tags['data_version']} trigger={run_tags['trigger_source']}",
    )
    extra_params = {"data_version": run_tags["data_version"]}
    
    # Load or create model
    if base_model_uri:
        print(f"Loading base model from {base_model_uri} for finetuning...")
        model = mlflow.pytorch.load_model(base_model_uri)
    else:
        print("Creating fresh GoogLeNet model...")
        model = create_model()
    
    train_loader, val_loader = get_dataloaders(data_dir, batch_size=batch_size)
    
    trained_model, best_acc, run_id = run_training(
        model, 
        train_loader, 
        val_loader, 
        epochs=epochs, 
        lr=lr, 
        device=device,
        experiment_name=experiment_name,
        run_name=run_name,
        run_tags=run_tags,
        run_description=run_description,
        extra_params=extra_params,
    )
    
    print(f"Training finished. Best Val Acc: {best_acc:.4f}, Run ID: {run_id}")
    return run_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--experiment_name", type=str, default=CANONICAL_EXPERIMENT_NAME)
    parser.add_argument("--run_name", type=str, default="train_run")
    parser.add_argument("--model_name", type=str, default="googlenet-thai-food")
    parser.add_argument("--base_model_uri", type=str, help="MLflow model URI to finetune from")
    parser.add_argument("--tracking_uri", type=str, default=None)
    
    args = parser.parse_args()
    
    run_training_task(
        data_dir=args.data_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        base_model_uri=args.base_model_uri,
        tracking_uri=args.tracking_uri,
    )

if __name__ == "__main__":
    main()
