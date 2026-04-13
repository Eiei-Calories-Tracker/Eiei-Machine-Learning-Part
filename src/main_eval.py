import argparse
import torch
import mlflow
from src.data_utils import get_eval_loader
from src.train_engine import evaluate
from src.mlflow_metadata import CANONICAL_EXPERIMENT_NAME, apply_run_metadata, infer_data_version, init_mlflow

def evaluate_model_uri(data_dir, model_uri, batch_size=32, split='test', device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = mlflow.pytorch.load_model(model_uri)
    model.to(device)
    eval_loader = get_eval_loader(data_dir, batch_size=batch_size, split=split)
    criterion = torch.nn.CrossEntropyLoss()
    metrics = evaluate(model, eval_loader, criterion, device)
    return metrics


def run_eval_task(data_dir, model_uri, experiment_name=CANONICAL_EXPERIMENT_NAME, run_name="eval_run", tracking_uri=None, split='test', **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    init_mlflow(tracking_uri)
    run_tags = {
        "trigger_source": kwargs.get("trigger_source", "manual"),
        "dag_id": kwargs.get("dag_id"),
        "task_id": kwargs.get("task_id"),
        "airflow_run_id": kwargs.get("airflow_run_id"),
        "phase": kwargs.get("phase", "evaluation"),
        "data_version": kwargs.get("data_version") or infer_data_version(data_dir),
        "drift_triggered": kwargs.get("drift_triggered"),
        "base_model": kwargs.get("base_model", "n/a"),
        "eval_purpose": kwargs.get("eval_purpose", "validation"),
        "eval_split": split,
        "parent_training_run": kwargs.get("parent_training_run"),
    }
    run_description = kwargs.get(
        "run_description",
        f"phase=evaluation data_version={run_tags['data_version']} split={split} model_uri={model_uri}",
    )
    
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        apply_run_metadata(tags=run_tags, description=run_description)
        metrics = evaluate_model_uri(data_dir=data_dir, model_uri=model_uri, batch_size=32, split=split, device=device)
        
        mlflow.log_metrics({
            f"{split}_loss": metrics["loss"],
            f"{split}_acc": metrics["acc"],
            f"{split}_f1_macro": metrics["f1_macro"],
            f"{split}_recall_macro": metrics["recall_macro"],
            f"{split}_roc_auc_macro": metrics["roc_auc_macro"]
        })
        acc = metrics["acc"]
        print(f"Evaluation Acc: {acc:.4f}")
        
        # Write accuracy to file for Airflow to read
        with open("eval_acc.txt", "w") as f:
            f.write(str(acc))
            
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_uri", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, default=CANONICAL_EXPERIMENT_NAME)
    parser.add_argument("--run_name", type=str, default="eval_run")
    parser.add_argument("--tracking_uri", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    
    args = parser.parse_args()
    
    run_eval_task(
        data_dir=args.data_dir,
        model_uri=args.model_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        tracking_uri=args.tracking_uri,
        split=args.split,
    )

if __name__ == "__main__":
    main()
