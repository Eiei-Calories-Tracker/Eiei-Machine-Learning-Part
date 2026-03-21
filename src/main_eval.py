import argparse
import os
import torch
import mlflow
from src.data_utils import get_eval_loader
from src.train_engine import evaluate

def evaluate_model_uri(data_dir, model_uri, batch_size=32, split='test', device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = mlflow.pytorch.load_model(model_uri)
    model.to(device)
    eval_loader = get_eval_loader(data_dir, batch_size=batch_size, split=split)
    criterion = torch.nn.CrossEntropyLoss()
    loss, acc = evaluate(model, eval_loader, criterion, device)
    return loss, acc


def run_eval_task(data_dir, model_uri, experiment_name="ThaiFood_Initial", run_name="eval_run", tracking_uri=None, split='test', **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved_tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(resolved_tracking_uri)
    
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        loss, acc = evaluate_model_uri(data_dir=data_dir, model_uri=model_uri, batch_size=32, split=split, device=device)
        
        mlflow.log_metrics({"test_loss": loss, "test_acc": acc})
        print(f"Evaluation Acc: {acc:.4f}")
        
        # Write accuracy to file for Airflow to read
        with open("eval_acc.txt", "w") as f:
            f.write(str(acc))
            
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_uri", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, default="ThaiFood_Initial")
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
