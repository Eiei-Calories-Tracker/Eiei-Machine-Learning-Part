import argparse
import os
import torch
import mlflow
from src.model import create_model
from src.data_utils import get_dataloaders
from src.train_engine import evaluate

def run_eval_task(data_dir, model_uri, experiment_name="ThaiFood_Initial", run_name="eval_run", **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        model = mlflow.pytorch.load_model(model_uri)
        model.to(device)
        
        _, test_loader = get_dataloaders(data_dir, batch_size=32)
        
        criterion = torch.nn.CrossEntropyLoss()
        loss, acc = evaluate(model, test_loader, criterion, device)
        
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
    
    args = parser.parse_args()
    
    run_eval_task(
        data_dir=args.data_dir,
        model_uri=args.model_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name
    )

if __name__ == "__main__":
    main()
