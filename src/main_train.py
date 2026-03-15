import argparse
import os
import torch
import mlflow
from src.model import create_model
from src.data_utils import get_dataloaders
from src.train_engine import run_training

def run_training_task(data_dir, epochs=1, lr=0.0005, batch_size=64, experiment_name="ThaiFoodClassification", run_name="train_run", base_model_uri=None, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load or create model
    if base_model_uri:
        print(f"Loading base model from {base_model_uri} for finetuning...")
        model = mlflow.pytorch.load_model(base_model_uri)
    else:
        print("Creating fresh GoogLeNet model...")
        model = create_model()
    
    train_loader, val_loader = get_dataloaders(data_dir, batch_size=batch_size)
    
    trained_model, best_acc = run_training(
        model, 
        train_loader, 
        val_loader, 
        epochs=epochs, 
        lr=lr, 
        device=device,
        experiment_name=experiment_name,
        run_name=run_name
    )
    
    print(f"Training finished. Best Val Acc: {best_acc:.4f}")
    return best_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--experiment_name", type=str, default="ThaiFoodClassification")
    parser.add_argument("--run_name", type=str, default="train_run")
    parser.add_argument("--model_name", type=str, default="googlenet-thai-food")
    parser.add_argument("--base_model_uri", type=str, help="MLflow model URI to finetune from")
    
    args = parser.parse_args()
    
    run_training_task(
        data_dir=args.data_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        base_model_uri=args.base_model_uri
    )

if __name__ == "__main__":
    main()
