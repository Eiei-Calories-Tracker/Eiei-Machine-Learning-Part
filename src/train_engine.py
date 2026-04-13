import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import time
from src.mlflow_metadata import apply_run_metadata
from sklearn.metrics import f1_score, recall_score, roc_auc_score
import numpy as np

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        # GoogLeNet returns (logits, aux2, aux1) during training if aux_logits=True
        if isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            logits = outputs.logits
            
        loss = criterion(logits, labels)
        
        # We could also add aux losses here if we wanted to follow GoogLeNet paper, 
        # but for simplicity and matching the main goal, we focus on main logits.
        if not isinstance(outputs, torch.Tensor):
            loss_aux1 = criterion(outputs.aux_logits1, labels)
            loss_aux2 = criterion(outputs.aux_logits2, labels)
            loss = loss + 0.3 * loss_aux1 + 0.3 * loss_aux2

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / total, correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            # For metrics
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    avg_loss = running_loss / total
    avg_acc = correct / total
    
    # Calculate advanced metrics
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    try:
        # One-vs-Rest for multi-class ROC AUC
        roc_auc_macro = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovr')
    except Exception as e:
        print(f"ROC AUC calculation failed (likely too few samples for some classes): {e}")
        roc_auc_macro = 0.0

    return {
        "loss": avg_loss,
        "acc": avg_acc,
        "f1_macro": f1_macro,
        "recall_macro": recall_macro,
        "roc_auc_macro": roc_auc_macro
    }

def run_training(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    device,
    experiment_name,
    run_name,
    run_tags=None,
    run_description=None,
    extra_params=None,
):
    # Disable autologging to prevent compatibility issues with server version
    mlflow.autolog(disable=True)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        apply_run_metadata(tags=run_tags, description=run_description)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

        params_to_log = {
            "epochs": epochs,
            "lr": lr,
            "optimizer": "Adam",
            "batch_size": train_loader.batch_size,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
        }
        if extra_params:
            params_to_log.update(extra_params)
        mlflow.log_params(params_to_log)
        
        best_acc = 0.0
        
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            metrics = evaluate(model, val_loader, criterion, device)
            
            val_loss = metrics["loss"]
            val_acc = metrics["acc"]
            
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1_macro": metrics["f1_macro"],
                "val_recall_macro": metrics["recall_macro"],
                "val_roc_auc_macro": metrics["roc_auc_macro"]
            }, step=epoch)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                # Save locally as a primary/fallback mechanism
                local_save_path = "best_model_local.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'num_classes': model.fc[1].out_features if isinstance(model.fc, nn.Sequential) else model.fc.out_features
                }, local_save_path)
                print(f"Model saved locally to {local_save_path}")

                # Attempt MLflow logging
                try:
                    mlflow.pytorch.log_model(model, "model")
                except Exception as e:
                    print(f"Warning: MLflow model logging failed (Permission/API issue), but training continues: {e}")
                
        return model, best_acc, run_id
