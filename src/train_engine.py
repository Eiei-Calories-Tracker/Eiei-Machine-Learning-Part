import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import time

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
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / total, correct / total

def run_training(model, train_loader, val_loader, epochs, lr, device, experiment_name, run_name):
    # Disable autologging to prevent compatibility issues with server version
    mlflow.autolog(disable=True)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        
        mlflow.log_params({
            "epochs": epochs,
            "lr": lr,
            "optimizer": "Adam",
            "batch_size": train_loader.batch_size
        })
        
        best_acc = 0.0
        
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
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
