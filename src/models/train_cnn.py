import os
import torch
import mlflow
import mlflow.pytorch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
from src.data.dataset import PlantDoctorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

def get_model(model_name="resnet18", num_classes=38, pretrained=True):
    """Return a CNN backbone with customized classification head."""
    if model_name == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(" Unknown model name")
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10, exp_name="PlantDoctor_Training"):
    """Train model with MLflow tracking."""
    mlflow.set_experiment(exp_name)
    best_val_acc = 0.0

    with mlflow.start_run():
        mlflow.log_params({
            "model": model.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
            "epochs": epochs,
            "device": device
        })

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            train_preds, train_labels = [], []

            for batch in tqdm(train_loader, desc=f"🌱 Epoch {epoch+1}/{epochs}"):
                imgs, labels = batch["image"].to(device), batch["label"].to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                train_preds.extend(outputs.argmax(1).cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

            train_acc = accuracy_score(train_labels, train_preds)
            val_acc, val_loss = evaluate_model(model, val_loader, criterion, device)

            mlflow.log_metrics({
                "train_loss": running_loss / len(train_loader),
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, step=epoch)

            print(f"✅ Epoch {epoch+1}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "best_model.pth")
                mlflow.log_artifact("best_model.pth")

        mlflow.pytorch.log_model(model, "model")

def evaluate_model(model, loader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    val_loss, preds, labels = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            imgs, lbls = batch["image"].to(device), batch["label"].to(device)
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            val_loss += loss.item()
            preds.extend(outputs.argmax(1).cpu().numpy())
            labels.extend(lbls.cpu().numpy())
    acc = accuracy_score(labels, preds)
    return acc, val_loss / len(loader)

if __name__ == "__main__":
    print(" Training module ready for PlantDoctor")
