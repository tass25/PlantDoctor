"""
Train script using PyTorch Lightning + Weights & Biases (W&B)
All code is Lightning.ai-friendly with LightningModule inside the file.

Usage example:
  cd PlantDoctor
  python -m src.models.train --data_root data/processed --backbone resnet18 --batch_size 32 --max_epochs 10 --project plantdoctor

Before running, login to W&B:
  wandb login <your-api-key>
"""

import os
from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from model import get_backbone  # your backbone model from model.py

# ---------------------------
# LightningModule inside train.py
# ---------------------------
from torchmetrics import Accuracy
import torch.nn as nn

class PlantLightningModule(pl.LightningModule):
    def __init__(self, model, lr=1e-3, num_classes=15):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, y)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"}}

# ---------------------------
# Seed for reproducibility
# ---------------------------
SEED = 42
pl.seed_everything(SEED)

# ---------------------------
# Data transforms
# ---------------------------
def make_transforms(is_train=True, img_size=224):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

# ---------------------------
# Data loaders
# ---------------------------
def get_dataloaders(data_root, batch_size, num_workers):
    train_ds = datasets.ImageFolder(root=os.path.join(data_root, "train"), transform=make_transforms(is_train=True))
    val_ds = datasets.ImageFolder(root=os.path.join(data_root, "val"), transform=make_transforms(is_train=False))
    test_ds = datasets.ImageFolder(root=os.path.join(data_root, "test"), transform=make_transforms(is_train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.classes

# ---------------------------
# Main function
# ---------------------------
def main(args):
    train_loader, val_loader, test_loader, classes = get_dataloaders(args.data_root, args.batch_size, args.num_workers)
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")

    backbone = get_backbone(backbone_name=args.backbone, pretrained=args.pretrained, num_classes=num_classes)
    pl_module = PlantLightningModule(model=backbone, lr=args.lr, num_classes=num_classes)

    # WandB logger
    wandb_logger = WandbLogger(project=args.project, name=args.run_name, log_model="all")

    # Checkpoints + Early stopping
    ckpt_cb = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="plantdoctor-{epoch:02d}-{val_loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        every_n_epochs=1
    )
    earlystop = EarlyStopping(monitor="val/loss", patience=5, mode="min")

    # Accelerator / devices logic
    if args.gpus == 0:
        accelerator = "cpu"
        devices = 1
    else:
        accelerator = "gpu"
        devices = args.gpus

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=[ckpt_cb, earlystop],
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=20,
        gradient_clip_val=1.0
    )

    # Train
    trainer.fit(pl_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Test best checkpoint
    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt and os.path.exists(best_ckpt):
        print("Testing best checkpoint:", best_ckpt)
        trainer.test(ckpt_path=best_ckpt, dataloaders=test_loader)

# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/processed")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs (0 for CPU)")
    parser.add_argument("--project", type=str, default="plantdoctor")
    parser.add_argument("--run_name", type=str, default="exp_resnet18")
    parser.add_argument("--ckpt_dir", type=str, default="src/models/checkpoints")
    args = parser.parse_args()
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    main(args)
