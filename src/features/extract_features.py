import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torchvision import models

def extract_features(model, dataloader, device='cuda'):
    """
    Extract CNN features from the model's penultimate layer.
    """
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="🔍 Extracting features"):
            imgs = batch["image"].to(device)
            lbls = batch["label"].cpu().numpy()
            with torch.no_grad():
                feats = model(imgs)
            features.append(feats.cpu().numpy())
            labels.extend(lbls)
    features = np.concatenate(features, axis=0)
    labels = np.array(labels)
    print(f"✅ Features shape: {features.shape}")
    return features, labels

def save_features(features, labels, out_path="data/features/features.npy"):
    """Save extracted embeddings and labels."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, {"features": features, "labels": labels})
    print(f"✅ Features saved to {out_path}")

def visualize_embeddings(features, labels, num_samples=500):
    """Visualize embeddings using PCA (2D projection)."""
    idx = np.random.choice(len(features), min(num_samples, len(features)), replace=False)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features[idx])
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(reduced[:,0], reduced[:,1], c=labels[idx], cmap="viridis", s=15)
    plt.title("🌿 PCA Visualization of CNN Features")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter)
    plt.show()

if __name__ == "__main__":
    print("This module is for extracting and visualizing CNN features.")
