import os
import random
import json
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

def get_augmentations(img_size=224):
    """Return a composed Albumentations augmentation pipeline."""
    augmentations = A.Compose([
        A.RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return augmentations

def preview_augmentations(image_path, num_samples=5, img_size=224):
    """Visualize multiple augmented samples for one image."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_size, img_size))
    img_np = np.array(img)
    aug = get_augmentations(img_size=img_size)

    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        augmented = aug(image=img_np)["image"]
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(augmented.permute(1, 2, 0).numpy() * 0.5 + 0.5)
        plt.axis("off")
    plt.show()

def save_augmentation_config(config_path="data/augmentation_config.json"):
    """Save augmentation config for DVC/MLOps reproducibility."""
    aug_config = {
        "RandomResizedCrop": {"scale": [0.8, 1.0]},
        "HorizontalFlip": 0.5,
        "VerticalFlip": 0.2,
        "RandomRotate90": 0.3,
        "ColorJitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1},
        "GaussianBlur": 0.2,
        "Normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(aug_config, f, indent=4)
    print(f"✅ Augmentation config saved to {config_path}")

if __name__ == "__main__":
    save_augmentation_config()
