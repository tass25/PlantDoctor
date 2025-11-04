import os
import zipfile
import random
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def download_dataset(kaggle_dataset="laboutess/completedataset", output_dir="data/raw"):
    """Download dataset from Kaggle."""
    os.makedirs(output_dir, exist_ok=True)
    os.system(f"kaggle datasets download -d {kaggle_dataset} -p {output_dir}")
    zip_path = Path(output_dir) / "completedataset.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        zip_path.unlink()
    print(f"✅ Dataset downloaded and extracted to {output_dir}")

def prepare_dataset(raw_dir="data/raw", processed_dir="data/processed", img_size=(224, 224)):
    """Resize, normalize, and split images into train/val/test."""
    os.makedirs(processed_dir, exist_ok=True)
    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_dir = Path(processed_dir) / "train"
    val_dir = Path(processed_dir) / "val"
    test_dir = Path(processed_dir) / "test"
    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    for cls in tqdm(classes, desc="Processing classes"):
        img_paths = list(Path(raw_dir, cls).glob("*"))
        train_imgs, test_imgs = train_test_split(img_paths, test_size=0.2, random_state=42)
        val_imgs, test_imgs = train_test_split(test_imgs, test_size=0.5, random_state=42)
        for split_name, imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            out_dir = Path(processed_dir, split_name, cls)
            os.makedirs(out_dir, exist_ok=True)
            for img_path in imgs:
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = transform(img)
                    img_pil = transforms.ToPILImage()(img)
                    img_pil.save(out_dir / img_path.name)
                except Exception as e:
                    print(f"❌ Error with {img_path}: {e}")
    print(f"✅ Dataset prepared at {processed_dir}")

if __name__ == "__main__":
    download_dataset()
    prepare_dataset()
