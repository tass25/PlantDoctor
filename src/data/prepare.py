import os
import cv2
import random
from pathlib import Path

RAW_DIR = "data/raw/PlantVillage"  # ✅ your Kaggle dataset folder
PROCESSED_DIR = "data/processed"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
IMG_SIZE = (224, 224)

def preprocess_and_split(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR):
    os.makedirs(processed_dir, exist_ok=True)
    splits = ["train", "val", "test"]
    for s in splits:
        os.makedirs(os.path.join(processed_dir, s), exist_ok=True)

    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    print(f"✅ Found {len(classes)} classes: {classes}")

    for cls in classes:
        cls_path = os.path.join(raw_dir, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        n_test = n_total - n_train - n_val

        split_dict = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split_name, imgs in split_dict.items():
            split_dir = os.path.join(processed_dir, split_name, cls)
            os.makedirs(split_dir, exist_ok=True)
            for img_file in imgs:
                img_path = os.path.join(cls_path, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"⚠️ Skipping corrupted: {img_path}")
                        continue
                    img = cv2.resize(img, IMG_SIZE)
                    cv2.imwrite(os.path.join(split_dir, img_file), img)
                except Exception as e:
                    print(f"❌ Error {img_path}: {e}")

    print("\n🎉 Preprocessing & splitting done!")
    print(f"Processed data saved in '{processed_dir}'")

if __name__ == "__main__":
    preprocess_and_split()
