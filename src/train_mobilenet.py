# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import wandb
from wandb.keras import WandbCallback
import json
import time

# -----------------------------
# W&B Setup
# -----------------------------
wandb.init(project="PlantDoctor_MobileNet")
wandb.config.update({"epochs":20, "batch_size":32, "lr":0.0001})

# -----------------------------
# Data directories (local paths)
# -----------------------------
train_dir = './newData/train_split'
valid_dir = './newData/valid_split'
test_dir = './newData/test_split'

# -----------------------------
# Data Generators
# -----------------------------
img_size = (224,224)
batch_size = wandb.config.batch_size

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# -----------------------------
# Build MobileNetV2 model
# -----------------------------
def build_mobilenet(num_classes):
    base_model = MobileNetV2(
        input_shape=img_size + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=wandb.config.lr),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

mobilenet_model = build_mobilenet(train_generator.num_classes)
mobilenet_model.summary()

# -----------------------------
# Callbacks
# -----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('mobilenet_model.keras', monitor='val_loss', save_best_only=True)

# -----------------------------
# Train the model
# -----------------------------
mobilenet_history = mobilenet_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=wandb.config.epochs,
    callbacks=[early_stop, checkpoint, WandbCallback()]
)

# -----------------------------
# Evaluate on Test Set
# -----------------------------
test_loss, test_acc, test_prec, test_recall = mobilenet_model.evaluate(test_generator)
print(f"MobileNet Test Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_recall:.4f}")

# -----------------------------
# Save final model
# -----------------------------
os.makedirs('./models', exist_ok=True)
timestamp = int(time.time())
mobilenet_model.save(f'./models/mobilenet_model_{timestamp}.keras')

# -----------------------------
# Save class names
# -----------------------------
class_names = {v:k for k,v in train_generator.class_indices.items()}
with open('./models/class_names_mobilenet.json', 'w') as f:
    json.dump(class_names, f)
