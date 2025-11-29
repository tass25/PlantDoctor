# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import wandb
from wandb.keras import WandbCallback
import json
import time

# -----------------------------
# W&B Setup
# -----------------------------
wandb.init(project="PlantDoctor_CNN")
wandb.config.update({"epochs":50, "batch_size":32, "lr":0.001})

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
# Build CNN Model
# -----------------------------
def create_optimized_cnn(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=img_size + (3,)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

cnn_model = create_optimized_cnn(train_generator.num_classes)

# -----------------------------
# Compile
# -----------------------------
lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=wandb.config.lr,
    decay_steps=10000,
    decay_rate=0.9
)
optimizer = optimizers.Adam(learning_rate=lr_schedule)

cnn_model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# -----------------------------
# Callbacks
# -----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('cnn_model.keras', monitor='val_loss', save_best_only=True)

# -----------------------------
# Train
# -----------------------------
history = cnn_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=wandb.config.epochs,
    callbacks=[early_stop, checkpoint, WandbCallback()]
)

# -----------------------------
# Evaluate
# -----------------------------
test_loss, test_acc, test_prec, test_recall = cnn_model.evaluate(test_generator)
print(f"CNN Test Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_recall:.4f}")

# -----------------------------
# Save final model
# -----------------------------
os.makedirs('./models', exist_ok=True)
timestamp = int(time.time())
cnn_model.save(f'./models/cnn_model_{timestamp}.keras')

