import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Paths to your models
MODEL_PATHS = {
    "mobilenet": "models/mobilenet_model.keras",
    "cnn": "models/cnn_model.keras"
}

# Fix image size for both models
IMG_SIZE = (224, 224)

def load_model(model_name):
    """
    Load MobileNet or CNN model.
    """
    path = MODEL_PATHS.get(model_name.lower())

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    print(f"[INFO] Loading model: {model_name}...")
    return tf.keras.models.load_model(path)

def preprocess(img):
    """
    Convert uploaded image to tensor ready for prediction.
    """
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(model_name, img):
    """
    Predict class of input image using the selected model.
    """
    model = load_model(model_name)
    processed = preprocess(img)

    preds = model.predict(processed)
    prob = float(np.max(preds))
    label = int(np.argmax(preds))

    return {
        "model_used": model_name,
        "label": label,
        "probability": prob
    }
