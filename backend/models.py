import base64
import io
import numpy as np
from PIL import Image
from keras.models import load_model  # plain Keras, no tensorflow

# Load your MobileNet model
model = load_model("models/mobilenet_model.keras")

# Dataset classes
CLASS_NAMES = [
    "Grape___Black_rot",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
]

# Disease metadata
DISEASE_INFO = {
    "Grape___Black_rot": {
        "pretty": "Black Rot",
        "symptoms": ["Circular black lesions", "Drying of infected areas"],
        "causes": ["Fungal infection (Guignardia bidwellii)"],
        "prevention": ["Remove infected debris", "Prune for airflow"],
        "treatment": ["Apply fungicides", "Remove infected leaves"]
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "pretty": "Leaf Blight",
        "symptoms": ["Brown irregular spots", "Premature leaf drop"],
        "causes": ["Isariopsis fungus"],
        "prevention": ["Improve crop spacing", "Avoid wet foliage"],
        "treatment": ["Fungicide spray", "Remove damaged leaves"]
    },
    "Grape___Esca_(Black_Measles)": {
        "pretty": "Esca (Black Measles)",
        "symptoms": ["Tiger-stripe leaf pattern", "Berry spotting"],
        "causes": ["Chronic fungal wood infection"],
        "prevention": ["Avoid pruning in humid weather", "Maintain vine health"],
        "treatment": ["Remove infected wood", "Apply protective fungicides"]
    },
    "Grape___healthy": {
        "pretty": "Healthy Plant",
        "symptoms": ["No visible disease signs"],
        "causes": ["Good practices & proper care"],
        "prevention": ["Maintain watering schedule", "Monitor regularly"],
        "treatment": ["No treatment needed"]
    }
}

def preprocess_image(image_base64):
    image_data = base64.b64decode(image_base64.split(",")[1])
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def predict_image(image_base64):
    img = preprocess_image(image_base64)
    preds = model.predict(img)[0]

    top_idx = np.argmax(preds)
    top_class = CLASS_NAMES[top_idx]
    top_confidence = float(preds[top_idx])

    # Top 3 predictions
    top3_idx = preds.argsort()[-3:][::-1]
    top3 = [
        {
            "name": DISEASE_INFO[CLASS_NAMES[i]]["pretty"],
            "confidence": float(preds[i]),
            "emoji": "🌿"
        }
        for i in top3_idx
    ]

    # Severity emoji
    if top_confidence > 0.75:
        emoji = "🔴"
        severity = "high"
    elif top_confidence > 0.45:
        emoji = "🟡"
        severity = "medium"
    else:
        emoji = "🟢"
        severity = "low"

    return {
        "disease": DISEASE_INFO[top_class]["pretty"],
        "raw_class": top_class,
        "confidence": top_confidence,
        "severity": severity,
        "emoji": emoji,
        "bestModel": "MobileNetV2",
        "predictions": top3,
        "recommendations": {
            "symptoms": DISEASE_INFO[top_class]["symptoms"],
            "causes": DISEASE_INFO[top_class]["causes"],
            "prevention": DISEASE_INFO[top_class]["prevention"],
            "treatment": DISEASE_INFO[top_class]["treatment"],
        }
    }
