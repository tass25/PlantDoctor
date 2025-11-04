import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from PIL import Image
import numpy as np
import io

# -----------------------
# 🔍 Model Loader
# -----------------------
def load_model(model_path="best_model.pth", model_name="resnet18", num_classes=38, device="cpu"):
    """Load trained model checkpoint."""
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("❌ Unknown model architecture")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# -----------------------
# 🌿 Preprocessing
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# -----------------------
# 🧠 Prediction + Health Score
# -----------------------
def predict_image(model, image_path, device="cpu"):
    """Predict plant species, health score, and return Grad-CAM heatmap."""
    img = Image.open(image_path).convert("RGB")
    tensor_img = transform(img).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(tensor_img)
        probs = F.softmax(outputs, dim=1)
        conf, pred_class = torch.max(probs, 1)

    # Health score = confidence * 100
    health_score = float(conf.item() * 100)

    # Grad-CAM visualization
    target_layer = "layer4" if hasattr(model, "layer4") else "features"
    cam_extractor = GradCAM(model, target_layer=target_layer)
    out = model(tensor_img)
    activation_map = cam_extractor(pred_class.item(), out)
    cam = activation_map[0].cpu().numpy()

    # Overlay mask
    result = overlay_mask(img, Image.fromarray((cam * 255).astype(np.uint8)), alpha=0.5)

    return {
        "predicted_class": int(pred_class.item()),
        "confidence": float(conf.item()),
        "health_score": round(health_score, 2),
        "gradcam_image": result
    }

# -----------------------
# 🧾 Save GradCAM
# -----------------------
def save_gradcam(result, save_path="gradcam_overlay.jpg"):
    """Save the Grad-CAM image."""
    result["gradcam_image"].save(save_path)
    print(f"✅ Grad-CAM saved at {save_path}")

if __name__ == "__main__":
    print("✅ Prediction + XAI module ready for PlantDoctor")
