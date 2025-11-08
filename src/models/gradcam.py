"""
Generate Grad-CAM heatmaps for a single image or a folder of images.

Dependencies:
  pip install pytorch-grad-cam
Usage example:
  python src/models/gradcam.py --ckpt src/models/checkpoints/plantdoctor-epoch=04-val_loss=0.1234.ckpt \
                              --image path/to/image.jpg \
                              --output out.png
"""
import argparse
import os
import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import get_backbone

def load_model_from_ckpt(ckpt_path, backbone_name, num_classes, device="cpu"):
    # We assume you saved a plain model checkpoint (if you used PL checkpoint, adapt accordingly)
    # Here we build backbone and load state_dict if available
    model = get_backbone(backbone_name=backbone_name, pretrained=False, num_classes=num_classes)
    # If it's a PL checkpoint, load 'state_dict' inside; try both possibilities:
    try:
        sd = torch.load(ckpt_path, map_location=device)
        if "state_dict" in sd:
            state_dict = sd["state_dict"]
            # PL checkpoints have keys like "model.model.layer..."
            # Try to strip "model." prefix if present
            new_sd = {}
            for k, v in state_dict.items():
                new_k = k
                if k.startswith("model."):
                    new_k = k.replace("model.", "", 1)
                new_sd[new_k] = v
            model.load_state_dict(new_sd, strict=False)
        else:
            model.load_state_dict(sd, strict=False)
    except Exception as e:
        print("Could not load checkpoint normally:", e)
    model.eval()
    return model

def preprocess_image(img_path, img_size=224):
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img) / 255.0
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    inp = transform(img).unsqueeze(0)
    return img_np, inp

def run_gradcam(ckpt, image_path, output_path, backbone="resnet18", num_classes=38, target_category=None, device="cpu"):
    img_np, inp_tensor = preprocess_image(image_path)
    model = load_model_from_ckpt(ckpt, backbone, num_classes, device=device)
    model.to(device)

    # Choose target layer automatically (ResNet last conv)
    if backbone.startswith("resnet"):
        target_layer = model.layer4[-1]
    elif backbone.startswith("efficientnet"):
        # efficientnet_b0 has features[-1]
        target_layer = model.features[-1]
    else:
        raise ValueError("Unknown backbone for target layer selection")

    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device!="cpu"))
    # Run forward to get logits and predicted class
    with torch.no_grad():
        outputs = model(inp_tensor.to(device))
        pred = int(outputs.argmax(dim=1).item())
    targets = None
    if target_category is not None:
        targets = [ClassifierOutputTarget(target_category)]

    grayscale_cam = cam(input_tensor=inp_tensor.to(device), targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"Saved Grad-CAM to {output_path}. Predicted class: {pred}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--num_classes", type=int, default=38)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_gradcam(args.ckpt, args.image, args.output, backbone=args.backbone, num_classes=args.num_classes, device=args.device)
