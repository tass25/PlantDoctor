import torch
import torch.nn as nn
import torchvision.models as tv_models

def get_backbone(backbone_name: str = "resnet18", pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    """
    Returns a classification model with the final classifier adapted to num_classes.
    Supported backbone_name: "resnet18", "efficientnet_b0"
    """
    backbone_name = backbone_name.lower()

    if backbone_name == "resnet18":
        model = tv_models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        assert isinstance(model.fc, nn.Linear), "Expected model.fc to be nn.Linear"
        in_features: int = model.fc.in_features  # type: ignore
        model.fc = nn.Linear(in_features, num_classes)

    elif backbone_name == "efficientnet_b0":
        model = tv_models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        if isinstance(model.classifier, nn.Sequential):
            last_linear = model.classifier[-1]
            if isinstance(last_linear, nn.Linear):
                in_features: int = last_linear.in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                raise TypeError("Unexpected layer type in EfficientNet classifier.")
        else:
            raise TypeError("Unexpected EfficientNet classifier structure.")

    else:
        raise ValueError(f"Backbone '{backbone_name}' not supported.")

    return model
