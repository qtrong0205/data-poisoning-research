import torch
from torch import nn
from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights
)


def build_model(
    num_classes: int = 38,
    device: torch.device = None,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build MobileNetV2 for image classification.
    Freeze feature extractor and replace classifier.
    """

    weights = (
        MobileNet_V2_Weights.DEFAULT
        if pretrained else None
    )

    model = mobilenet_v2(weights=weights)

    # Freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_ftrs, num_classes)
    )

    if device is not None:
        model = model.to(device)

    return model
