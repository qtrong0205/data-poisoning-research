import torch
from torch import nn
from torchvision.models import mobilenet_v2

def build_model(num_classes: int = 38, device: torch.device = None) -> nn.Module:
    """
    Load pretrained MobileNetV2, freeze feature extractor, 
    replace classifier for the target number of classes.
    """
    model = mobilenet_v2(pretrained=True)

    # Freeze every feature extractor
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier[1].in_features   # = 1280

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=num_ftrs, out_features=num_classes),
    )

    # Send model to the target device
    if device:
        model = model.to(device)

    return model
    