
import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 37

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


def create_model():

    # Load pretrained MobileNetV3-Small
    weights = models.MobileNet_V3_Small_Weights.DEFAULT

    model = models.mobilenet_v3_small(
        weights=weights
    )

    # Get input features of final layer
    num_features = model.classifier[3].in_features

    # Replace final layer with 37 classes
    model.classifier[3] = nn.Linear(
        num_features,
        NUM_CLASSES
    )

    return model.to(device)
