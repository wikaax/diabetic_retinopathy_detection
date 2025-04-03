import sys
sys.path.append("external/RETFound_MAE")

from external.RETFound_MAE.models_vit import RETFound_mae
import torch
import torch.nn as nn

def get_retfound(num_classes=5):
    model = RETFound_mae()

    for param in model.parameters():
        param.requires_grad = False

    model.head = nn.Linear(model.head.in_features, num_classes)

    return model
