import torch
import torch.nn as nn
import timm

def get_medvit(num_classes=5):
    model = timm.create_model("medvit_large", pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.head = nn.Linear(model.head.in_features, num_classes)
    return model
