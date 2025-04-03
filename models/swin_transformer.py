import torch
import torch.nn as nn
import timm

def get_swin(num_classes=5):
    model = timm.create_model("swin_large_patch4_window7_224", pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.head = nn.Linear(model.head.in_features, num_classes)
    return model
