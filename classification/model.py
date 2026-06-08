import torch
import torch.nn as nn
import torchvision.models as models


def build_resnet18(num_classes=10, pretrained=True, freeze_backbone=False):
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=None)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes),
    )
    
    return model


def get_model_params(model, lr=0.001, weight_decay=1e-4):
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "fc" in name:
                params.append({"params": param, "lr": lr * 10})
            else:
                params.append({"params": param, "lr": lr})
    
    return params
