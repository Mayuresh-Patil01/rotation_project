import torch
import torch.nn as nn
import torchvision
from torchvision.models import (
    VGG16_Weights, ResNet18_Weights,
    AlexNet_Weights, Inception_V3_Weights,
    ViT_B_16_Weights
)

class BasicModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def get_model(model_name, num_classes=4):
    model = None
    if model_name == "vgg":
        model = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif model_name == "resnet":
        model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "alexnet":
        model = torchvision.models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif model_name == "inception":
        model = torchvision.models.inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True
        )
        model.aux_logits = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vit":
        model = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name == "basic":
        model = BasicModel(num_classes)
    
    return model