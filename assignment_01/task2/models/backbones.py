"""Fine-tuned ResNet-18 backbone with a seven-class PACS head."""

import torch.nn as nn
from torchvision import models

from assignment_01.task2.config import task_config


class PacsResNet18(nn.Module):
    """torchvision ResNet-18 (IMAGENET1K_V1) with its classifier replaced."""

    def __init__(self, num_classes=None, pretrained=True):
        super().__init__()
        num_classes = num_classes or task_config.NUM_CLASSES

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        self.feature_dim = backbone.fc.in_features      # 512 for ResNet-18
        backbone.fc = nn.Identity()                     # keep the penultimate feature
        self.backbone = backbone
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x):
        """Return the 512-dimensional feature before the classifier."""
        return self.backbone(x)

    def forward(self, x, return_features=False):
        """x [N, 3, 224, 224] -> logits [N, 7], optionally with features [N, 512]."""
        features = self.forward_features(x)          # [N, 512]
        logits = self.classifier(features)           # [N, 7]  (7 = PACS classes)
        return (logits, features) if return_features else logits


def build_model(num_classes=None, pretrained=True, device=None):
    model = PacsResNet18(num_classes=num_classes, pretrained=pretrained)
    return model.to(device or task_config.DEVICE)
