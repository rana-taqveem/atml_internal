"""CIFAR ResNet-18 with a 3x3 stem and no initial max-pooling.

The layer2 split supports PROSER manifold mixup; features are 512-dimensional.
"""

import torch.nn as nn
from torchvision.models import resnet18

from assignment_01.task4.config import task_config


class CifarResNet18(nn.Module):
    """ResNet-18 adapted to 32x32 inputs, with a 10-way linear classifier."""

    def __init__(self, num_classes=None):
        super().__init__()
        num_classes = num_classes or task_config.NUM_CLASSES

        backbone = resnet18(weights=None, num_classes=num_classes)

        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.layer1 = backbone.layer1          # [N,  64, 32, 32]
        self.layer2 = backbone.layer2          # [N, 128, 16, 16]
        self.layer3 = backbone.layer3          # [N, 256,  8,  8]
        self.layer4 = backbone.layer4          # [N, 512,  4,  4]
        self.avgpool = backbone.avgpool        # [N, 512,  1,  1]
        self.classifier = nn.Linear(512, num_classes)

        self.feature_dim = 512
        self.num_classes = num_classes

    def forward_pre(self, x):
        """Return layer2 activations [N, 128, 16, 16]."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        return self.layer2(x)

    def forward_post(self, h):
        """Map layer2 activations to penultimate features [N, 512]."""
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.avgpool(h)                    # [N, 512, 1, 1]
        return h.flatten(1)                    # [N, 512]

    def forward_features(self, x):
        """x [N,3,32,32] -> penultimate feature [N, 512]."""
        return self.forward_post(self.forward_pre(x))

    def forward(self, x, return_features=False):
        features = self.forward_features(x)    # [N, 512]
        logits = self.classifier(features)     # [N, 10]
        if return_features:
            return logits, features
        return logits


def build_model(num_classes=None, device=None):
    model = CifarResNet18(num_classes=num_classes)
    return model.to(device or task_config.DEVICE)
