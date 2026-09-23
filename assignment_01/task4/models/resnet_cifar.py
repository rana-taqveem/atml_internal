"""CIFAR-appropriate ResNet-18, used unchanged for Vanilla, GCSC and PROSER.

torchvision's ResNet-18 is built for 224x224 ImageNet inputs: it opens with a
7x7 stride-2 convolution followed by a stride-2 max-pool, which shrinks the
input eightfold before the first residual block. Applied to a 32x32 CIFAR
image that leaves a 4x4 map, and most of the spatial detail is gone before any
real processing happens.

The assignment therefore specifies the standard CIFAR variant: replace the
first convolution with 3x3 stride-1 and drop the max-pool. The four residual
stages then reduce 32x32 -> 32x32 -> 16x16 -> 8x8 -> 4x4, which is the layout
the CIFAR ResNet literature uses.

    forward(x)                      -> logits [N, 10]
    forward(x, return_features=True) -> (logits [N, 10], features [N, 512])
    forward_features(x)             -> features [N, 512]

The penultimate 512-d feature is what the Mahalanobis score uses, and the
manifold mixup in PROSER needs the network split at layer2, so the forward
pass is written in two halves (forward_pre / forward_post) that the plain
forward simply chains.
"""

import torch.nn as nn
from torchvision.models import resnet18

from assignment_01.task4.config import task_config


class CifarResNet18(nn.Module):
    """ResNet-18 adapted to 32x32 inputs, with a 10-way linear classifier."""

    def __init__(self, num_classes=None):
        super().__init__()
        num_classes = num_classes or task_config.NUM_CLASSES

        # No pretrained weights: Task 4 trains from random initialization.
        backbone = resnet18(weights=None, num_classes=num_classes)

        # The two CIFAR adaptations.
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
        """Input -> activations after layer2, where manifold mixup is applied.

        x        [N, 3, 32, 32]
        returns  [N, 128, 16, 16]
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        return self.layer2(x)

    def forward_post(self, h):
        """Activations after layer2 -> penultimate feature.

        h        [N, 128, 16, 16]
        returns  [N, 512]
        """
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
