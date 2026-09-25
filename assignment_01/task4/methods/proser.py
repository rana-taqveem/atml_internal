"""PROSER classifier and manifold-mixup data placeholders.

Based on Zhou et al. (2021), "Learning Placeholders for Open-Set Recognition".
No CIFAR-100 image is used during training.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from assignment_01.task4.config import task_config


class ProserNet(nn.Module):
    """Append dummy classifier units to a trained CIFAR ResNet."""

    def __init__(self, backbone, num_dummy=None):
        super().__init__()
        self.backbone = backbone
        self.num_dummy = num_dummy or task_config.PROSER_NUM_DUMMY
        self.num_known = backbone.num_classes
        self.dummy_classifier = nn.Linear(backbone.feature_dim, self.num_dummy)

    def logits_from_features(self, features):
        """features [N, 512] -> [N, 10 + C_dummy]."""
        known = self.backbone.classifier(features)      # [N, 10]
        dummy = self.dummy_classifier(features)         # [N, C_dummy]
        return torch.cat([known, dummy], dim=1)

    def forward(self, x, return_features=False):
        features = self.backbone.forward_features(x)    # [N, 512]
        logits = self.logits_from_features(features)    # [N, 10 + C_dummy]
        if return_features:
            return logits, features
        return logits

    def forward_mixed(self, x, permutation, lam):
        """Return logits after layer2 manifold mixup."""
        h = self.backbone.forward_pre(x)                # [N, 128, 16, 16]
        mixed = lam * h + (1.0 - lam) * h[permutation]  # [N, 128, 16, 16]
        features = self.backbone.forward_post(mixed)    # [N, 512]
        return self.logits_from_features(features)


def classifier_placeholder_loss(logits, labels, num_known, beta=None):
    """Keep the known target first and a dummy second."""
    beta = task_config.PROSER_BETA if beta is None else beta

    closed_set = F.cross_entropy(logits, labels)

    masked = logits.clone()
    masked.scatter_(1, labels.view(-1, 1), float("-inf"))   # remove the true class
    best_dummy = masked[:, num_known:].argmax(dim=1) + num_known   # [N]
    placeholder = F.cross_entropy(masked, best_dummy)

    return closed_set + beta * placeholder


def data_placeholder_loss(mixed_logits, num_known, gamma=None):
    """Train between-class interpolations toward a dummy classifier."""
    gamma = task_config.PROSER_GAMMA if gamma is None else gamma
    best_dummy = mixed_logits[:, num_known:].argmax(dim=1) + num_known
    return gamma * F.cross_entropy(mixed_logits, best_dummy)


def different_class_permutation(labels, generator=None):
    """Return a random pairing and mask out same-class pairs."""
    batch_size = labels.size(0)
    if generator is None:
        permutation = torch.randperm(batch_size, device=labels.device)
    else:
        permutation = torch.randperm(batch_size, generator=generator).to(labels.device)
    usable = labels != labels[permutation]
    return permutation, usable


def placeholder_detection_score(logits, num_known):
    """Return max dummy logit minus max known logit."""
    known = logits[:, :num_known].max(axis=1)
    dummy = logits[:, num_known:].max(axis=1)
    return dummy - known


def known_logits(logits, num_known):
    """The ten known columns, for CSA and for MLS comparable with Vanilla."""
    return logits[:, :num_known]


def train_one_epoch(model, loader, optimizer, device=None, beta=None, gamma=None):
    """Train classifier and data placeholders on separate batch halves."""
    device = device or task_config.DEVICE
    model.train()

    totals = {"loss": 0.0, "classifier": 0.0, "data": 0.0}
    correct = 0
    seen = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        half = images.size(0) // 2
        if half < 2:
            continue

        first_images, first_labels = images[:half], labels[:half]
        second_images, second_labels = images[half:], labels[half:]

        optimizer.zero_grad(set_to_none=True)

        # Half 1: classifier placeholders on real examples.
        logits = model(first_images)                        # [half, 10 + C_dummy]
        classifier_loss = classifier_placeholder_loss(
            logits, first_labels, model.num_known, beta=beta)

        # Half 2: data placeholders on between-class manifold mixup.
        permutation, usable = different_class_permutation(second_labels)
        if usable.any():
            lam = float(np.random.beta(task_config.PROSER_MIXUP_ALPHA,
                                       task_config.PROSER_MIXUP_ALPHA))
            mixed_logits = model.forward_mixed(second_images, permutation, lam)
            data_loss = data_placeholder_loss(
                mixed_logits[usable], model.num_known, gamma=gamma)
        else:
            data_loss = logits.new_zeros(())

        loss = classifier_loss + data_loss
        loss.backward()
        optimizer.step()

        batch = first_images.size(0)
        totals["loss"] += loss.item() * batch
        totals["classifier"] += classifier_loss.item() * batch
        totals["data"] += float(data_loss) * batch
        correct += (logits[:, :model.num_known].argmax(1) == first_labels).sum().item()
        seen += batch

    result = {key: value / max(seen, 1) for key, value in totals.items()}
    result["accuracy"] = 100.0 * correct / max(seen, 1)
    return result
