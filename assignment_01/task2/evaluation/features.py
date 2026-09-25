"""Collect features and predictions from a frozen backbone."""

import numpy as np
import torch

from assignment_01.task2.scripts.main import DEVICE


@torch.no_grad()
def collect_features(model, loaders, max_per_group=None, seed=6304):
    """Run the backbone and optionally balance samples across loader groups."""
    model.eval()
    names = list(loaders)
    features, labels, groups, predictions = [], [], [], []

    for index, name in enumerate(names):
        group_features, group_labels, group_predictions = [], [], []
        for batch in loaders[name]:
            images = batch[0].to(DEVICE).float()
            logits, batch_features = model(images, return_features=True)
            group_features.append(batch_features.cpu().numpy())
            group_labels.append(batch[1].numpy())
            group_predictions.append(logits.argmax(dim=1).cpu().numpy())

        group_features = np.concatenate(group_features)
        group_labels = np.concatenate(group_labels)
        group_predictions = np.concatenate(group_predictions)

        if max_per_group is not None and len(group_features) > max_per_group:
            rng = np.random.default_rng(seed + index)
            keep = rng.choice(len(group_features), size=max_per_group, replace=False)
            group_features, group_labels = group_features[keep], group_labels[keep]
            group_predictions = group_predictions[keep]

        features.append(group_features)
        labels.append(group_labels)
        predictions.append(group_predictions)
        groups.append(np.full(len(group_features), index))

    return {
        "features": np.concatenate(features),
        "labels": np.concatenate(labels),
        "groups": np.concatenate(groups),
        "predictions": np.concatenate(predictions),
        "group_names": names,
    }
