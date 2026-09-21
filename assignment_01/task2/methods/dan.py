"""DAN: Maximum Mean Discrepancy alignment between source and target features.

The objective is the ordinary classification loss plus a discrepancy penalty
on the 512-dimensional feature immediately before the classifier:

    L_DAN = L_cls + lambda_MMD * MMD^2(source features, target features)

MMD compares two sets of vectors through average pairwise similarity:

    MMD^2 = mean(k(s, s')) + mean(k(t, t')) - 2 * mean(k(s, t))

If the two clouds occupy the same region the first two terms are cancelled by
the third and the value is near zero; if they are separated, source-to-target
similarity is small and the value grows. The kernel trick means we never build
the feature map phi explicitly, only evaluate similarities.

The kernel is a sum of three RBF kernels whose bandwidths are 0.5, 1 and 2
times the median pairwise squared distance in the current combined batch, so
the scale adapts as the representation changes during training.

Target labels are never used: the penalty only knows which features came from
which domain.
"""

import torch

from assignment_01.task2.config import task_config


def _squared_distances(a, b):
    """Pairwise squared Euclidean distances, shape [len(a), len(b)]."""
    return torch.cdist(a, b, p=2).pow(2)


def median_bandwidth(features):
    """Median pairwise squared distance over the combined batch.

    Self-distances are excluded: they are all zero and would drag the median
    down. The value is detached because it is a scale statistic, not a
    quantity we want gradients to flow through.
    """
    distances = _squared_distances(features, features)
    n = distances.size(0)
    off_diagonal = distances[~torch.eye(n, dtype=torch.bool, device=distances.device)]
    median = off_diagonal.median().detach()
    return torch.clamp(median, min=1e-8)


def mmd_rbf(source_features, target_features, multipliers=None):
    """Squared MMD with a sum of RBF kernels at several bandwidths.

    Returns a scalar tensor; zero means the two batches are indistinguishable
    under this kernel, larger means more discrepancy.
    """
    multipliers = multipliers or task_config.MMD_BANDWIDTH_MULTIPLIERS

    source_features = source_features.float()
    target_features = target_features.float()
    combined = torch.cat([source_features, target_features], dim=0)
    base = median_bandwidth(combined)

    source_source = _squared_distances(source_features, source_features)
    target_target = _squared_distances(target_features, target_features)
    source_target = _squared_distances(source_features, target_features)

    total = source_features.new_zeros(())
    for multiplier in multipliers:
        bandwidth = base * multiplier
        total = total + (
            torch.exp(-source_source / bandwidth).mean()
            + torch.exp(-target_target / bandwidth).mean()
            - 2 * torch.exp(-source_target / bandwidth).mean()
        )
    return total


def make_dan_loss(lambda_mmd=None, multipliers=None):
    """Build the extra_loss_fn the training loop calls each step.

    The training loop already computed the source features for the
    classification loss, so only the target features need a forward pass.
    """
    lambda_mmd = task_config.DAN_LAMBDA_MMD if lambda_mmd is None else lambda_mmd

    def dan_loss(model, batch, source_features, progress=None):
        
        target_images = batch[3]
        
        if target_images is None:
            raise ValueError("DAN needs target images; build the loader with a target_loader.")

        target_features = model.forward_features(target_images.to(source_features.device).float())
        return lambda_mmd * mmd_rbf(source_features, target_features, multipliers)

    dan_loss.name = "mmd"
    dan_loss.lambda_value = lambda_mmd
    return dan_loss
