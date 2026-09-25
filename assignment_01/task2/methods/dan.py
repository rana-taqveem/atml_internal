"""DAN source-target alignment using multi-kernel MMD."""

import torch

from assignment_01.task2.config import task_config


def _squared_distances(a, b):
    """Pairwise squared Euclidean distances.

    a        [N, D]
    b        [M, D]
    returns  [N, M]   entry (i, j) is ||a_i - b_j||^2
    """
    return torch.cdist(a, b, p=2).pow(2)


def median_bandwidth(features):
    """Return the differentiable median off-diagonal squared distance.

    Keeping the bandwidth in the autograd graph prevents feature contraction
    from trivially minimizing the discrepancy.
    """
    distances = _squared_distances(features, features)   # [N, N], N = B_s + B_t
    n = distances.size(0)
    # Drop the N zero self-distances: [N, N] -> [N*(N-1)] off-diagonal entries.
    off_diagonal = distances[~torch.eye(n, dtype=torch.bool, device=distances.device)]
    median = off_diagonal.median()                      # scalar, differentiable
    return torch.clamp(median, min=1e-8)


def mmd_rbf(source_features, target_features, multipliers=None):
    """Squared MMD with a sum of RBF kernels at several bandwidths.

    Returns a scalar tensor; zero means the two batches are indistinguishable
    under this kernel, larger means more discrepancy.
    """
    multipliers = multipliers or task_config.MMD_BANDWIDTH_MULTIPLIERS

    source_features = source_features.float()           # [B_s, D]  B_s = 24, D = 512
    target_features = target_features.float()           # [B_t, D]  B_t = 24
    combined = torch.cat([source_features, target_features], dim=0)   # [B_s+B_t, D]
    base = median_bandwidth(combined)                   # scalar

    source_source = _squared_distances(source_features, source_features)   # [B_s, B_s]
    target_target = _squared_distances(target_features, target_features)   # [B_t, B_t]
    source_target = _squared_distances(source_features, target_features)   # [B_s, B_t]

    total = source_features.new_zeros(())               # scalar accumulator
    for multiplier in multipliers:
        bandwidth = base * multiplier                   # scalar
        # Each exp(...) is elementwise on its distance matrix; .mean() reduces
        # it to a scalar, so `total` stays a scalar throughout.
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
        
        # batch = (images [B_s,3,224,224], labels [B_s], domain_ids [B_s],
        #          target_images [B_t,3,224,224]); source_features is [B_s, 512].
        target_images = batch[3]

        if target_images is None:
            raise ValueError("DAN needs target images; build the loader with a target_loader.")

        target_features = model.forward_features(
            target_images.to(source_features.device).float()
        )                                               # [B_t, 512]
        return lambda_mmd * mmd_rbf(source_features, target_features, multipliers)  # scalar

    dan_loss.name = "mmd"
    dan_loss.lambda_value = lambda_mmd
    return dan_loss
