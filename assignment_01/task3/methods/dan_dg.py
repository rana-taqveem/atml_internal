"""DAN-DG pairwise MMD alignment over the observed source domains."""

import torch

from assignment_01.task2.methods.dan import mmd_rbf
from assignment_01.task3.config import task_config


def pairwise_source_mmd(features, domain_ids, multipliers=None):
    """Return mean MMD² over source-domain pairs present in the batch."""
    multipliers = multipliers or task_config.MMD_BANDWIDTH_MULTIPLIERS
    domains = torch.unique(domain_ids).tolist()

    total = features.new_zeros(())
    pairs = 0
    for index, first in enumerate(domains):
        for second in domains[index + 1:]:
            left = features[domain_ids == first]      # [8, 512]
            right = features[domain_ids == second]    # [8, 512]
            if left.size(0) < 2 or right.size(0) < 2:
                continue                              # MMD needs at least two points
            total = total + mmd_rbf(left, right, multipliers)
            pairs += 1

    return total / pairs if pairs else total


def make_dan_dg_loss(lambda_dg=None, multipliers=None):
    """Build the source-only MMD loss used by the shared training loop."""
    lambda_dg = task_config.LAMBDA_DG if lambda_dg is None else lambda_dg

    def dan_dg_loss(model, batch, source_features, progress=None):
        domain_ids = batch[2].to(source_features.device)
        return lambda_dg * pairwise_source_mmd(source_features, domain_ids, multipliers)

    dan_dg_loss.name = "mmd_dg"
    dan_dg_loss.lambda_value = lambda_dg
    return dan_dg_loss
