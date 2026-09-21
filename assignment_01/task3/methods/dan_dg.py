"""DAN-DG: pairwise alignment of the three observed source domains.

    L_DAN-DG = L_ERM + (lambda_DG / 3) * sum_{e < e'} MMD^2(F(X_e), F(X_e'))

Same discrepancy mechanism as Task 2's DAN, different information. Task 2's
DAN aligns the labelled sources against the unlabelled Sketch target; DAN-DG
never sees Sketch at all and instead aligns Photo, Art Painting and Cartoon
with each other. Comparing the two is how the report measures what unlabelled
target data was worth.

The MMD implementation and kernel construction are imported unchanged from
Task 2, so the only difference between the tasks is which features are
compared. With three domains there are three unordered pairs
(photo-art, photo-cartoon, art-cartoon), hence the division by 3.
"""

import torch

from assignment_01.task2.methods.dan import mmd_rbf
from assignment_01.task3.config import task_config


def pairwise_source_mmd(features, domain_ids, multipliers=None):
    """Mean MMD^2 over the unordered pairs of source domains present in the batch.

    features    [B_s, 512]   B_s = 24 (8 per domain)
    domain_ids  [B_s]        0, 1, 2 in SOURCE_DOMAINS order
    returns     scalar
    """
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
    """Build the extra_loss_fn the shared Task 2 training loop calls each step.

    Uses only the source features the loop already computed: no target data,
    and no extra forward pass.
    """
    lambda_dg = task_config.LAMBDA_DG if lambda_dg is None else lambda_dg

    def dan_dg_loss(model, batch, source_features, progress=None):
        # batch = (images [24,3,224,224], labels [24], domain_ids [24], target_images)
        # target_images is None here by construction: Task 3 never loads Sketch.
        domain_ids = batch[2].to(source_features.device)
        return lambda_dg * pairwise_source_mmd(source_features, domain_ids, multipliers)

    dan_dg_loss.name = "mmd_dg"
    dan_dg_loss.lambda_value = lambda_dg
    return dan_dg_loss
