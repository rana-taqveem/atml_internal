"""CDAN: class-conditional adversarial alignment.

DANN matches the overall source and target feature distributions. That can
succeed while mixing classes: the domains become indistinguishable, but a
target dog lands where source guitars live. CDAN conditions the discriminator
on the predicted class as well as the feature, through the outer product

    g(x) = vec(f (x) p),   f = 512-d feature,  p = classifier softmax,

which gives a 512 x 7 = 3584-dimensional input. The discriminator therefore
sees "which region of feature space" together with "which class the model
thinks this is", so confusing the domains requires aligning semantically
corresponding regions rather than only the marginal distributions.

Everything else matches DANN exactly: hidden width, activation, dropout,
gradient-reversal schedule and loss weight. Per the assignment, neither f nor
p is detached, and no entropy conditioning is used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from assignment_01.task2.config import task_config
from assignment_01.task2.methods.dann import (
    DomainDiscriminator,
    dann_alpha,
    gradient_reverse,
)


def multilinear_map(features, probabilities):
    """Outer product of feature and class probabilities, flattened per example.

    features       [N, D]        D = 512
    probabilities  [N, C]        C = 7, rows sum to 1
    returns        [N, D*C]      D*C = 3584

    bmm does a per-example matrix product:
        [N, C, 1] x [N, 1, D] -> [N, C, D],  entry (n, c, d) = p[n,c] * f[n,d]
    then view flattens the last two axes into one 3584-long vector.
    """
    outer = torch.bmm(probabilities.unsqueeze(2), features.unsqueeze(1))   # [N, C, D]
    return outer.view(features.size(0), -1)                                # [N, C*D]


def make_cdan_loss(feature_dim=512, num_classes=None, max_alpha=None, weight=None, device=None):
    """Return (loss_fn, discriminator), matching make_dann_loss."""
    num_classes = num_classes or task_config.NUM_CLASSES
    weight = task_config.DOMAIN_LOSS_WEIGHT if weight is None else weight
    max_alpha = task_config.DANN_MAX_ALPHA if max_alpha is None else max_alpha

    discriminator = DomainDiscriminator(
        input_dim=feature_dim * num_classes
    ).to(device or task_config.DEVICE)
    domain_criterion = nn.CrossEntropyLoss()

    def cdan_loss(model, batch, source_features, progress=0.0):
        target_images = batch[3]
        if target_images is None:
            raise ValueError("CDAN needs target images; build the loader with a target_loader.")

        # Source probabilities come from the same features the caller computed.
        source_logits = model.classifier(source_features)        # [B_s, 7]
        target_logits, target_features = model(
            target_images.to(source_features.device).float(), return_features=True
        )                                                        # [B_t, 7], [B_t, 512]

        features = torch.cat([source_features, target_features], dim=0)      # [B_s+B_t, 512]
        probabilities = F.softmax(
            torch.cat([source_logits, target_logits], dim=0), dim=1)         # [B_s+B_t, 7]
        conditioned = multilinear_map(features, probabilities)               # [B_s+B_t, 3584]

        domain_labels = torch.cat([
            torch.zeros(source_features.size(0), dtype=torch.long),   # [B_s] zeros = source
            torch.ones(target_features.size(0), dtype=torch.long),    # [B_t] ones  = target
        ]).to(features.device)                                   # [B_s+B_t]

        alpha = dann_alpha(progress, max_alpha)                  # scalar in [0, max_alpha]
        predictions = discriminator(gradient_reverse(conditioned, alpha))    # [B_s+B_t, 2]

        cdan_loss.last_alpha = alpha
        cdan_loss.last_domain_accuracy = (
            100 * (predictions.argmax(dim=1) == domain_labels).float().mean().item()
        )
        return weight * domain_criterion(predictions, domain_labels)

    cdan_loss.name = "domain"
    cdan_loss.discriminator = discriminator
    cdan_loss.max_alpha = max_alpha
    cdan_loss.last_alpha = 0.0
    cdan_loss.last_domain_accuracy = float("nan")
    return cdan_loss, discriminator
