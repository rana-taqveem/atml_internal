"""DANN: adversarial alignment with a domain discriminator.

Instead of measuring the source-target gap with a fixed formula, DANN learns
a binary classifier that tries to tell source features from target features,
and trains the backbone to defeat it. If a well-trained discriminator cannot
separate the domains, the representation carries little domain information.

The two opposing objectives share one backward pass through a gradient
reversal layer: it is the identity going forward, and multiplies the gradient
by -alpha going backward. So the discriminator descends the domain loss while
the backbone ascends it.

alpha follows the standard schedule

    alpha(p) = 2 / (1 + exp(-10p)) - 1,   p in [0, 1] training progress,

which starts near 0 and approaches 1. Early on the representation is free to
learn the class task; later the reversed gradient increasingly pushes for
domain confusion.

Only source examples contribute to the classification loss; both source and
target contribute to the domain loss. Target labels are never used.
"""

import torch
import torch.nn as nn

from assignment_01.task2.config import task_config


class GradientReversal(torch.autograd.Function):
    """Identity forward; negated, scaled gradient backward.

    Shape is unchanged in both directions: whatever goes in comes out, and the
    gradient keeps the same shape. Only the sign and scale of the gradient change.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)                      # [N, D] -> [N, D], values untouched

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output [N, D] -> [N, D] scaled by -alpha; None for the alpha argument.
        return -ctx.alpha * grad_output, None


def gradient_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


def dann_alpha(progress, max_alpha=None):
    """alpha(p) = 2 / (1 + exp(-10p)) - 1, optionally capped for the study."""
    max_alpha = task_config.DANN_MAX_ALPHA if max_alpha is None else max_alpha
    progress = min(max(float(progress), 0.0), 1.0)
    schedule = 2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * progress))).item() - 1.0
    return max_alpha * schedule


class DomainDiscriminator(nn.Module):
    """256-unit hidden layer, ReLU, dropout 0.5, two-class output."""

    def __init__(self, input_dim=None, hidden=None, dropout=None):
        """input_dim is 512 for DANN (the raw feature) and 512*7 = 3584 for CDAN."""
        super().__init__()
        input_dim = input_dim or 512
        hidden = hidden or task_config.DISCRIMINATOR_HIDDEN
        dropout = task_config.DISCRIMINATOR_DROPOUT if dropout is None else dropout

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        """x [N, input_dim] -> [N, 2] domain logits (column 0 source, 1 target)."""
        return self.net(x)


def make_dann_loss(feature_dim=512, max_alpha=None, weight=None, device=None):
    """Return (loss_fn, discriminator).

    The discriminator has its own parameters, so the caller must add them to
    the optimizer; they are trained normally while the reversed gradient
    reaches the backbone.
    """
    weight = task_config.DOMAIN_LOSS_WEIGHT if weight is None else weight
    max_alpha = task_config.DANN_MAX_ALPHA if max_alpha is None else max_alpha
    discriminator = DomainDiscriminator(input_dim=feature_dim).to(device or task_config.DEVICE)
    domain_criterion = nn.CrossEntropyLoss()

    def dann_loss(model, batch, source_features, progress=0.0):
        # batch = (images [B_s,3,224,224], labels [B_s], domain_ids [B_s],
        #          target_images [B_t,3,224,224]); source_features is [B_s, 512].
        target_images = batch[3]
        if target_images is None:
            raise ValueError("DANN needs target images; build the loader with a target_loader.")

        target_features = model.forward_features(
            target_images.to(source_features.device).float()
        )                                               # [B_t, 512]

        features = torch.cat([source_features, target_features], dim=0)   # [B_s+B_t, 512]
        domain_labels = torch.cat([
            torch.zeros(source_features.size(0), dtype=torch.long),   # [B_s] zeros = source
            torch.ones(target_features.size(0), dtype=torch.long),    # [B_t] ones  = target
        ]).to(features.device)                          # [B_s+B_t]

        alpha = dann_alpha(progress, max_alpha)         # scalar in [0, max_alpha]
        predictions = discriminator(gradient_reverse(features, alpha))   # [B_s+B_t, 2]

        dann_loss.last_alpha = alpha
        dann_loss.last_domain_accuracy = (
            100 * (predictions.argmax(dim=1) == domain_labels).float().mean().item()
        )
        # CrossEntropyLoss: ([B_s+B_t, 2], [B_s+B_t]) -> scalar
        return weight * domain_criterion(predictions, domain_labels)

    dann_loss.name = "domain"
    dann_loss.discriminator = discriminator
    dann_loss.max_alpha = max_alpha
    dann_loss.last_alpha = 0.0
    dann_loss.last_domain_accuracy = float("nan")
    return dann_loss, discriminator
