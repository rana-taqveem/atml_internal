"""Local sharpness proxy, shared across models.

    delta_sharp = L_val(theta + eps) - L_val(theta),
    eps = radius * grad L_val(theta) / ||grad L_val(theta)||

Take one normalized step uphill and measure how much the loss rises. A larger
increase means the solution sits in a sharper region under this specific
perturbation.

The assignment is careful about what this shows, and the report should be too:
it is a standardized local diagnostic under one perturbation of one radius on
one fixed batch. It does not establish that a model's whole loss landscape is
flatter, and it does not prove that domain shift was solved.

Every model must be measured on the same fixed batch, drawn with seed 6304,
with the model in evaluation mode.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from assignment_01.task2.config import task_config
from assignment_01.task2.scripts.main import DEVICE


def fixed_validation_batch(validation_loaders, per_domain=32, seed=None, num_workers=0):
    """One fixed batch: `per_domain` examples from each source validation split.

    The same indices every time, so the proxy is comparable across models.
    """
    seed = task_config.SEED if seed is None else seed
    images, labels = [], []

    for index, (domain, loader) in enumerate(validation_loaders.items()):
        dataset = loader.dataset
        rng = np.random.default_rng(seed + index)
        count = min(per_domain, len(dataset))
        chosen = sorted(rng.choice(len(dataset), size=count, replace=False).tolist())

        for batch in DataLoader(Subset(dataset, chosen), batch_size=count, shuffle=False,
                                num_workers=num_workers):
            images.append(batch[0])
            labels.append(batch[1])

    return torch.cat(images), torch.cat(labels)


def sharpness_proxy(model, images, labels, radius=None, criterion=None):
    """Increase in loss after one normalized gradient-ascent step.

    The model is restored to its original parameters before returning.
    """
    radius = task_config.SHARPNESS_RADIUS if radius is None and hasattr(
        task_config, "SHARPNESS_RADIUS") else (0.05 if radius is None else radius)
    criterion = criterion or nn.CrossEntropyLoss()

    model.eval()                       # evaluation mode, as the assignment specifies
    images, labels = images.to(DEVICE).float(), labels.to(DEVICE)

    model.zero_grad(set_to_none=True)
    base_loss = criterion(model(images), labels)
    base_loss.backward()

    parameters = [p for p in model.parameters() if p.grad is not None]
    grad_norm = torch.norm(torch.stack([p.grad.norm(p=2) for p in parameters]), p=2)

    # Snapshot and copy back rather than add/subtract: add_ then sub_ is not
    # exactly reversible in floating point, and this diagnostic must leave
    # every model bit-identical so repeated measurements stay comparable.
    original = [p.detach().clone() for p in parameters]

    with torch.no_grad():
        for p in parameters:
            p.add_(radius * p.grad / (grad_norm + 1e-12))

        perturbed_loss = criterion(model(images), labels)

        for p, saved in zip(parameters, original):
            p.copy_(saved)

    model.zero_grad(set_to_none=True)
    return {
        "loss": float(base_loss.detach()),
        "perturbed_loss": float(perturbed_loss),
        "delta_sharp": float(perturbed_loss - base_loss.detach()),
        "radius": radius,
        "n_examples": int(images.size(0)),
        "gradient_norm": float(grad_norm),
    }
