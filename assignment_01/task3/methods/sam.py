"""Standard non-adaptive Sharpness-Aware Minimization."""

import torch

from assignment_01.task2.scripts.main import DEVICE, freeze_batchnorm_statistics
from assignment_01.task3.config import task_config


class SAM(torch.optim.Optimizer):
    """Wraps a base optimizer (AdamW here) with the two-step SAM update."""

    def __init__(self, params, base_optimizer_cls, rho=None, **base_kwargs):
        rho = task_config.SAM_RHO if rho is None else rho
        if rho <= 0:
            raise ValueError("rho must be positive.")

        defaults = dict(rho=rho, **base_kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.rho = rho

    def _grad_norm(self):
        """Single norm over every parameter with a gradient (non-adaptive SAM)."""
        device = self.param_groups[0]["params"][0].device
        return torch.norm(torch.stack([
            p.grad.norm(p=2).to(device)
            for group in self.param_groups for p in group["params"]
            if p.grad is not None
        ]), p=2)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Move to theta + eps, the worst point within radius rho."""
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                perturbation = p.grad * scale.to(p)
                p.add_(perturbation)                       # theta -> theta + eps
                self.state[p]["perturbation"] = perturbation

        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Restore theta, then apply the gradient measured at theta + eps."""
        for group in self.param_groups:
            for p in group["params"]:
                perturbation = self.state[p].pop("perturbation", None)
                if perturbation is not None:
                    p.sub_(perturbation)                   # theta + eps -> theta

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self, closure=None):
        raise RuntimeError("SAM requires first_step()/second_step(); use train_one_epoch_sam.")


def train_one_epoch_sam(model, loader, criterion, optimizer, epoch=0, num_epochs=1,
                        grad_clip=None):
    """Run one two-pass SAM training epoch without mixed precision."""
    model.train()
    freeze_batchnorm_statistics(model)

    totals = {"loss": 0.0, "classification_loss": 0.0, "alignment_loss": 0.0}
    accurate_predictions = 0
    sample_count = 0

    for batch in loader:
        images, labels = batch[0].to(DEVICE).float(), batch[1].to(DEVICE)

        # Pass 1: gradient at theta, then step to the worst nearby point.
        outputs = model(images)                            # [24, 7]
        loss = criterion(outputs, labels)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]], grad_clip)
        optimizer.first_step(zero_grad=True)

        # Pass 2: gradient at theta + eps, restore theta, apply that gradient.
        perturbed_loss = criterion(model(images), labels)
        perturbed_loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]], grad_clip)
        optimizer.second_step(zero_grad=True)

        _, predicted_classes = torch.max(outputs.data, 1)
        accurate_predictions += (predicted_classes == labels).sum().item()
        batch_size = images.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["classification_loss"] += loss.item() * batch_size
        sample_count += batch_size

    result = {key: value / sample_count for key, value in totals.items()}
    result["accuracy"] = 100 * accurate_predictions / sample_count
    result["domain_accuracy"] = None
    result["nonfinite_steps"] = 0
    return result
