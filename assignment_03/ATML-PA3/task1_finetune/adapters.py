"""LoRA and OPLoRA linear adapters (own implementation, so LoRA-16 and OPLoRA-16 share
identical initial unprojected factors under the same seed).

LoRA:    W = W0 + (alpha/r) B A,         A in R^{r x d_in}, B in R^{d_out x r}
OPLoRA:  B~ = B - U_k (U_k^T B),  A~ = A - (A V_k) V_k^T,  dW = (alpha/r) B~ A~
with U_k, V_k the leading k singular vectors of the starting weight W0 = U S V^T (FP32).
Effective factors are formed on the fly; dense projection matrices are never built.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float,
                 k: int | None = None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        dev = base.weight.device
        self.r, self.alpha, self.scale = r, alpha, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # PEFT default: random A, zero B
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.k = k
        if k:
            self.register_buffer("U_k", torch.empty(base.out_features, k, device=dev), persistent=False)
            self.register_buffer("V_k", torch.empty(base.in_features, k, device=dev), persistent=False)

    @torch.no_grad()
    def set_bases(self, U_k: torch.Tensor, V_k: torch.Tensor):
        self.U_k.copy_(U_k)
        self.V_k.copy_(V_k)

    def effective_factors(self):
        A, B = self.lora_A, self.lora_B
        if not self.k:
            return A, B
        with torch.autocast(device_type=A.device.type, enabled=False):
            A_eff = A - (A @ self.V_k) @ self.V_k.T
            B_eff = B - self.U_k @ (self.U_k.T @ B)
        return A_eff, B_eff

    def delta_weight(self) -> torch.Tensor:
        """Dense FP32 correction dW = (alpha/r) B~ A~ (diagnostics/merging only)."""
        A, B = self.effective_factors()
        with torch.autocast(device_type=A.device.type, enabled=False):
            return self.scale * (B.float() @ A.float())

    def forward(self, x):
        out = self.base(x)
        A, B = self.effective_factors()
        h = self.dropout(x)
        if not torch.is_autocast_enabled(x.device.type):
            h = h.to(A.dtype)
        lora = F.linear(F.linear(h, A), B) * self.scale
        return out + lora.to(out.dtype)


def iter_targets(model, targets):
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in targets:
            yield name, mod


def attach_adapters(model, targets, r, alpha, dropout, k=None, seed=6304) -> dict:
    """Replace target Linear layers with LoRALinear; returns overhead info.

    The RNG is reseeded here so LoRA-16 and OPLoRA-16 draw identical A factors.
    """
    torch.manual_seed(seed)
    info = {"basis_seconds": 0.0, "basis_bytes": 0, "n_modules": 0}
    names = [n for n, _ in iter_targets(model, targets)]
    for name in names:
        parent_name, child = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        base = getattr(parent, child)
        wrapped = LoRALinear(base, r, alpha, dropout, k)
        if k:
            t0 = time.perf_counter()
            W0 = base.weight.detach().float()
            U, S, Vh = torch.linalg.svd(W0, full_matrices=False)
            wrapped.set_bases(U[:, :k].contiguous(), Vh[:k].T.contiguous())
            if W0.is_cuda:
                torch.cuda.synchronize()
            info["basis_seconds"] += time.perf_counter() - t0
            info["basis_bytes"] += (wrapped.U_k.numel() + wrapped.V_k.numel()) * 4
        setattr(parent, child, wrapped)
        info["n_modules"] += 1
    return info


def lora_modules(model):
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def adapter_state(model) -> dict:
    """Raw trainable factors (for resuming)."""
    return {f"{n}.lora_A": m.lora_A.detach().cpu() for n, m in lora_modules(model).items()} | \
           {f"{n}.lora_B": m.lora_B.detach().cpu() for n, m in lora_modules(model).items()}


def load_adapter_state(model, sd: dict):
    for n, m in lora_modules(model).items():
        m.lora_A.data.copy_(sd[f"{n}.lora_A"])
        m.lora_B.data.copy_(sd[f"{n}.lora_B"])


def export_effective(model) -> dict:
    """Effective (projected for OPLoRA) FP32 factors for inference/merging.

    Keys use the base module name, so they can be merged into a fresh backbone.
    """
    out = {}
    with torch.no_grad():
        for n, m in lora_modules(model).items():
            A, B = m.effective_factors()
            out[f"{n}.A"] = A.detach().float().cpu().contiguous()
            out[f"{n}.B"] = B.detach().float().cpu().contiguous()
            out[f"{n}.scale"] = torch.tensor(m.scale)
    return out


@torch.no_grad()
def merge_effective(model, eff: dict) -> int:
    """W <- W0 + scale * B A on a fresh backbone (computed in FP32, cast to the model dtype)."""
    names = sorted({k.rsplit(".", 1)[0] for k in eff})
    for n in names:
        lin = model.get_submodule(n)
        W = lin.weight
        dW = eff[f"{n}.scale"].item() * (eff[f"{n}.B"].to(W.device) @ eff[f"{n}.A"].to(W.device))
        W.copy_((W.float() + dW).to(W.dtype))
    return len(names)


def trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
