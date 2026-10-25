"""python -m pytest tests/test_adapters.py -q"""
import torch
import torch.nn as nn

from task1_finetune import adapters as ad


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.layer = nn.Module()
        self.layer.q_proj = nn.Linear(40, 24, bias=False)
        self.layer.down_proj = nn.Linear(24, 40, bias=False)

    def forward(self, x):
        return self.layer.down_proj(self.layer.q_proj(x))


def _make(k):
    m = Toy()
    ad.attach_adapters(m, ["q_proj", "down_proj"], r=4, alpha=8, dropout=0.0, k=k, seed=6304)
    for mod in ad.lora_modules(m).values():  # non-zero B so the correction is non-trivial
        nn.init.normal_(mod.lora_B, std=0.1)
    return m


def test_same_initial_A():
    a, b = Toy(), Toy()
    ad.attach_adapters(a, ["q_proj", "down_proj"], 4, 8, 0.0, None, 6304)
    ad.attach_adapters(b, ["q_proj", "down_proj"], 4, 8, 0.0, 8, 6304)
    for (n, x), (_, y) in zip(ad.lora_modules(a).items(), ad.lora_modules(b).items()):
        assert torch.equal(x.lora_A, y.lora_A), n
        assert torch.equal(x.lora_B, y.lora_B), n


def test_oplora_orthogonality():
    m = _make(k=8)
    for mod in ad.lora_modules(m).values():
        dW = mod.delta_weight()
        assert (mod.U_k.T @ dW).abs().max() < 1e-5
        assert (dW @ mod.V_k).abs().max() < 1e-5
        assert dW.norm() > 1e-3


def test_merge_matches_forward():
    for k in (None, 8):
        m = _make(k).eval()
        x = torch.randn(5, 40)
        y_unmerged = m(x)
        eff = ad.export_effective(m)
        fresh = Toy()
        ad.merge_effective(fresh, eff)
        assert torch.allclose(fresh(x), y_unmerged, atol=1e-5)
