"""Task 1 weight-space and activation diagnostics on down_proj, layers 0/13/27 (FP32).

Weight space, per condition (Full FT midpoint + five finals):
    rho_k = ||U_k U_k^T dW||_F^2 / ||dW||_F^2,   eta = ||dW||_F / ||W0||_F
with U_k (k=128) from the starting weight W0 = U S V^T. Full FT: dW = W_trained - W0;
adapters: dW = (alpha/r) B~ A~ from the exported effective (projected) factors.
Also records the OPLoRA constraint residuals ||U_k^T dW||/||dW|| and ||dW V_k||/||dW||.

Activations: down_proj inputs z of the *starting* model at the last prompt token for the
fixed probe set (32 GSM8K test + 32 HellaSwag val) in benchmark prompt formats, then
    R(z) = ||dW z||_2 / (||W0 z||_2 + 1e-8)
for every condition, reusing the same cached z.

    python -m task1_finetune.diagnostics
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from common.store import Store
from common.utils import ensure_dir, load_config, read_json, rel, write_json
from task1_finetune.data import build_probe_ids, format_prompt, get_tokenizer
from task1_finetune.evaluate import weights_dir

OUT = "results/task1/diagnostics"


def key(layer, module):
    return f"model.layers.{layer}.mlp.{module}"


def load_tensors(model_dir_or_id, names):
    """Read selected weights from (possibly sharded) safetensors without loading the model."""
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    d = Path(model_dir_or_id)
    if not d.exists():
        d = Path(snapshot_download(str(model_dir_or_id), allow_patterns=["*.safetensors", "*.json"]))
    out = {}
    for f in sorted(d.glob("*.safetensors")):
        with safe_open(str(f), "pt") as sf:
            for n in names:
                if n in sf.keys():
                    out[n] = sf.get_tensor(n).float()
    missing = set(names) - set(out)
    assert not missing, f"missing {missing} in {d}"
    return out


def deltas(cfg, cond, W0, store):
    c = cfg["eval"]["conditions"][cond]
    v, stage = c["variant"], c["stage"]
    wdir = weights_dir(cfg, v, stage)
    if not rel(wdir).exists():
        assert store.pull(wdir), f"missing {wdir}"
    mod = cfg["diagnostics"]["module"]
    names = [f"{key(l, mod)}.weight" for l in cfg["diagnostics"]["layers"]]
    if cfg["variants"][v]["kind"] == "full":
        Wt = load_tensors(rel(wdir), names)
        return {n: Wt[n] - W0[n] for n in names}
    from safetensors.torch import load_file

    eff = load_file(str(rel(wdir) / "adapter_effective.safetensors"))
    out = {}
    for n in names:
        base = n[: -len(".weight")]
        out[n] = eff[f"{base}.scale"].item() * (eff[f"{base}.B"].float() @ eff[f"{base}.A"].float())
    return out


def probe_prompts(cfg, tok):
    """Probe prompts in the benchmark formats used by lm-eval (gsm8k_cot zero-shot with the
    non-thinking chat template; HellaSwag stock completion context)."""
    from datasets import load_dataset
    from lm_eval.tasks.hellaswag.utils import process_docs

    ids = build_probe_ids(cfg)
    gsm = load_dataset(cfg["data"]["gsm8k_id"], "main", split="test")
    hs = process_docs(load_dataset("Rowan/hellaswag", split="validation"))
    rows = [{"set": "math", "source": "gsm8k_test", "doc_id": i,
             "prompt": format_prompt(tok, f"Q: {gsm[i]['question']}\nA:")} for i in ids["gsm8k_test"]]
    rows += [{"set": "nonmath", "source": "hellaswag_validation", "doc_id": i,
              "prompt": hs[i]["query"]} for i in ids["hellaswag_validation"]]
    return rows


@torch.no_grad()
def cache_inputs(cfg, rows, tok, device):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], dtype=torch.float32).to(device).eval()
    layers, mod = cfg["diagnostics"]["layers"], cfg["diagnostics"]["module"]
    cache, hooks = {}, []
    for l in layers:
        m = model.get_submodule(key(l, mod))
        hooks.append(m.register_forward_hook(
            lambda _m, inp, _o, l=l: cache.__setitem__(l, inp[0][0, -1].detach().float().cpu())))
    Z = {l: [] for l in layers}
    for r in rows:
        ids = tok(r["prompt"], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        model(input_ids=ids)  # unpadded, so position -1 is the last prompt token
        for l in layers:
            Z[l].append(cache[l])
    for h in hooks:
        h.remove()
    return {l: torch.stack(v) for l, v in Z.items()}


def run(cfg, store=None):
    store = store or Store()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = ensure_dir(OUT)
    dcfg = cfg["diagnostics"]
    names = [f"{key(l, dcfg['module'])}.weight" for l in dcfg["layers"]]
    W0 = load_tensors(cfg["model_id"], names)
    bases = {}
    for n in names:
        U, S, Vh = torch.linalg.svd(W0[n].to(device), full_matrices=False)
        bases[n] = (U[:, :dcfg["k"]].cpu(), Vh[:dcfg["k"]].T.cpu())

    tok = get_tokenizer(cfg["model_id"])
    rows = probe_prompts(cfg, tok)
    Z = cache_inputs(cfg, rows, tok, device)

    conds = [c for c, v in cfg["eval"]["conditions"].items() if v["variant"]]
    w_rows, a_rows = [], []
    for cond in conds:
        D = deltas(cfg, cond, W0, store)
        for n, l in zip(names, dcfg["layers"]):
            dW, Uk, Vk = D[n], *bases[n]
            fro2 = float(dW.pow(2).sum())
            w_rows.append({
                "condition": cond, "layer": l,
                "rho_k": float((Uk.T @ dW).pow(2).sum()) / fro2 if fro2 > 0 else float("nan"),
                "eta": fro2 ** 0.5 / float(W0[n].norm()),
                "dW_fro": fro2 ** 0.5,
                "left_residual": float((Uk.T @ dW).norm()) / fro2 ** 0.5 if fro2 > 0 else float("nan"),
                "right_residual": float((dW @ Vk).norm()) / fro2 ** 0.5 if fro2 > 0 else float("nan"),
            })
            num = (Z[l] @ dW.T).norm(dim=1)
            den = (Z[l] @ W0[n].T).norm(dim=1)
            R = (num / (den + 1e-8)).numpy()
            for r, val in zip(rows, R):
                a_rows.append({"condition": cond, "layer": l, "set": r["set"],
                               "doc_id": r["doc_id"], "R": float(val)})

    wdf = pd.DataFrame(w_rows)
    adf = pd.DataFrame(a_rows)
    wdf.to_csv(out / "weight_space.csv", index=False)
    adf.to_csv(out / "activation_per_prompt.csv", index=False)
    summ = (adf.groupby(["condition", "layer", "set"])["R"]
            .agg(["mean", "median", "std"]).reset_index())
    piv = adf.groupby(["condition", "layer", "set"])["R"].mean().unstack("set")
    piv["math_over_nonmath"] = piv["math"] / piv["nonmath"]
    summ.to_csv(out / "activation_summary.csv", index=False)
    piv.reset_index().to_csv(out / "activation_ratio.csv", index=False)
    write_json([{k: r[k] for k in ("set", "source", "doc_id", "prompt")} for r in rows],
               out / "probe_prompts.json")
    write_json({"k": dcfg["k"], "layers": dcfg["layers"], "module": dcfg["module"],
                "z_source": "starting model, FP32, last prompt token, unpadded",
                "delta_full": "W_trained(FP32 master) - W0", "delta_adapter": "(alpha/r) B~ A~"},
               out / "meta.json")
    store.push(OUT, "task1 diagnostics")
    print(wdf.round(4).to_string())
    print(piv.round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml")
    run(load_config(ap.parse_args().config))


if __name__ == "__main__":
    main()
