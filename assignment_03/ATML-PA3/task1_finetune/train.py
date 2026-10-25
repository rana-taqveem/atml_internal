"""Task 1 training for all five variants (full, l2sp, lora4, lora16, oplora16).

Shared recipe: one epoch over 40,000 fixed-order examples, microbatch 2 x accumulation 16,
1,250 AdamW updates, cosine schedule with 38 warmup updates, clip 1.0, FP16 autocast with
dynamic loss scaling, gradient checkpointing. Full-FT variants keep FP32 master weights;
adapter variants use a frozen FP16 backbone with FP32 adapters. Loss = summed response-token
cross-entropy / number of response tokens in the whole effective batch.

    python -m task1_finetune.train --variant full [--max-hours 11] [--smoke-updates 20]

Resumable: recovery checkpoints at updates 312/625/937/1250 (and on the time guard) hold
weights/adapters, optimizer, scheduler, scaler, RNG and data position (= update index).
"""
from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from common.store import Store
from common.utils import (append_jsonl, dir_size_bytes, ensure_dir, env_info, gpu_memory,
                          load_config, read_jsonl, rel, seed_everything, write_json)
from task1_finetune import adapters as ad
from task1_finetune.data import TrainData, get_tokenizer

CE_CHUNK = 512


def out_dir(variant):
    return f"outputs/task1/{variant}"


def res_dir(variant):
    return f"results/task1/train/{variant}"


def load_model(model_id, dtype, device):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa")
    m.config.use_cache = False
    return m.to(device)


def build(cfg, variant, device):
    vcfg = cfg["variants"][variant]
    info = {"variant": variant, **vcfg}
    if vcfg["kind"] == "full":
        model = load_model(cfg["model_id"], torch.float32, device)  # FP32 master weights
    else:
        # frozen FP16 backbone (FP32 only for CPU tests, where FP16 GEMMs are impractically slow)
        model = load_model(cfg["model_id"], torch.float16 if device == "cuda" else torch.float32, device)
        for p in model.parameters():
            p.requires_grad_(False)
        info |= ad.attach_adapters(model, cfg["lora"]["targets"], vcfg["r"], vcfg["alpha"],
                                   cfg["lora"]["dropout"], vcfg.get("k"), seed=cfg["seed"])
    if cfg["train"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    info["trainable_params"] = ad.trainable_params(model)
    info["total_params"] = sum(p.numel() for p in model.parameters())
    return model, info


def param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or n.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def cosine_lambda(warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return f


def sft_loss_sum(model, input_ids, attn, labels):
    """Summed next-token CE over supervised positions; logits only for those positions,
    in checkpointed chunks so the 152k-vocab logits are never all stored."""
    h = model.model(input_ids=input_ids, attention_mask=attn).last_hidden_state
    tgt = labels[:, 1:]
    mask = tgt != -100
    h_sel = h[:, :-1][mask]
    y = tgt[mask]
    head = model.lm_head

    def chunk_ce(hc, yc):
        return F.cross_entropy(head(hc).float(), yc, reduction="sum")

    total = h_sel.new_zeros((), dtype=torch.float32)
    for s in range(0, h_sel.shape[0], CE_CHUNK):
        total = total + checkpoint(chunk_ce, h_sel[s:s + CE_CHUNK], y[s:s + CE_CHUNK],
                                   use_reentrant=False)
    return total, int(y.numel())


class L2SP:
    """Omega = 0.5 * sum_j (theta_j - theta0_j)^2 over unique parameters (tied counted once).

    theta0 is the released checkpoint, stored in bf16 - the checkpoint's own dtype - so the
    reference is exact. The penalty gradient lambda*(theta - theta0) is added to the
    unscaled SFT gradient once per optimizer update (equivalent to adding lambda*Omega to
    the loss of the effective batch)."""

    def __init__(self, model, lam):
        self.lam = lam
        t0 = time.perf_counter()
        self.params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.ref = [p.detach().to(torch.bfloat16).clone() for _, p in self.params]
        if self.ref and self.ref[0].is_cuda:
            torch.cuda.synchronize()
        self.setup_seconds = time.perf_counter() - t0
        self.ref_bytes = sum(r.numel() * r.element_size() for r in self.ref)
        self.penalty_seconds = 0.0

    @torch.no_grad()
    def omega_and_grad(self, add_grad: bool):
        t0 = time.perf_counter()
        omega = 0.0
        for (_, p), r in zip(self.params, self.ref):
            d = p.detach() - r.float()
            omega += float(d.pow(2).sum()) * 0.5
            if add_grad and p.grad is not None:
                p.grad.add_(d, alpha=self.lam)
        self.penalty_seconds += time.perf_counter() - t0
        return omega


@torch.no_grad()
def displacement(model, model_id) -> float:
    """||theta - theta0||_2 over all unique parameters (full-FT variants)."""
    from transformers import AutoModelForCausalLM

    ref = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    ref_p = dict(ref.named_parameters())
    sq = 0.0
    for n, p in model.named_parameters():
        sq += float((p.detach().float().cpu() - ref_p[n].detach()).pow(2).sum())
    del ref
    return math.sqrt(sq)


def rng_state():
    st = {"python": random.getstate(), "numpy": np.random.get_state(),
          "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def save_recovery(variant, kind, update, model, opt, sched, scaler, extra, store):
    d = ensure_dir(f"{out_dir(variant)}/recovery_u{update:04d}")
    t0 = time.perf_counter()
    state = model.state_dict() if kind == "full" else ad.adapter_state(model)
    torch.save(state, d / "weights.pt")
    torch.save(opt.state_dict(), d / "optimizer.pt")
    torch.save({"scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                "rng": rng_state(), "update": update, **extra}, d / "trainer_state.pt")
    size = dir_size_bytes(d)
    store.push(d, f"{variant} recovery u{update}")
    store.push(res_dir(variant), f"{variant} logs u{update}")
    # keep only the newest recovery checkpoint (local and remote)
    for old in rel(out_dir(variant)).glob("recovery_u*"):
        if old.name != d.name or store.enabled:  # local copy no longer needed once uploaded
            store.delete(old, remote=old.name != d.name)
    for f in store.list(out_dir(variant)):
        part = f.split("/")[3] if f.count("/") >= 3 else ""
        if part.startswith("recovery_u") and part != d.name:
            store.delete(f"{out_dir(variant)}/{part}", local=False)
    store.squash()
    return size, time.perf_counter() - t0


def save_weights(variant, kind, tag, model, store, tok):
    """Evaluation artifacts: full model (FP32 master, safetensors) or effective adapters."""
    d = ensure_dir(f"{out_dir(variant)}/weights_{tag}")
    if kind == "full":
        model.save_pretrained(d, safe_serialization=True)
        tok.save_pretrained(d)
    else:
        from safetensors.torch import save_file

        save_file(ad.export_effective(model), str(d / "adapter_effective.safetensors"))
        torch.save(ad.adapter_state(model), d / "adapter_raw.pt")
    size = dir_size_bytes(d)
    store.push(d, f"{variant} weights {tag}")
    return size


def find_recovery(variant, store):
    """Latest recovery checkpoint available locally or remotely."""
    cands = {p.name for p in rel(out_dir(variant)).glob("recovery_u*") if (p / "trainer_state.pt").exists()}
    for f in store.list(out_dir(variant)):
        parts = f.split("/")
        if len(parts) > 3 and parts[3].startswith("recovery_u"):
            cands.add(parts[3])
    if not cands:
        return None
    name = sorted(cands)[-1]
    if not (rel(out_dir(variant)) / name / "trainer_state.pt").exists():
        store.pull(f"{out_dir(variant)}/{name}")
    return rel(out_dir(variant)) / name


def train(cfg, variant, max_hours=None, smoke_updates=None, store=None):
    store = store or Store()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(cfg["seed"])
    tcfg, vcfg = cfg["train"], cfg["variants"][variant]
    kind = vcfg["kind"]
    total_updates = smoke_updates or tcfg["updates"]
    ckpts = set(tcfg["checkpoint_updates"]) if not smoke_updates else {total_updates // 2, total_updates}
    mid = tcfg["midpoint_update"] if not smoke_updates else total_updates // 2
    run = f"{variant}_smoke" if smoke_updates else variant
    rdir = ensure_dir(res_dir(run))
    log_path = rdir / "log.jsonl"

    store.pull(res_dir(run))
    if (rdir / "summary.json").exists() and not smoke_updates:
        print(f"[{variant}] already finished; skipping")
        return
    tok = get_tokenizer(cfg["model_id"])
    store.pull(f"{cfg['data']['cache_dir']}/train_tokens.npz")
    data = TrainData(cfg)
    assert data.n >= total_updates * data.mb * data.accum, f"only {data.n} examples for {total_updates} updates"

    model, info = build(cfg, variant, device)
    model.train()
    opt = torch.optim.AdamW(param_groups(model, tcfg["weight_decay"]), lr=vcfg["lr"],
                            betas=tuple(tcfg["betas"]), eps=tcfg["eps"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_lambda(tcfg["warmup"], total_updates))
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))
    l2sp = L2SP(model, vcfg["l2sp_lambda"]) if vcfg.get("l2sp_lambda", 0) > 0 else None

    start, extra = 0, {"train_seconds": 0.0, "ckpt_seconds": 0.0, "skipped_updates": 0,
                       "checkpoint_sizes": {}, "displacement": {}, "peak_allocated_gb": 0.0}
    rec = None if smoke_updates else find_recovery(run, store)
    if rec is not None:
        st = torch.load(rec / "trainer_state.pt", weights_only=False)
        w = torch.load(rec / "weights.pt", map_location=device, weights_only=False)
        model.load_state_dict(w) if kind == "full" else ad.load_adapter_state(model, w)
        opt.load_state_dict(torch.load(rec / "optimizer.pt", map_location=device, weights_only=False))
        sched.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        set_rng_state(st["rng"])
        start = st["update"]
        extra = {k: st[k] for k in extra}
        # drop log lines written after the checkpoint (from a session that died)
        lines = [r for r in read_jsonl(log_path) if r["update"] <= start]
        log_path.write_text("".join(__import__("json").dumps(r) + "\n" for r in lines))
        print(f"[{variant}] resumed from {rec.name}")
    elif log_path.exists():
        log_path.unlink()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    session_t0 = time.perf_counter()
    autocast = dict(device_type=device, dtype=torch.float16, enabled=(device == "cuda"))

    for u in range(start, total_updates):
        t0 = time.perf_counter()
        n_tok = data.update_target_tokens(u)
        loss_sum = 0.0
        for input_ids, attn, labels in data.microbatches(u, tok.pad_token_id):
            input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)
            with torch.autocast(**autocast):
                ls, _ = sft_loss_sum(model, input_ids, attn, labels)
            loss = ls / n_tok
            scaler.scale(loss).backward()
            loss_sum += float(ls.detach())
        scaler.unscale_(opt)
        omega = l2sp.omega_and_grad(add_grad=True) if l2sp else None
        params = [p for g in opt.param_groups for p in g["params"]]
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, tcfg["clip"]))
        lr = opt.param_groups[0]["lr"]
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        skipped = scaler.get_scale() < scale_before
        sched.step()
        opt.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        extra["train_seconds"] += dt
        extra["skipped_updates"] += int(skipped)
        rec_line = {"update": u + 1, "examples": (u + 1) * data.mb * data.accum,
                    "loss_sft": loss_sum / n_tok, "lr": lr, "grad_norm": gnorm,
                    "scale": scale_before, "skipped": bool(skipped), "tokens": n_tok,
                    "seconds": dt}
        if l2sp:
            rec_line |= {"omega": omega, "lambda_omega": l2sp.lam * omega,
                         "total_loss": loss_sum / n_tok + l2sp.lam * omega}
        append_jsonl(rec_line, log_path)
        if (u + 1) % 25 == 0 or u < 3:
            print(f"[{variant}] u{u + 1} loss={rec_line['loss_sft']:.4f} lr={lr:.2e} "
                  f"gn={gnorm:.3f} {dt:.1f}s", flush=True)

        done = u + 1
        if done == mid or done == total_updates:
            tag = "final" if done == total_updates else f"u{done:04d}"
            if kind == "full":
                extra["displacement"][tag] = displacement(model, cfg["model_id"])
            t1 = time.perf_counter()
            extra["checkpoint_sizes"][f"weights_{tag}"] = save_weights(run, kind, tag, model, store, tok)
            extra["ckpt_seconds"] += time.perf_counter() - t1
        extra["peak_allocated_gb"] = max(extra["peak_allocated_gb"],
                                         gpu_memory().get("peak_allocated_gb", 0.0))
        out_of_time = max_hours and (time.perf_counter() - session_t0) / 3600 > max_hours
        if done in ckpts or (out_of_time and done < total_updates):
            if not smoke_updates or done == total_updates:
                size, secs = save_recovery(run, kind, done, model, opt, sched, scaler, extra, store)
                extra["checkpoint_sizes"]["recovery"] = size
                extra["ckpt_seconds"] += secs
        if out_of_time and done < total_updates:
            print(f"[{variant}] time guard hit at update {done}; exiting for resume")
            sys.exit(3)

    summary = {**info, "updates": total_updates, "examples": total_updates * data.mb * data.accum,
               **gpu_memory(), **extra, "env": env_info(), "smoke": bool(smoke_updates)}
    summary["train_hours"] = summary["train_seconds"] / 3600
    if l2sp:
        summary |= {"l2sp_reference_bytes": l2sp.ref_bytes,
                    "l2sp_reference_setup_seconds": l2sp.setup_seconds,
                    "l2sp_penalty_seconds_this_session": l2sp.penalty_seconds}
    write_json(summary, rdir / "summary.json")
    store.push(res_dir(run), f"{run} finished")
    print(f"[{variant}] done: {summary['train_hours']:.2f} h, peak "
          f"{summary.get('peak_allocated_gb', 0):.1f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--smoke-updates", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    train(cfg, args.variant, args.max_hours, args.smoke_updates)


if __name__ == "__main__":
    main()
