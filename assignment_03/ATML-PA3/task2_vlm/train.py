"""Task 2 training: one model through three curriculum stages (4,500 updates, 72,000 examples).

Stage 1 caption alignment (connector only, 2 passes over 18,000 captions, LR 1e-3);
Stage 2 visual instruction tuning (connector 1e-4 + new LoRA r16/a32 on q,k,v,o at 2e-5;
draws: presence 0.5 / counting 0.25 / caption replay 0.25); Stage 3 natural VQA
(connector 5e-5, LoRA 1e-5; VQAv2 0.7 / presence 0.1 / counting 0.1 / caption 0.1).
Each draw contributes two examples (a complete question pair, or two caption examples).
Microbatch 8 x accumulation 2; AdamW wd 0.01 (new optimizer per stage); per-stage 5% warmup
then cosine decay to 10% of peak; clip 1.0; FP16 autocast + dynamic loss scaling; frozen FP16
Qwen with FP32 connector/adapters. Loss: token-mean NLL per example, mean over the batch.

Development evaluation before training, every 500 updates and at stage ends (2250/3750/4500):
caption reference loss and reference-answer NLL (no termination token) under correct,
complementary and no-image inputs, plus the image-swap penalty dNLL = NLL(partner) - NLL(correct).

    python -m task2_vlm.train [--max-hours 11] [--smoke-scale 0.01]
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import Counter

import numpy as np
import torch

from common.store import Store
from common.utils import (append_jsonl, ensure_dir, env_info, gpu_memory, load_config, read_json,
                          read_jsonl, rel, seed_everything, write_json)
from task1_finetune import adapters as ad
from task2_vlm.features import load_features
from task2_vlm.vlm import VLM, Connector, Example, build_lm, calibrate_scale

DATA = "results/task2/data"


def run_dirs(cfg):
    sfx = cfg.get("run_suffix", "")
    return f"outputs/task2{sfx}", f"results/task2/train{sfx}"


class Pools:
    """Training/dev examples resolved to feature rows."""

    def __init__(self, cfg, index):
        r = lambda split, iid: index[f"{split}:{iid}"]
        self.captions = [Example(r("train2014", c["image_id"]), cfg["caption_prompt"], cap)
                         for c in read_json(f"{DATA}/captions_train.json") for cap in c["captions"]]
        pool = read_json(f"{DATA}/grounding_train_pool.json")
        self.pairs = {k: [[Example(r("train2014", it["image_id"]), p["question"], it["answer"])
                           for it in p["items"]] for p in v] for k, v in pool.items()}
        self.pairs["vqa"] = [[Example(r("train2014", it["image_id"]), p["question"], it["target"])
                              for it in p["items"]] for p in read_json(f"{DATA}/vqa_train.json")]
        self.train_rows = sorted({e.feat for e in self.captions})
        self.dev_captions = [Example(r("val2014", c["image_id"]), cfg["caption_prompt"], c["caption"])
                             for c in read_json(f"{DATA}/captions_dev.json")]
        self.dev_suites = {}
        for suite, key, tgt in (("grounding", "grounding_dev", "answer"), ("vqa", "vqa_dev", "target")):
            self.dev_suites[suite] = [(p["pair_id"], [(r("val2014", it["image_id"]), p["question"], it[tgt])
                                                      for it in p["items"]]) for p in read_json(f"{DATA}/{key}.json")]


class Sampler:
    """Stage 1: seeded permutation passes over captions. Stages 2-3: seeded mixture draws."""

    def __init__(self, pools, stage_cfg, seed, per_update):
        self.p, self.cfg, self.per_update = pools, stage_cfg, per_update
        self.rng = np.random.default_rng(seed)
        self.order, self.pos = None, 0

    def state(self):
        return {"rng": self.rng.bit_generator.state, "order": None if self.order is None else self.order.tolist(),
                "pos": self.pos}

    def load(self, st):
        self.rng.bit_generator.state = st["rng"]
        self.order = None if st["order"] is None else np.asarray(st["order"])
        self.pos = st["pos"]

    def next_update(self):
        mix = self.cfg["mix"]
        if list(mix) == ["caption"]:
            out = []
            while len(out) < self.per_update:
                if self.order is None or self.pos >= len(self.order):
                    self.order, self.pos = self.rng.permutation(len(self.p.captions)), 0
                out.append(self.p.captions[self.order[self.pos]]); self.pos += 1
            return out, Counter({"caption": len(out)})
        names, probs = list(mix), np.array(list(mix.values()))
        out, counts = [], Counter()
        for _ in range(self.per_update // 2):
            c = names[self.rng.choice(len(names), p=probs / probs.sum())]
            counts[c] += 1
            if c == "caption":
                out += [self.p.captions[i] for i in self.rng.integers(0, len(self.p.captions), 2)]
            else:
                out += self.p.pairs[c][self.rng.integers(len(self.p.pairs[c]))]
        return out, counts


def lr_lambda(total, warm_frac, min_frac):
    warm = max(1, round(warm_frac * total))

    def f(step):
        if step < warm:
            return (step + 1) / warm
        prog = min(1.0, (step - warm) / max(1, total - warm))
        return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * prog))
    return f


def caps_for(cfg):
    caps = cfg["train"]["caps"]
    prompt = cfg["caption_prompt"]
    return lambda e: caps["caption"] if e.question == prompt else caps["answer"]


@torch.no_grad()
def dev_eval(vlm, feats, pools, cfg, update):
    vlm.eval()
    cap = caps_for(cfg)
    ac = dict(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available())
    rec = {"update": update}
    with torch.autocast(**ac):
        l = [vlm.loss(pools.dev_captions[i:i + 16], feats, cap) for i in range(0, len(pools.dev_captions), 16)]
        rec["dev_caption_loss"] = float(torch.cat(l).mean())
        for suite, pairs in pools.dev_suites.items():
            nll = {}
            for cond in ("correct", "complementary", "none"):
                exs = []
                for _, items in pairs:
                    for k, (row, q, t) in enumerate(items):
                        feat = {"correct": row, "complementary": items[1 - k][0], "none": None}[cond]
                        exs.append(Example(feat, q, t))
                vals = torch.cat([vlm.loss(exs[i:i + 16], feats, cap, reduce_tokens="no_term")
                                  for i in range(0, len(exs), 16)])
                nll[cond] = vals.float().cpu().numpy()
                rec[f"{suite}_nll_{cond}"] = float(vals.mean())
            rec[f"{suite}_swap_penalty"] = float((nll["complementary"] - nll["correct"]).mean())
    vlm.train()
    return rec


def save_trainables(vlm, path, effective=False):
    sd = {f"connector.{k}": v.detach().cpu() for k, v in vlm.connector.state_dict().items()}
    sd.update({f"lora.{k}": v for k, v in (ad.export_effective(vlm.lm) if effective else ad.adapter_state(vlm.lm)).items()})
    torch.save(sd, path)


def load_trainables(vlm, sd, cfg):
    vlm.connector.load_state_dict({k[10:]: v for k, v in sd.items() if k.startswith("connector.")})
    lora = {k[5:]: v for k, v in sd.items() if k.startswith("lora.")}
    if lora and not ad.lora_modules(vlm.lm):
        attach_lora(vlm, cfg)
    if lora:
        ad.load_adapter_state(vlm.lm, lora)


def attach_lora(vlm, cfg):
    l = cfg["train"]["lora"]
    ad.attach_adapters(vlm.lm, l["targets"], l["r"], l["alpha"], l["dropout"], None, seed=cfg["seed"])


def build_vlm(cfg, device):
    lm, tok = build_lm(cfg["model_id"], device)
    conn = Connector(768, lm.config.hidden_size).to(device)
    return VLM(lm, tok, conn, cfg["n_visual_tokens"], cfg["train"]["caps"])


def stage_bounds(cfg):
    b, s = [], 0
    for st in cfg["train"]["stages"]:
        b.append((s, s + st["updates"]))
        s += st["updates"]
    return b


def main_train(cfg, max_hours=None):
    store = Store()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(cfg["seed"])
    odir, rdir = run_dirs(cfg)
    ensure_dir(odir); ensure_dir(rdir)
    store.pull(rdir)
    if rel(f"{rdir}/summary.json").exists():
        print("task2 training already finished"); return
    feats, index = load_features(cfg)
    pools = Pools(cfg, index)
    tcfg = cfg["train"]
    per_update = tcfg["microbatch"] * tcfg["grad_accum"]
    bounds = stage_bounds(cfg)
    total = bounds[-1][1]
    dev_points = sorted({0, *range(tcfg["dev_eval_every"], total + 1, tcfg["dev_eval_every"]),
                         *[b[1] for b in bounds]})

    torch.manual_seed(cfg["seed"])
    vlm = build_vlm(cfg, device)
    if tcfg["gradient_checkpointing"]:
        vlm.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    vlm.lm.train()
    tok_ids = [t for e in pools.captions[:2000] for t in vlm.fmt.enc(e.target)] + \
              [t for prs in pools.pairs.values() for p in prs[:500] for e in p for t in vlm.fmt.enc(e.question)]
    calib = calibrate_scale(vlm, feats, pools.train_rows, tok_ids, seed=cfg["seed"])
    write_json(calib, f"{rdir}/connector_calibration.json")

    # ---- resume
    state = {"update": 0, "stage_seconds": [0.0] * len(bounds), "stage_peak_gb": [0.0] * len(bounds),
             "skipped": 0, "trainable_by_stage": {}}
    rec_dir = latest_recovery(odir, store)
    ckpt = None
    if rec_dir is not None:
        ckpt = torch.load(rec_dir / "state.pt", weights_only=False)
        load_trainables(vlm, ckpt["trainables"], cfg)
        state = ckpt["state"]
        lines = [r for r in read_jsonl(f"{rdir}/log.jsonl") if r["update"] <= state["update"]]
        rel(f"{rdir}/log.jsonl").write_text("".join(__import__("json").dumps(r) + "\n" for r in lines))
        dl = [r for r in read_jsonl(f"{rdir}/dev_log.jsonl") if r["update"] <= state["update"]]
        rel(f"{rdir}/dev_log.jsonl").write_text("".join(__import__("json").dumps(r) + "\n" for r in dl))
        print(f"resumed at update {state['update']}")
    else:
        for f in ("log.jsonl", "dev_log.jsonl"):
            if rel(f"{rdir}/{f}").exists():
                rel(f"{rdir}/{f}").unlink()

    session_t0 = time.perf_counter()
    ac = dict(device_type=device, dtype=torch.float16, enabled=device == "cuda")
    cap = caps_for(cfg)
    if state["update"] == 0:
        append_jsonl(dev_eval(vlm, feats, pools, cfg, 0), f"{rdir}/dev_log.jsonl")

    for si, (s0, s1) in enumerate(bounds):
        if state["update"] >= s1:
            continue
        st = tcfg["stages"][si]
        if st["lora_lr"] and not ad.lora_modules(vlm.lm):
            attach_lora(vlm, cfg)
            if ckpt is not None and state["update"] > s0:
                load_trainables(vlm, ckpt["trainables"], cfg)
        groups = [{"params": list(vlm.connector.parameters()), "lr": st["connector_lr"], "name": "connector"}]
        if st["lora_lr"]:
            groups.append({"params": [p for m in ad.lora_modules(vlm.lm).values() for p in (m.lora_A, m.lora_B)],
                           "lr": st["lora_lr"], "name": "lora"})
        for g in groups:
            g["weight_decay"] = tcfg["weight_decay"]
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(s1 - s0, tcfg["warmup_frac"], tcfg["min_lr_frac"]))
        scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
        sampler = Sampler(pools, st, cfg["seed"] + si, per_update)
        if ckpt is not None and ckpt["stage"] == si and state["update"] > s0:
            opt.load_state_dict(ckpt["optimizer"]); sched.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"]); sampler.load(ckpt["sampler"])
            set_rng(ckpt["rng"])
        state["trainable_by_stage"][st["name"]] = sum(p.numel() for g in groups for p in g["params"])
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for u in range(state["update"], s1):
            t0 = time.perf_counter()
            batch, counts = sampler.next_update()
            loss_val = 0.0
            for m in range(tcfg["grad_accum"]):
                mb = batch[m * tcfg["microbatch"]:(m + 1) * tcfg["microbatch"]]
                with torch.autocast(**ac):
                    l = vlm.loss(mb, feats, cap)
                (scaler.scale(l.sum() / per_update)).backward()
                loss_val += float(l.detach().sum())
            scaler.unscale_(opt)
            params = [p for g in groups for p in g["params"]]
            gn = float(torch.nn.utils.clip_grad_norm_(params, tcfg["clip"]))
            lrs = {g["name"]: g["lr"] for g in opt.param_groups}
            before = scaler.get_scale()
            scaler.step(opt); scaler.update()
            skipped = scaler.get_scale() < before
            sched.step(); opt.zero_grad(set_to_none=True)
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            state["update"] = u + 1
            state["stage_seconds"][si] += dt
            state["skipped"] += int(skipped)
            state["stage_peak_gb"][si] = max(state["stage_peak_gb"][si], gpu_memory().get("peak_allocated_gb", 0.0))
            append_jsonl({"update": u + 1, "stage": st["name"], "loss": loss_val / per_update,
                          "lr_connector": lrs["connector"], "lr_lora": lrs.get("lora"), "grad_norm": gn,
                          "skipped": bool(skipped), "seconds": dt, "mix": dict(counts)}, f"{rdir}/log.jsonl")
            if (u + 1) % 50 == 0:
                print(f"[task2] u{u + 1} {st['name']} loss={loss_val / per_update:.4f} gn={gn:.2f} {dt:.2f}s", flush=True)
            if (u + 1) in dev_points:
                append_jsonl(dev_eval(vlm, feats, pools, cfg, u + 1), f"{rdir}/dev_log.jsonl")
            if u + 1 == s1:
                save_trainables(vlm, rel(odir) / f"weights_stage{si + 1}.pt")
                save_trainables(vlm, rel(odir) / f"weights_stage{si + 1}_effective.pt", effective=True)
                store.push(f"{odir}/weights_stage{si + 1}.pt"); store.push(f"{odir}/weights_stage{si + 1}_effective.pt")
            out_of_time = max_hours and (time.perf_counter() - session_t0) / 3600 > max_hours
            if (u + 1) % tcfg["checkpoint_every"] == 0 or (out_of_time and u + 1 < total):
                save_recovery(vlm, odir, rdir, si, state, opt, sched, scaler, sampler, store)
            if out_of_time and u + 1 < total:
                print(f"time guard at update {u + 1}; exiting for resume"); sys.exit(3)
        ckpt = None
    summary = {"updates": total, "examples": total * per_update, "calibration": calib,
               "stage_seconds": dict(zip([s["name"] for s in tcfg["stages"]], state["stage_seconds"])),
               "stage_peak_gb": dict(zip([s["name"] for s in tcfg["stages"]], state["stage_peak_gb"])),
               "trainable_by_stage": state["trainable_by_stage"], "skipped_updates": state["skipped"],
               "train_hours": sum(state["stage_seconds"]) / 3600, "env": env_info()}
    write_json(summary, f"{rdir}/summary.json")
    store.push(rdir, "task2 training finished")


def set_rng(st):
    random.setstate(st["python"]); np.random.set_state(st["numpy"]); torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def save_recovery(vlm, odir, rdir, si, state, opt, sched, scaler, sampler, store):
    d = ensure_dir(f"{odir}/recovery_u{state['update']:04d}")
    rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    sd = {f"connector.{k}": v.detach().cpu() for k, v in vlm.connector.state_dict().items()}
    sd.update({f"lora.{k}": v for k, v in ad.adapter_state(vlm.lm).items()})
    torch.save({"trainables": sd, "state": state, "stage": si, "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                "sampler": sampler.state(), "rng": rng}, d / "state.pt")
    store.push(d, f"task2 recovery u{state['update']}")
    store.push(rdir, f"task2 logs u{state['update']}")
    for old in rel(odir).glob("recovery_u*"):
        if old.name != d.name:
            store.delete(old)
    for f in store.list(odir):
        part = f.split("/")[2] if f.count("/") >= 2 else ""
        if part.startswith("recovery_u") and part != d.name:
            store.delete(f"{odir}/{part}", local=False)
    store.squash()


def latest_recovery(odir, store):
    names = {p.name for p in rel(odir).glob("recovery_u*") if (p / "state.pt").exists()}
    names |= {f.split("/")[2] for f in store.list(odir) if f.count("/") >= 3 and f.split("/")[2].startswith("recovery_u")}
    if not names:
        return None
    name = sorted(names)[-1]
    if not (rel(odir) / name / "state.pt").exists():
        store.pull(f"{odir}/{name}")
    return rel(odir) / name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2.yaml")
    ap.add_argument("--max-hours", type=float, default=None)
    a = ap.parse_args()
    main_train(load_config(a.config), a.max_hours)


if __name__ == "__main__":
    main()
