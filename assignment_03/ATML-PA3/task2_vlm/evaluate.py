"""Task 2 final evaluation on both final-test suites under four image conditions.

Conditions: correct image, complementary swap (partner image), mismatched (fixed unrelated
image from another pair of the same suite), no image (visual block removed). Greedy decoding,
thinking disabled, 16 new tokens, batch size 8; references and questions fixed.

Grounding: normalised exact match (presence / counting separately). VQAv2: local soft
consensus accuracy against all ten answers (yes/no / number / other separately).
Per suite and condition: accuracy, pair accuracy (both predictions match their original
canonical answers), identical-answer rate, empty-answer rate, mean reference-answer NLL,
generation time per answer; swap behaviour (follows partner answer / keeps original / other).
Correct-image advantages over complementary swaps (accuracy and NLL) with 95% intervals from
4,000 bootstrap resamples of complete pairs (same sampled pair IDs for every comparison).

    python -m task2_vlm.evaluate [--stage 3]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from common.store import Store
from common.utils import ensure_dir, gpu_memory, load_config, read_json, rel, write_json
from task2_vlm.features import load_features
from task2_vlm.train import DATA, build_vlm, load_trainables, run_dirs
from task2_vlm.vlm import Example
from task2_vlm.vqa_eval import normalize, soft_accuracy

CONDS = ["correct", "complementary", "mismatched", "none"]


def suite_items(suite, index):
    key = {"grounding": "grounding_test", "vqa": "vqa_test"}[suite]
    mm = read_json(f"{DATA}/mismatched_test.json")
    rows = []
    for p in read_json(f"{DATA}/{key}.json"):
        for k, it in enumerate(p["items"]):
            partner = p["items"][1 - k]
            rows.append({
                "suite": suite, "pair_id": p["pair_id"], "k": k, "question": p["question"],
                "qtype": p.get("type") or p.get("answer_type"),
                "image_id": it["image_id"], "partner_image_id": partner["image_id"],
                "reference": it.get("answer") or it["target"],
                "canonical": it.get("answer") or it["consensus"],
                "partner_canonical": partner.get("answer") or partner["consensus"],
                "answers": it.get("answers"),
                "rows": {"correct": index[f"val2014:{it['image_id']}"],
                         "complementary": index[f"val2014:{partner['image_id']}"],
                         "mismatched": index[f"val2014:{mm[f'{suite}:{p['pair_id']}:{k}']}"],
                         "none": None}})
    return rows


def score(row, pred):
    if row["suite"] == "vqa":
        return soft_accuracy(pred, row["answers"])
    return float(normalize(pred) == normalize(row["canonical"]))


@torch.no_grad()
def run_condition(vlm, feats, rows, cond, ecfg):
    vlm.eval()
    ac = dict(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available())
    preds, nlls = [], []
    t0 = time.perf_counter()
    for i in range(0, len(rows), ecfg["batch_size"]):
        chunk = rows[i:i + ecfg["batch_size"]]
        exs = [Example(r["rows"][cond], r["question"]) for r in chunk]
        with torch.autocast(**ac):
            preds += vlm.generate(exs, feats, ecfg["max_new_tokens"])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gen_s = time.perf_counter() - t0
    for i in range(0, len(rows), ecfg["batch_size"]):
        chunk = rows[i:i + ecfg["batch_size"]]
        exs = [Example(r["rows"][cond], r["question"], r["reference"]) for r in chunk]
        with torch.autocast(**ac):
            nlls += vlm.loss(exs, feats, lambda e: 16, reduce_tokens="no_term").float().cpu().tolist()
    return preds, nlls, gen_s / len(rows)


def pair_bootstrap(df, n_boot, seed):
    """Correct-image advantage over complementary swaps, resampling complete pairs."""
    pairs = sorted(df["pair_id"].unique())
    rng = np.random.default_rng(seed)
    B = rng.integers(0, len(pairs), size=(n_boot, len(pairs)))
    piv = df.pivot_table(index="pair_id", columns="condition", values=["score", "nll"], aggfunc="mean").loc[pairs]
    out = {}
    for metric, sign in (("score", 1), ("nll", -1)):
        d = sign * (piv[(metric, "correct")] - piv[(metric, "complementary")]).to_numpy()
        db = d[B].mean(1)
        out[f"{metric}_advantage"] = {"mean": float(d.mean()), "lo": float(np.percentile(db, 2.5)),
                                      "hi": float(np.percentile(db, 97.5))}
    for c in CONDS:
        a = piv[("score", c)].to_numpy()
        ab = a[B].mean(1)
        out[f"score_{c}"] = {"mean": float(a.mean()), "lo": float(np.percentile(ab, 2.5)),
                             "hi": float(np.percentile(ab, 97.5))}
    return out


def metrics(df):
    rows = []
    for (suite, cond), d in df.groupby(["suite", "condition"]):
        pair = d.groupby("pair_id").agg(
            both=("canon_match", "all"),
            identical=("pred_norm", lambda s: s.nunique() == 1))
        r = {"suite": suite, "condition": cond, "accuracy": d["score"].mean(),
             "pair_accuracy": pair["both"].mean(), "identical_answer_rate": pair["identical"].mean(),
             "empty_answer_rate": (d["pred_norm"] == "").mean(), "mean_ref_nll": d["nll"].mean(),
             "gen_seconds_per_answer": d["gen_s_per_answer"].iloc[0], "n": len(d)}
        for t, dt in d.groupby("qtype"):
            r[f"acc_{t}"] = dt["score"].mean()
        if cond == "complementary":
            r["swap_follows_partner"] = d["follows_partner"].mean()
            r["swap_keeps_original"] = d["canon_match"].mean()
            r["swap_other"] = 1 - d["follows_partner"].mean() - d["canon_match"].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def examples(df, n=3):
    """Selection rule: per suite, the first n test pairs (file order) whose two correct-image
    predictions are both right, and the first n whose correct-image predictions are identical."""
    out = {}
    for suite, d in df.groupby("suite"):
        corr = d[d["condition"] == "correct"].groupby("pair_id", sort=False)
        good = [p for p, g in corr if g["canon_match"].all()][:n]
        same = [p for p, g in corr if g["pred_norm"].nunique() == 1][:n]
        def pack(pid):
            g = d[d["pair_id"] == pid]
            return [{"k": int(k), "question": gk["question"].iloc[0], "image_id": int(gk["image_id"].iloc[0]),
                     "reference": gk["reference"].iloc[0],
                     "predictions": dict(zip(gk["condition"], gk["prediction"]))}
                    for k, gk in g.groupby("k")]
        out[suite] = {"both_correct": {p: pack(p) for p in good}, "identical_answers": {p: pack(p) for p in same}}
    return out


def run(cfg, stage=3):
    store = Store()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    odir, rdir = run_dirs(cfg)
    out = ensure_dir(f"results/task2/eval{cfg.get('run_suffix', '')}")
    wpath = f"{odir}/weights_stage{stage}.pt"
    if not rel(wpath).exists():
        assert store.pull(wpath), f"missing {wpath}"
    feats, index = load_features(cfg)
    vlm = build_vlm(cfg, device)
    load_trainables(vlm, torch.load(rel(wpath), weights_only=False), cfg)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    recs = []
    for suite in ("grounding", "vqa"):
        rows = suite_items(suite, index)
        for cond in CONDS:
            preds, nlls, gs = run_condition(vlm, feats, rows, cond, cfg["eval"])
            for r, p, l in zip(rows, preds, nlls):
                pn = normalize(p)
                recs.append({**{k: v for k, v in r.items() if k not in ("rows", "answers")},
                             "condition": cond, "prediction": p, "pred_norm": pn, "score": score(r, p),
                             "canon_match": pn == normalize(r["canonical"]),
                             "follows_partner": pn == normalize(r["partner_canonical"]),
                             "nll": l, "gen_s_per_answer": gs})
    df = pd.DataFrame(recs)
    df.to_csv(out / "predictions.csv", index=False)
    m = metrics(df)
    m.to_csv(out / "metrics.csv", index=False)
    boot = {s: pair_bootstrap(d, cfg["eval"]["n_boot"], cfg["seed"]) for s, d in df.groupby("suite")}
    write_json({"bootstrap": boot, "gpu": gpu_memory(), "stage": stage}, out / "bootstrap.json")
    write_json(examples(df), out / "qualitative_examples.json")
    store.push(out, "task2 final eval")
    print(m.round(3).to_string())
    print(json.dumps(boot, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2.yaml")
    ap.add_argument("--stage", type=int, default=3)
    a = ap.parse_args()
    run(load_config(a.config), a.stage)


if __name__ == "__main__":
    main()
