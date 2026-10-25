"""Task 3 Step 3: what the signals capture, at N = 32 (all rollouts of a question).

* Within-question quartile contrast: for each score, accuracy of the top-8 minus bottom-8
  rollouts of a question (ties broken by rollout id), averaged over questions (all questions
  and "mixed" questions with both correct and incorrect rollouts). Length (-T) is the
  length-only control.
* Within-question Spearman correlation of each score with length, averaged over questions.
* MathVista: per-rollout Pearson correlation between calibrated confidence c_t and visual
  engagement v_t across tokens (calibration on all 32 rollouts), summarised over rollouts and
  questions; numerical check of Z^tok - Z = Cov_t(c, v).
* Fixed illustrative trajectories (selection rule below) for plotting c_t and v_t.

    python -m task3_ttc.signals
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common.utils import ensure_dir, load_config, write_json
from task3_ttc.data import splits
from task3_ttc.offline import pools_for

OUT = "results/task3/signals"


def calibrated(p, idx):
    pooled = np.concatenate([p.kl[i] for i in idx])
    lo, hi = np.percentile(pooled, [5, 95])
    s = max(hi - lo, 1e-8)
    return [np.clip((p.kl[i] - lo) / s, 0, 1) for i in idx]


def scores_at_32(p, vlm):
    idx = np.arange(32)
    sc = {"logp": p.L, "cert": p.K, "neg_length": -p.T.astype(float)}
    if vlm:
        c = calibrated(p, idx)
        sc["vision"] = p.V
        sc["visconf"] = np.array([p.V[i] * c[i].mean() if len(c[i]) else np.nan for i in idx])
        sc["tokvisconf"] = np.array([(c[i] * p.v[i]).mean() if len(c[i]) else np.nan for i in idx])
        return sc, c
    return sc, None


def quartile_rows(name, pools, ids, vlm):
    rows = []
    for q in ids:
        p = pools[q]
        corr = p.correct.astype(float)
        sc, _ = scores_at_32(p, vlm)
        mixed = 0 < corr.sum() < 32
        for s, v in sc.items():
            v = np.where(np.isnan(v), -np.inf, v)
            order = np.lexsort((np.arange(32), -v))       # descending score, then rollout id
            top, bot = order[:8], order[-8:]
            rho_len = spearmanr(v, p.T).statistic if np.isfinite(v).all() else np.nan
            rows.append({"dataset": name, "sample_id": q, "score": s, "mixed": mixed,
                         "top_q_acc": corr[top].mean(), "bottom_q_acc": corr[bot].mean(),
                         "top_minus_bottom": corr[top].mean() - corr[bot].mean(),
                         "spearman_with_length": rho_len})
    return pd.DataFrame(rows)


def conf_vision(pools, ids):
    rows, cov_err = [], 0.0
    for q in ids:
        p = pools[q]
        c = calibrated(p, np.arange(32))
        for i in range(32):
            ci, vi = c[i], p.v[i]
            if len(ci) < 3:
                continue
            zt, z = (ci * vi).mean(), ci.mean() * vi.mean()
            cov = ((ci - ci.mean()) * (vi - vi.mean())).mean()
            cov_err = max(cov_err, abs((zt - z) - cov))
            r = np.corrcoef(ci, vi)[0, 1] if ci.std() > 0 and vi.std() > 0 else np.nan
            rows.append({"sample_id": q, "rollout_id": i, "pearson_c_v": r, "cov_c_v": cov,
                         "T": len(ci), "correct": bool(p.correct[i])})
    return pd.DataFrame(rows), cov_err


def trajectories(pools, ids, n=3):
    """Rule: the first n held-out questions (split order) whose N=32 VisConf-top and
    tokenwise-VisConf-top rollouts differ; store c_t and v_t for both rollouts."""
    out = []
    for q in ids:
        p = pools[q]
        sc, c = scores_at_32(p, True)
        a = int(np.nanargmax(sc["visconf"]))
        b = int(np.nanargmax(sc["tokvisconf"]))
        if a == b:
            continue
        out.append({"sample_id": q, "visconf_top": a, "tokvisconf_top": b,
                    "series": {str(i): {"c": c[i].round(4).tolist(), "v": p.v[i].round(4).tolist(),
                                        "correct": bool(p.correct[i]),
                                        "group": int(p.group[i])} for i in (a, b)}})
        if len(out) == n:
            break
    return out


def run(cfg):
    out = ensure_dir(OUT)
    sp = splits(cfg)
    summary = {}
    for name in cfg["data"]["datasets"]:
        vlm = name == "mathvista"
        pools = pools_for(cfg, name)
        ids = sp[name]["eval"]
        qdf = quartile_rows(name, pools, ids, vlm)
        qdf.to_csv(out / f"quartiles_per_question_{name}.csv", index=False)
        agg = []
        for subset, d in (("all", qdf), ("mixed", qdf[qdf["mixed"]])):
            g = d.groupby("score").agg(top_q_acc=("top_q_acc", "mean"),
                                       bottom_q_acc=("bottom_q_acc", "mean"),
                                       top_minus_bottom=("top_minus_bottom", "mean"),
                                       spearman_with_length=("spearman_with_length", "mean"),
                                       n_questions=("sample_id", "nunique")).reset_index()
            agg.append(g.assign(subset=subset))
        agg = pd.concat(agg)
        agg.to_csv(out / f"quartiles_{name}.csv", index=False)
        summary[name] = {"quartiles": agg.round(4).to_dict("records")}
        if vlm:
            cv, err = conf_vision(pools, ids)
            cv.to_csv(out / "conf_vision_per_rollout.csv", index=False)
            per_q = cv.groupby("sample_id")["pearson_c_v"].mean()
            summary[name]["conf_vision"] = {
                "rollouts": int(cv["pearson_c_v"].notna().sum()),
                "mean_rollout_corr": float(cv["pearson_c_v"].mean()),
                "median_rollout_corr": float(cv["pearson_c_v"].median()),
                "frac_positive": float((cv["pearson_c_v"] > 0).mean()),
                "mean_question_corr": float(per_q.mean()),
                "mean_corr_correct": float(cv.loc[cv["correct"], "pearson_c_v"].mean()),
                "mean_corr_incorrect": float(cv.loc[~cv["correct"], "pearson_c_v"].mean()),
                "max_abs_error_ztok_minus_z_vs_cov": err}
            write_json(trajectories(pools, ids), out / "trajectories.json")
        print(name, agg.round(3).to_string())
    write_json(summary, out / "summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3.yaml")
    run(load_config(ap.parse_args().config))


if __name__ == "__main__":
    main()
