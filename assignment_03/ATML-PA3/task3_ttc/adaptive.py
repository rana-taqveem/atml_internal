"""Task 3 Step 5: offline adaptive sampling over the saved random orders.

Candidates are revealed in each order; stopping is checked after n in {4, 8, 16} candidates,
otherwise sampling stops at 32. Both controllers return the current self-consistency answer.
  agreement(tau):        unique non-NO_ANSWER leading answer with count/n >= tau
  quality(tau, gamma):   agreement(tau) AND SC winner == RWC winner AND RWC margin >= gamma
RWC uses self-certainty (MATH) or VisConf (MathVista), with scores, VisConf calibration and
ranks recomputed from the revealed candidates only. Margin = (W1 - W2) / sum(W), W2 = 0 when
only one answer group exists. Settings are chosen on the development split: lowest mean token
cost with dev accuracy >= fixed-32 SC accuracy - 1 pp (ties: higher accuracy, smaller tau,
smaller gamma); the always-32 policy is included as a fallback. Frozen settings are then
evaluated on the held-out split.

    python -m task3_ttc.adaptive
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from common.utils import ensure_dir, load_config, write_json
from task3_ttc.data import make_orders, splits
from task3_ttc.offline import pools_for
from task3_ttc.selection import margin, rwc, visconf, vote

OUT = "results/task3/adaptive"


def checkpoint_states(cfg, name, pools, orders) -> pd.DataFrame:
    """Everything a controller can observe after n revealed candidates."""
    vlm = name == "mathvista"
    rows = []
    for q, p in pools.items():
        for o, perm in enumerate(orders[q]):
            for n in cfg["adaptive"]["checkpoints"] + [cfg["adaptive"]["max_n"]]:
                idx = perm[:n]
                groups = p.group[idx]
                sc_g, tot = vote(groups, np.ones(n))
                counts = sorted(tot.values(), reverse=True)
                unique_lead = len(counts) == 1 or counts[0] > counts[1]
                score = visconf(p, idx)[0] if vlm else p.K[idx]
                rwc_g, wtot = rwc(groups, score)
                rows.append({"sample_id": q, "order": o, "n": n, "sc_group": int(sc_g),
                             "sc_correct": bool(p.correct_group[sc_g]),
                             "lead_frac": tot[sc_g] / n,
                             "lead_ok": bool(unique_lead and not p.no_answer[idx][groups == sc_g][0]),
                             "rwc_agree": bool(rwc_g == sc_g), "rwc_margin": margin(wtot),
                             "cost": int(p.T[idx].sum()),
                             "pool_correct_frac": float(p.correct.mean())})
    return pd.DataFrame(rows)


def simulate(states: pd.DataFrame, cfg, tau=None, gamma=None) -> pd.DataFrame:
    """Per (question, order) outcome of one policy. tau=None -> always 32."""
    s = states.copy()
    if tau is None:
        stop = s["n"] == cfg["adaptive"]["max_n"]
    else:
        stop = s["lead_ok"] & (s["lead_frac"] >= tau - 1e-12)
        if gamma is not None:
            stop &= s["rwc_agree"] & (s["rwc_margin"] >= gamma - 1e-12)
        stop |= s["n"] == cfg["adaptive"]["max_n"]
    s = s[stop].sort_values("n").groupby(["sample_id", "order"], as_index=False).first()
    return s


def summarize_policy(sim: pd.DataFrame) -> dict:
    pq = sim.groupby("sample_id").agg(acc=("sc_correct", "mean"), cost=("cost", "mean"),
                                      n=("n", "mean"))
    return {"acc": pq["acc"].mean(), "cost": pq["cost"].mean(), "rollouts": pq["n"].mean(),
            "stop_dist": sim["n"].value_counts(normalize=True).sort_index().to_dict()}


def grid(states, cfg):
    a = cfg["adaptive"]
    res = [{"family": "fixed32", "tau": None, "gamma": None, **summarize_policy(simulate(states, cfg))}]
    for t in a["taus"]:
        res.append({"family": "agreement", "tau": t, "gamma": None,
                    **summarize_policy(simulate(states, cfg, t))})
    for t, g in itertools.product(a["taus"], a["gammas"]):
        res.append({"family": "quality", "tau": t, "gamma": g,
                    **summarize_policy(simulate(states, cfg, t, g))})
    return pd.DataFrame(res)


def choose(dev: pd.DataFrame, family: str, tol_pp: float) -> dict:
    target = dev.loc[dev["family"] == "fixed32", "acc"].iloc[0] - tol_pp / 100
    cand = dev[(dev["family"].isin([family, "fixed32"])) & (dev["acc"] >= target - 1e-12)].copy()
    cand["tau_s"] = cand["tau"].fillna(np.inf)
    cand["gamma_s"] = cand["gamma"].fillna(np.inf)
    best = cand.sort_values(["cost", "acc", "tau_s", "gamma_s"],
                            ascending=[True, False, True, True]).iloc[0]
    return {"family": best["family"], "tau": None if pd.isna(best["tau"]) else float(best["tau"]),
            "gamma": None if pd.isna(best["gamma"]) else float(best["gamma"]),
            "dev_acc": float(best["acc"]), "dev_cost": float(best["cost"]), "dev_target": float(target)}


def bootstrap_diff(a: pd.Series, b: pd.Series, n_boot, seed):
    d = (a - b).to_numpy()
    B = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    db = d[B].mean(1)
    return float(d.mean()), float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))


def run(cfg):
    out = ensure_dir(OUT)
    sp = splits(cfg)
    report = {}
    for name in cfg["data"]["datasets"]:
        pools = pools_for(cfg, name)
        orders = make_orders(cfg, list(pools))
        states = checkpoint_states(cfg, name, pools, orders)
        states.to_parquet(out / f"states_{name}.parquet", index=False)
        dev = states[states["sample_id"].isin(set(sp[name]["dev"]))]
        ev = states[states["sample_id"].isin(set(sp[name]["eval"]))]
        dgrid = grid(dev, cfg)
        dgrid.to_csv(out / f"dev_grid_{name}.csv", index=False)
        egrid = grid(ev, cfg)  # retrospective only - never used for selection
        egrid.to_csv(out / f"eval_grid_retrospective_{name}.csv", index=False)
        report[name] = {}
        fixed32 = simulate(ev, cfg)
        f32q = fixed32.groupby("sample_id")["sc_correct"].mean()
        for family in ("agreement", "quality"):
            sel = choose(dgrid, family, cfg["adaptive"]["tolerance_pp"])
            sim = simulate(ev, cfg, sel["tau"], sel["gamma"] if family == "quality" else None) \
                if sel["family"] != "fixed32" else fixed32
            sim.to_parquet(out / f"eval_{family}_{name}.parquet", index=False)
            pol = summarize_policy(sim)
            accq = sim.groupby("sample_id")["sc_correct"].mean()
            d, lo, hi = bootstrap_diff(accq, f32q.loc[accq.index], cfg["offline"]["n_boot"], cfg["seed"])
            merged = sim.merge(fixed32[["sample_id", "order", "sc_correct"]],
                               on=["sample_id", "order"], suffixes=("", "_f32"))
            early_wrong = merged[(merged["n"] < 32) & ~merged["sc_correct"]]
            report[name][family] = {
                **sel, "eval": pol, "diff_vs_fixed32": d, "diff_lo": lo, "diff_hi": hi,
                "early_stops": float((sim["n"] < 32).mean()),
                "incorrect_early_stops": float(len(early_wrong) / len(sim)),
                "incorrect_early_recoverable_by_fixed32": float(early_wrong["sc_correct_f32"].mean())
                if len(early_wrong) else 0.0,
                "mean_pool_correct_frac_incorrect_early": float(early_wrong["pool_correct_frac"].mean())
                if len(early_wrong) else None}
        # fixed-N SC reference at the checkpoints, for cost-matched comparison
        report[name]["fixed_sc"] = {int(n): summarize_policy(ev[ev["n"] == n]) for n in sorted(ev["n"].unique())}
        print(name, {k: {kk: vv for kk, vv in v.items() if kk != "stop_dist"} if isinstance(v, dict) else v
                     for k, v in report[name].items() if k != "fixed_sc"})
    write_json(report, out / "selected_policies.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3.yaml")
    run(load_config(ap.parse_args().config))


if __name__ == "__main__":
    main()
