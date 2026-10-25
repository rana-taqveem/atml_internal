"""Task 3 offline evaluation (Steps 1-2): every method x budget N x order x question.

Ten seeded random orders of the 32 rollouts per question; nested prefixes of each order give
the candidate sets for N in {1,2,4,8,16,32}. Outcomes are averaged over orders within a
question, then across questions. Generated-token cost = sum of retained token counts of the
candidates consumed (FirstAns consumes only its first candidate).

    python -m task3_ttc.offline
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from common.utils import ensure_dir, load_config, rel, write_json
from task3_ttc.data import load, make_orders, splits
from task3_ttc.grading import grades
from task3_ttc.selection import build_pools, methods_for, select_all

OUT = "results/task3/offline"


def pools_for(cfg, name):
    return build_pools(load(cfg, name), grades(name))


def evaluate(cfg, name, pools, orders) -> pd.DataFrame:
    vlm = name == "mathvista"
    budgets = cfg["offline"]["budgets"]
    rows = []
    t0 = time.time()
    for qi, (q, p) in enumerate(pools.items()):
        corr = p.correct
        for o, perm in enumerate(orders[q]):
            for N in budgets:
                idx = perm[:N]
                chosen = select_all(p, idx, vlm)
                cost_all = int(p.T[idx].sum())
                base = {"sample_id": q, "order": o, "N": N, "oracle": bool(corr[idx].any()),
                        "n_groups": int(len(set(p.group[idx]))),
                        "n_correct_cands": int(corr[idx].sum())}
                for m, g in chosen.items():
                    rows.append({**base, "method": m, "group": int(g),
                                 "correct": bool(p.correct_group[g]),
                                 "cost": int(p.T[idx[0]]) if m == "first" else cost_all})
        if qi % 200 == 0:
            print(f"[{name}] {qi}/{len(pools)} ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def per_question(df):
    """Average over orders within each question."""
    return (df.groupby(["sample_id", "method", "N"], sort=False)
              .agg(acc=("correct", "mean"), oracle=("oracle", "mean"), cost=("cost", "mean"),
                   n_groups=("n_groups", "mean")).reset_index())


def bootstrap(pq: pd.DataFrame, n_boot: int, seed: int, ref="sc") -> pd.DataFrame:
    """Paired question-level bootstrap (same resampled questions for every method/N)."""
    qs = sorted(pq["sample_id"].unique())
    qpos = {q: i for i, q in enumerate(qs)}
    rng = np.random.default_rng(seed)
    B = rng.integers(0, len(qs), size=(n_boot, len(qs)))
    out = []
    piv = pq.pivot_table(index="sample_id", columns=["method", "N"], values="acc").loc[qs]
    for (m, N) in piv.columns:
        a = piv[(m, N)].to_numpy()
        boots = a[B].mean(axis=1)
        r = {"method": m, "N": N, "acc": a.mean(), "acc_lo": np.percentile(boots, 2.5),
             "acc_hi": np.percentile(boots, 97.5)}
        if (ref, N) in piv.columns and m != ref:
            d = a - piv[(ref, N)].to_numpy()
            db = d[B].mean(axis=1)
            r.update({f"diff_vs_{ref}": d.mean(), "diff_lo": np.percentile(db, 2.5),
                      "diff_hi": np.percentile(db, 97.5)})
        out.append(r)
    res = pd.DataFrame(out)
    extra = pq.groupby(["method", "N"]).agg(oracle=("oracle", "mean"), cost=("cost", "mean"),
                                           n_groups=("n_groups", "mean")).reset_index()
    return res.merge(extra, on=["method", "N"])


def decomposition(df):
    """Per (method, N): correct / correct-available-but-missed / no correct candidate."""
    d = df.assign(missed=lambda x: x["oracle"] & ~x["correct"], none=lambda x: ~x["oracle"])
    q = d.groupby(["sample_id", "method", "N"]).agg(
        correct=("correct", "mean"), missed=("missed", "mean"), none=("none", "mean")).reset_index()
    return q.groupby(["method", "N"])[["correct", "missed", "none"]].mean().reset_index()


def run(cfg):
    out = ensure_dir(OUT)
    sp = splits(cfg)
    for name in cfg["data"]["datasets"]:
        pools = pools_for(cfg, name)
        orders = make_orders(cfg, list(pools))
        np.savez_compressed(rel(f"{OUT}/orders_{name}.npz"),
                            **{q.replace(":", "_"): v for q, v in orders.items()})
        df = evaluate(cfg, name, pools, orders)
        df.to_parquet(out / f"selections_{name}.parquet", index=False)
        for split in ("dev", "eval"):
            sub = df[df["sample_id"].isin(set(sp[name][split]))]
            pq = per_question(sub)
            pq.to_parquet(out / f"per_question_{name}_{split}.parquet", index=False)
            bootstrap(pq, cfg["offline"]["n_boot"], cfg["seed"]).to_csv(
                out / f"summary_{name}_{split}.csv", index=False)
            decomposition(sub).to_csv(out / f"decomposition_{name}_{split}.csv", index=False)
        print(pd.read_csv(out / f"summary_{name}_eval.csv").query("N in (1, 8, 32)")
              .round(4).to_string())
    write_json({"methods": {n: methods_for(n == "mathvista") for n in cfg["data"]["datasets"]},
                "budgets": cfg["offline"]["budgets"], "orders": cfg["offline"]["n_orders"],
                "bootstrap": cfg["offline"]["n_boot"], "seed": cfg["seed"]},
               f"{OUT}/meta.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3.yaml")
    run(load_config(ap.parse_args().config))


if __name__ == "__main__":
    main()
