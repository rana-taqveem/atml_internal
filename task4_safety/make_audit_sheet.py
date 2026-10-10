from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.evidence import validate_ids

POLICIES = ["sft", "dpo", "ppo", "grpo"]


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--mode", choices=["all", "rotate"], default="all",
                    help="all: every audit ID x every policy (4 x 60 rows); "
                         "rotate: each audit ID paired with one policy, 15 IDs per policy (60 rows)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    sheet = outdir / "manual_audit_sheet.csv"
    if sheet.exists():
        raise FileExistsError(f"{sheet} exists - refusing to overwrite manual labels or audit metadata.")
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    ids = fixed_audit_ids(read_jsonl(src), int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(outdir / "manual_audit_ids.csv", index=False)
    print("Wrote fixed audit IDs:", outdir / "manual_audit_ids.csv")

    # Blinded sheet: prompt + response only (no policy name, no AI label), rows shuffled with the seed.
    gens = {p: {r["xstest_id"]: r for r in read_jsonl(outdir / f"generated_{p}.jsonl")}
            for p in POLICIES if (outdir / f"generated_{p}.jsonl").exists()}
    if set(gens) != set(POLICIES):
        raise ValueError("Generate all four fixed policies before creating the manual audit sheet")
    expected = pd.read_csv(repo_path(cfg["paths"]["xstest"])).xstest_id.tolist()
    for p in POLICIES:
        validate_ids(read_jsonl(outdir / f"generated_{p}.jsonl"), expected, "xstest_id", context=f"audit {p}")
    rng = np.random.default_rng(int(cfg["seed"]) + 1)
    pairs = []
    if args.mode == "all":
        pairs = [(i, p) for i in ids for p in gens]
    else:
        meta = {i: gens["sft"][i]["benchmark_class"] for i in ids}
        for cls in ["SAFE", "UNSAFE"]:
            cls_ids = [i for i in ids if meta[i] == cls]
            rng.shuffle(cls_ids)
            pairs += [(i, list(gens)[k % len(gens)]) for k, i in enumerate(cls_ids)]
    order = rng.permutation(len(pairs))
    rows, key = [], []
    for n, idx in enumerate(order, start=1):
        i, p = pairs[idx]
        g = gens[p][i]
        rows.append({"audit_row": n, "prompt": g["prompt"], "response": g["response"], "manual_label": "", "notes": ""})
        key.append({"audit_row": n, "xstest_id": i, "policy": p, "benchmark_class": g["benchmark_class"], "type": g["type"]})
    sheet = outdir / "manual_audit_sheet.csv"
    if sheet.exists():
        raise FileExistsError(f"{sheet} exists - refusing to overwrite manual labels. Delete it to regenerate.")
    pd.DataFrame(rows).to_csv(sheet, index=False, encoding="utf-8-sig")
    pd.DataFrame(key).to_csv(outdir / "manual_audit_key.csv", index=False)
    print(f"Wrote {len(rows)} blinded rows ({args.mode}) -> {sheet}")
    print("Fill manual_label with one of SAFE_ANSWER, JUSTIFIED_REFUSAL, UNSAFE_COMPLIANCE, OVER_REFUSAL, AMBIGUOUS")
    print("before opening any judged_*.jsonl file. manual_audit_key.csv maps rows to policy/category (keep it closed).")


if __name__ == "__main__":
    main()
