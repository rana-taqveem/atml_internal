"""Task 3 data: download, join/verify the supplied rollout tables, fixed splits and orders.

    python -m task3_ttc.data            # download (if needed), verify, write splits/orders
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np
import pandas as pd

from common.utils import ensure_dir, load_config, read_json, rel, write_json

KEYS = ["dataset", "sample_id", "rollout_id"]
SPLITS = "results/task3/splits.json"


def download(cfg):
    raw = rel(cfg["data"]["raw_dir"])
    if all((raw / d / "generations.parquet").exists() for d in cfg["data"]["datasets"].values()):
        return
    raw.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "gdown", cfg["data"]["drive_folder"], "-O", str(raw)],
                   check=True)
    meta = rel(cfg["data"]["cache_dir"]) / "mathvista_testmini.json"
    if not meta.exists():
        import urllib.request

        meta.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(cfg["data"]["mathvista_meta_url"], meta)


def load(cfg, name: str) -> pd.DataFrame:
    """One row per rollout with token-level arrays; verifies alignment."""
    d = rel(cfg["data"]["raw_dir"]) / cfg["data"]["datasets"][name]
    gen = pd.read_parquet(d / "generations.parquet")
    prob = pd.read_parquet(d / "probability_metrics.parquet")
    df = gen.merge(prob, on=KEYS, how="inner", validate="one_to_one")
    if (d / "attention_metrics.parquet").exists():
        att = pd.read_parquet(d / "attention_metrics.parquet").rename(
            columns={"n_generated_text_tokens": "att_prev_counts"})
        df = df.merge(att, on=KEYS, how="inner", validate="one_to_one")
    assert len(df) == len(gen) == 32000, len(df)
    T = df["n_generated_text_tokens"].to_numpy()
    for col in ["generated_token_ids", "logp", "kl_u_p"]:
        lens = df[col].map(len).to_numpy()
        assert (lens == T).all(), f"{name}: {col} misaligned in {(lens != T).sum()} rows"
    if "img_attn_total" in df:
        for col in ["img_attn_total", "prompt_text_attn_total", "generated_text_attn_total"]:
            assert (df[col].map(len).to_numpy() == T).all(), col
        assert all((np.asarray(a) == np.arange(len(a))).all() for a in df["att_prev_counts"])
    df["sample_id"] = df["sample_id"].astype(str)
    return df.sort_values(["sample_id", "rollout_id"]).reset_index(drop=True)


def make_splits(cfg, frames: dict) -> dict:
    """Sort the 1,000 IDs lexicographically, permute with default_rng(6304);
    first 200 = development, remaining 800 = evaluation; MATH live subset = first 50 eval IDs."""
    out = {"seed": cfg["seed"], "rule": "sorted(ids) permuted by np.random.default_rng(6304)"}
    for name, df in frames.items():
        ids = sorted(df["sample_id"].unique())
        assert len(ids) == 1000
        perm = np.random.default_rng(cfg["seed"]).permutation(len(ids))
        order = [ids[i] for i in perm]
        out[name] = {"dev": order[:cfg["split"]["n_dev"]], "eval": order[cfg["split"]["n_dev"]:]}
    out["math"]["live"] = out["math"]["eval"][:cfg["split"]["n_live"]]
    return out


def make_orders(cfg, ids: list[str]) -> dict:
    """Ten seeded permutations of the 32 rollout IDs per question (questions in sorted ID order)."""
    rng = np.random.default_rng(cfg["seed"])
    return {q: np.stack([rng.permutation(32) for _ in range(cfg["offline"]["n_orders"])])
            for q in sorted(ids)}


def splits(cfg=None) -> dict:
    p = rel(SPLITS)
    if not p.exists():
        cfg = cfg or load_config("configs/task3.yaml")
        write_json(make_splits(cfg, {n: load(cfg, n) for n in cfg["data"]["datasets"]}), p)
    return read_json(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3.yaml")
    cfg = load_config(ap.parse_args().config)
    download(cfg)
    frames = {n: load(cfg, n) for n in cfg["data"]["datasets"]}
    write_json(make_splits(cfg, frames), SPLITS)
    stats = {}
    for n, df in frames.items():
        stats[n] = {"rollouts": len(df), "questions": df["sample_id"].nunique(),
                    "retained_tokens": int(df["n_generated_text_tokens"].sum()),
                    "capped": int(df["hit_max_new_tokens"].sum()),
                    "null_extracted": int(df["extracted_answer"].isna().sum())}
    write_json(stats, "results/task3/data_stats.json")
    ensure_dir("results/task3")
    print(stats)


if __name__ == "__main__":
    main()
