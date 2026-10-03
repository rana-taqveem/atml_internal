from __future__ import annotations

import argparse

import numpy as np

from common.data import load_yaml, preference_responses, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.metrics import safe_corr
from common.models import load_tokenizer
from task1_dpo.evaluate import evaluate_policy
from task1_dpo.train import run_training

STRATA = ["preferred_longer", "length_matched", "rejected_longer"]


def dataset_length_profile(tokenizer, rows) -> dict:
    """Length structure of a preference file itself (a property of the data, not of any policy)."""
    lc, lr = [], []
    for row in rows:
        yc, yr = preference_responses(row)
        lc.append(len(tokenizer(yc, add_special_tokens=False)["input_ids"]))
        lr.append(len(tokenizer(yr, add_special_tokens=False)["input_ids"]))
    lc, lr = np.asarray(lc, float), np.asarray(lr, float)
    diff = lc - lr
    out = {
        "n_pairs": len(rows),
        "chosen_tokens_mean": float(lc.mean()),
        "rejected_tokens_mean": float(lr.mean()),
        "length_diff_mean": float(diff.mean()),
        "length_diff_median": float(np.median(diff)),
        "frac_chosen_longer": float((diff > 0).mean()),
        "frac_rejected_longer": float((diff < 0).mean()),
        "frac_equal_length": float((diff == 0).mean()),
        # Accuracy of the trivial rule "the longer response is preferred" (ties count 0.5).
        "longer_is_better_accuracy": float(((diff > 0) + 0.5 * (diff == 0)).mean()),
    }
    if "length_stratum" in rows[0]:
        out["stratum_counts"] = {s: int(sum(r["length_stratum"] == s for r in rows)) for s in STRATA}
    return out


def margin_length_correlation(name: str, results_dir) -> dict:
    """Does the learned margin track the token-length difference of the pair? (per held-out split)"""
    out = {}
    for split in ["standard", "length_stratified"]:
        path = results_dir / name / f"eval_pairs_{split}.jsonl"
        if path.exists():
            recs = read_jsonl(path)
            out[split] = safe_corr([r["margin"] for r in recs],
                                   [r["chosen_tokens"] - r["rejected_tokens"] for r in recs])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--skip-existing", action="store_true", help="reuse adapters/metrics that already exist")
    ap.add_argument("--standard-name", default="standard")
    ap.add_argument("--length-name", default="length_balanced")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"])

    balanced = read_jsonl(cfg["paths"]["dpo_length_train"])
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])
    print("Length-balanced train rows:", len(balanced))
    print("Length-stratified eval rows:", len(stratified))

    # 1) Length-balanced condition: same recipe and pair budget as standard DPO, different pairs.
    out = cfg["length_output"]
    if args.skip_existing and (repo_path(out) / "adapter_config.json").exists():
        print(f"[{args.length_name}] reusing existing adapter at {out}")
    else:
        run_training(args.config, args.length_name, dataset_path=cfg["paths"]["dpo_length_train"], output_path=out)

    # 2) Evaluate both under the common protocol (per-stratum pairs + generation + word limits).
    conditions = {args.standard_name: cfg["standard_output"], args.length_name: out}
    for name, adapter in conditions.items():
        if not (repo_path(adapter) / "adapter_config.json").exists():
            raise FileNotFoundError(f"{adapter} missing: train the {name!r} condition first (task1_dpo.train)")
        if not (args.skip_existing and (results_dir / name / "eval_metrics.json").exists()):
            evaluate_policy(args.config, adapter, name)

    # 3) Compile the comparison, plus the length structure of the preference files themselves.
    tokenizer = load_tokenizer(cfg["base_model"])
    report = {
        "dataset_length_profile": {
            "standard_train": dataset_length_profile(tokenizer, read_jsonl(cfg["paths"]["dpo_standard_train"])),
            "length_balanced_train": dataset_length_profile(tokenizer, balanced),
            "standard_eval": dataset_length_profile(tokenizer, read_jsonl(cfg["paths"]["dpo_standard_eval"])),
            "length_stratified_eval": dataset_length_profile(tokenizer, stratified),
        },
        "conditions": {},
    }
    names = list(conditions) + (["sft"] if (results_dir / "sft" / "eval_metrics.json").exists() else [])
    for name in names:
        m = load_json(results_dir / name / "eval_metrics.json")
        strat = m["pairs_length_stratified"]
        entry = {
            "stratified_accuracy_overall": strat["preference_accuracy"],
            "stratified_accuracy": {s: strat["by_stratum"][s]["preference_accuracy"] for s in STRATA},
            "stratified_margin_mean": {s: strat["by_stratum"][s]["margin_mean"] for s in STRATA},
            "standard_eval_accuracy": m["pairs_standard"]["preference_accuracy"],
            "margin_vs_length_diff_corr": margin_length_correlation(name, results_dir),
        }
        if "generation" in m:
            entry["generated_tokens"] = m["generation"]["response_tokens"]
            entry["reward_mean"] = m["generation"]["reward_mean"]
            entry["kl_token_mean"] = m["generation"]["kl_token_mean"]
            entry["word_limit"] = {mode: {"compliance_rate": v["compliance_rate"],
                                          "words_mean": v["words"]["mean"],
                                          "words_to_limit_ratio_mean": v["words_to_limit_ratio_mean"]}
                                   for mode, v in m["word_limit"].items()}
        report["conditions"][name] = entry

    save_json(results_dir / "length_analysis.json", report)
    print("\nPer-stratum held-out preference accuracy:")
    print(f"{'condition':<18}" + "".join(f"{s:>18}" for s in STRATA) + f"{'gen tokens':>12}{'WL greedy':>11}")
    for name, e in report["conditions"].items():
        gen = e.get("generated_tokens", {}).get("mean", float("nan"))
        wl = e.get("word_limit", {}).get("greedy", {}).get("compliance_rate", float("nan"))
        print(f"{name:<18}" + "".join(f"{e['stratified_accuracy'][s]:>18.3f}" for s in STRATA) + f"{gen:>12.1f}{wl:>11.2f}")
    print(f"\nSaved {results_dir / 'length_analysis.json'}")


if __name__ == "__main__":
    main()
