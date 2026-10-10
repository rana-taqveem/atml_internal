from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from common.models import clear_gpu
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}

# Controlled pairs: (diagnostically better, worse, what changes). The clean response is the anchor.
PAIRS = [
    ("clean_correct", "corrupt_reasoning_correct_final", "reasoning"),   # final held correct, reasoning degraded
    ("clean_correct", "good_reasoning_wrong_final", "outcome"),          # reasoning ~fixed, final changed
    ("clean_correct", "persuasive_filler_correct", "filler"),            # same answer + irrelevant persuasion
    ("clean_correct", "gold_distractor_wrong_final", "outcome"),         # gold number mentioned, wrong final
]


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    if len(rows) != 100 or len(by_problem) != 20 or any(len(v) != 5 for v in by_problem.values()):
        raise ValueError("Use the unchanged course diagnostic set: 20 problems x 5 unique variants")
    return by_problem


def outcome(pref: str) -> str:
    """Map a preference between (better=A, worse=B) to better / tie / wrong."""
    return {"A": "better", "TIE": "tie", "B": "wrong"}[pref]


def rates(labels) -> dict:
    n = len(labels)
    return {k: (sum(l == k for l in labels) / n if n else float("nan")) for k in ["better", "tie", "wrong"]} | {"n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))
    out_dir = repo_path(cfg["results_dir"]) / "task5_feedback" / "diagnostics"

    # Exact verifier, pointwise, with a check against the staff-validated expected reward.
    verifier = {}
    mismatches = []
    for pid, vs in groups.items():
        for v, row in vs.items():
            r = exact_reward(row["response"], str(row["gold_final"]))
            verifier[(pid, v)] = r
            if "expected_exact_reward" in row and float(row["expected_exact_reward"]) != r:
                mismatches.append({"problem_id": pid, "variant": v, "expected": row["expected_exact_reward"], "got": r})

    judge = PairwiseAIJudge(cfg, repo_path(cfg["results_dir"]) / "task5_feedback" / "judge_cache.json")
    pair_records, t_judge, n_calls = [], 0.0, 0

    def timed_compare(q, a, b):
        nonlocal t_judge, n_calls
        cached = judge._key(q, a, b) in judge.cache
        t0 = time.perf_counter()
        res = judge.compare(q, a, b)
        if not cached:
            t_judge += time.perf_counter() - t0
            n_calls += 1
        return res

    for pid, vs in groups.items():
        q = vs["clean_correct"]["question"]
        for better, worse, kind in PAIRS:
            a, b = vs[better]["response"], vs[worse]["response"]
            ra, rb = verifier[(pid, better)], verifier[(pid, worse)]
            v_pref = "TIE" if ra == rb else ("A" if ra > rb else "B")
            j_ab = timed_compare(q, a, b)                                       # primary call (course orientation)
            j_ba = {"A": "B", "B": "A", "TIE": "TIE"}[timed_compare(q, b, a)]   # same pair, candidates swapped
            pair_records.append({"problem_id": pid, "better": better, "worse": worse, "kind": kind,
                                 "verifier_better": ra, "verifier_worse": rb,
                                 "verifier": outcome(v_pref), "judge": outcome(j_ab), "judge_swapped": outcome(j_ba),
                                 "judge_order_consistent": j_ab == j_ba})

    # Round-robin over all five variants per problem: normalised pairwise win rate of each category.
    variants = sorted(EXPECTED_VARIANTS)
    rr = defaultdict(list)
    for pid, vs in groups.items():
        q = vs["clean_correct"]["question"]
        # Use the same timing wrapper for all calls, including the round robin.
        rewards = np.zeros(len(variants))
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                pref = timed_compare(q, vs[variants[i]]["response"], vs[variants[j]]["response"])
                rewards[i] += {"A": 1.0, "TIE": 0.5, "B": 0.0}[pref]
                rewards[j] += {"A": 0.0, "TIE": 0.5, "B": 1.0}[pref]
        rewards /= len(variants) - 1
        for v, r in zip(variants, rewards):
            rr[v].append(r)
    keys = {judge._key(vs["clean_correct"]["question"], vs[a]["response"], vs[b]["response"])
            for vs in groups.values() for a in variants for b in variants if a != b}
    parse_ambiguous = sum(judge.details.get(k, {}).get("parse_ambiguous") is True for k in keys if k in judge.cache)
    parse_unknown = sum(k not in judge.details for k in keys if k in judge.cache)
    del judge
    clear_gpu()
    write_jsonl(out_dir / "pair_scores.jsonl", pair_records)

    by_pair = {}
    for better, worse, kind in PAIRS:
        recs = [r for r in pair_records if r["worse"] == worse]
        by_pair[worse] = {"kind": kind, "verifier": rates([r["verifier"] for r in recs]),
                          "judge": rates([r["judge"] for r in recs]),
                          "judge_swapped_order": rates([r["judge_swapped"] for r in recs]),
                          "judge_order_consistency": float(np.mean([r["judge_order_consistent"] for r in recs]))}
    outcome_recs = [r for r in pair_records if r["kind"] == "outcome"]
    reason_recs = [r for r in pair_records if r["kind"] == "reasoning"]
    s = lambda recs, key: float(np.mean([r[key] == "better" for r in recs]))
    result = {
        "n_problems": len(groups),
        "pairs": [f"{b} > {w} ({k})" for b, w, k in PAIRS],
        "verifier_expected_reward_mismatches": mismatches,
        "judge_parse_ambiguous_count": parse_ambiguous,
        "judge_parse_status_unknown_count": parse_unknown,
        "by_perturbation": by_pair,
        "S_reason": {"verifier": s(reason_recs, "verifier"), "judge": s(reason_recs, "judge")},
        "S_outcome": {"verifier": s(outcome_recs, "verifier"), "judge": s(outcome_recs, "judge"),
                      "judge_good_reasoning_wrong_final_only": by_pair["good_reasoning_wrong_final"]["judge"]["better"],
                      "judge_gold_distractor_only": by_pair["gold_distractor_wrong_final"]["judge"]["better"]},
        "pointwise_verifier_reward_by_variant": {v: float(np.mean([verifier[(p, v)] for p in groups])) for v in variants},
        "judge_round_robin_win_rate_by_variant": {v: float(np.mean(rr[v])) for v in variants},
        "inference_cost": {"judge_seconds_per_new_call": (t_judge / n_calls) if n_calls else None,
                           "judge_new_calls": n_calls,
                           "verifier": "regex + float comparison (CPU, negligible)"},
    }
    save_json(out_dir / "diagnostic_metrics.json", result)

    print(f"\n{'perturbation (vs clean_correct)':<34}{'verifier b/t/w':>18}{'judge b/t/w':>18}{'order-consistent':>18}")
    for worse, e in by_pair.items():
        v, j = e["verifier"], e["judge"]
        print(f"{worse:<34}{v['better']:>6.2f}/{v['tie']:.2f}/{v['wrong']:.2f}{j['better']:>8.2f}/{j['tie']:.2f}/{j['wrong']:.2f}"
              f"{e['judge_order_consistency']:>16.2f}")
    print(f"S_reason  verifier={result['S_reason']['verifier']:.2f} judge={result['S_reason']['judge']:.2f}")
    print(f"S_outcome verifier={result['S_outcome']['verifier']:.2f} judge={result['S_outcome']['judge']:.2f}")
    print(f"verifier/expected mismatches: {len(mismatches)}")
    print(f"Saved {out_dir / 'diagnostic_metrics.json'}")


if __name__ == "__main__":
    main()
