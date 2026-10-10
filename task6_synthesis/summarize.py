"""Task 6: one cross-task table from the saved Task 1-5 results (no training, CPU only).

Every number is copied from a result file written by an earlier task; missing tasks are skipped.
Reward-model scores are only comparable *within* a task's protocol (DPO: 256-token cap on the DPO
held-out prompts; PPO/GRPO: 768-token cap on the RL eval pool) and never with Task 5 rewards.
"""
from __future__ import annotations

import argparse

import pandas as pd

from common.data import repo_path
from common.logging_utils import load_json, save_json

TRAINING_SIGNAL = {
    "sft": "none (starting policy)",
    "dpo": "offline AI preference pairs (UltraFeedback), implicit reward",
    "ppo": "online samples, learned reward model + critic, KL-shaped reward",
    "grpo": "online samples, learned reward model, group-relative baseline (no critic)",
    "rlvr": "online samples, exact final-answer verifier (binary)",
    "rlaif": "online samples, pairwise AI-judge win rate",
}


def maybe(path):
    p = repo_path(path)
    if not p.exists():
        return None
    return pd.read_csv(p) if p.suffix == ".csv" else load_json(p)


def row_of(df, condition):
    if df is None:
        return {}
    hit = df[df["condition"] == condition]
    return hit.iloc[0].to_dict() if len(hit) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/task6_synthesis")
    args = ap.parse_args()
    out = repo_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t1 = maybe("results/task1_dpo/summary.csv")
    t2 = maybe("results/task2_ppo/summary.csv")
    t3 = maybe("results/task3_grpo/summary.csv")
    t4 = maybe("results/task4_safety/safety_metrics.json")
    t5 = maybe("results/task5_feedback/summary.json")
    tr = {k: maybe(f"results/{d}/standard/train_summary.json") for k, d in
          [("dpo", "task1_dpo"), ("ppo", "task2_ppo"), ("grpo", "task3_grpo")]}
    print("found:", {k: v is not None for k, v in
                     {"task1": t1, "task2": t2, "task3": t3, "task4": t4, "task5": t5}.items()})

    rows = []
    # Tasks 1-3: drift / reward / length / compute for the standard policies.
    for policy, df, proto in [("dpo", t1, "Task1 protocol: 128 DPO eval prompts, 256-token cap"),
                              ("ppo", t2, "Task2/3 protocol: 64 RL eval prompts, 768-token cap"),
                              ("grpo", t3, "Task2/3 protocol: 64 RL eval prompts, 768-token cap")]:
        r = row_of(df, "standard")
        if not r:
            continue
        s = tr[policy] or {}
        rows.append({
            "policy": policy, "training_signal": TRAINING_SIGNAL[policy], "eval_protocol": proto,
            "rm_reward": r.get("rm_reward_mean", r.get("heldout_reward")),
            "rm_reward_se": r.get("rm_reward_se", r.get("heldout_reward_se")),
            "rm_reward_vs_start": r.get("rm_reward_diff_vs_sft", r.get("reward_diff_vs_midpoint")),
            "kl_token": r.get("kl_token_mean", r.get("heldout_kl_token")),
            "entropy_token": r.get("entropy_token_mean", r.get("heldout_entropy")),
            "tokens_mean": r.get("gen_tokens_mean", r.get("heldout_tokens_mean")),
            "truncation_rate": r.get("truncation_rate", r.get("heldout_truncation")),
            "train_wall_clock_s": s.get("wall_clock_s"), "train_peak_vram_gib": s.get("peak_vram_gib"),
            "train_updates": s.get("updates"), "train_tokens_generated": s.get("tokens_generated"),
        })
    # Baselines on the same two protocols, so the "vs start" comparisons have a reference row.
    for label, df, cond in [("sft (Task1 protocol)", t1, "sft"), ("ppo midpoint", t2, "midpoint"),
                            ("grpo midpoint", t3, "midpoint"), ("sft (Task2 protocol)", t2, "sft")]:
        r = row_of(df, cond)
        if r:
            rows.append({"policy": label, "training_signal": TRAINING_SIGNAL["sft"] if "sft" in label else "supplied checkpoint",
                         "rm_reward": r.get("rm_reward_mean", r.get("heldout_reward")),
                         "rm_reward_se": r.get("rm_reward_se", r.get("heldout_reward_se")),
                         "kl_token": r.get("kl_token_mean", r.get("heldout_kl_token")),
                         "entropy_token": r.get("entropy_token_mean", r.get("heldout_entropy")),
                         "tokens_mean": r.get("gen_tokens_mean", r.get("heldout_tokens_mean")),
                         "truncation_rate": r.get("truncation_rate", r.get("heldout_truncation"))})
    drift = pd.DataFrame(rows)

    # Task 4: safety calibration of the four fixed policies.
    safety = None
    if t4:
        safety = pd.DataFrame([{
            "policy": p, "safe_answer": m["safe_answer_rate"]["rate"], "over_refusal": m["safe_over_refusal_rate"]["rate"],
            "unsafe_compliance": m["unsafe_compliance_rate"]["rate"],
            "justified_refusal": m["unsafe_justified_refusal_rate"]["rate"], "ambiguous": m["ambiguous_rate"]["rate"],
            "tokens_mean": m["response_tokens_mean"]} for p, m in t4["policies"].items()])
    # Task 5: reward-source comparison.
    feedback = pd.DataFrame(t5["policies"]) if t5 else None

    drift.to_csv(out / "drift_reward_compute.csv", index=False)
    if safety is not None:
        safety.to_csv(out / "safety.csv", index=False)
    if feedback is not None:
        feedback.to_csv(out / "feedback_source.csv", index=False)
    save_json(out / "cross_task_summary.json", {
        "note": "RM scores comparable only within one protocol; Task 5 rewards are on a different scale.",
        "training_signal": TRAINING_SIGNAL,
        "drift_reward_compute": drift.to_dict(orient="records"),
        "safety": None if safety is None else safety.to_dict(orient="records"),
        "feedback_source": None if feedback is None else feedback.to_dict(orient="records"),
        "task5_S_reason": t5.get("S_reason") if t5 else None, "task5_S_outcome": t5.get("S_outcome") if t5 else None,
        "task5_inference_cost": t5.get("inference_cost") if t5 else None,
    })
    with pd.option_context("display.max_columns", None, "display.width", 220, "display.precision", 4):
        print("\nDrift / reward / compute:\n", drift.drop(columns=["training_signal", "eval_protocol"], errors="ignore").to_string(index=False))
        if safety is not None:
            print("\nSafety (Task 4):\n", safety.to_string(index=False))
        if feedback is not None:
            print("\nFeedback source (Task 5):\n", feedback.to_string(index=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
