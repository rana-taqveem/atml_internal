from __future__ import annotations

import argparse

import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from task5_feedback.score_perturbations import load_diagnostic_groups

POLICIES = ["sft", "rlvr", "rlaif"]


def clip(text, n=500):
    text = str(text).strip()
    return text if len(text) <= n else text[:n] + " [...]"


def quote(text, n=500):
    return "> " + clip(text, n).replace("\n", "\n> ")


def policy_table(base) -> pd.DataFrame:
    gsm, tr = load_json(base / "gsm" / "metrics.json"), load_json(base / "transfer" / "metrics.json")
    rows = []
    for p in POLICIES:
        g, t = gsm["policies"][p], tr["policies"][p]
        row = {"policy": p,
               "gsm_exact_acc": g["exact_accuracy"], "transfer_exact_acc": t["exact_accuracy"],
               "exact_acc_drop": g["exact_accuracy"] - t["exact_accuracy"],
               "gsm_format": g["format_compliance"], "transfer_format": t["format_compliance"],
               "gsm_tokens_mean": g["response_tokens"]["mean"], "gsm_tokens_std": g["response_tokens"]["std"],
               "transfer_tokens_mean": t["response_tokens"]["mean"], "transfer_tokens_std": t["response_tokens"]["std"],
               "gsm_truncation": g["truncation_rate"], "transfer_truncation": t["truncation_rate"]}
        for split, m in [("gsm", g), ("transfer", t)]:
            for ft, n in m["failure_types"].items():
                row[f"{split}_{ft}"] = n
            if "judge_counts_policy_win_tie_loss" in m:
                row[f"{split}_win_rate_vs_sft"] = m["ai_pairwise_win_rate_vs_sft"]
                w, ti, lo = m["judge_counts_policy_win_tie_loss"]
                row[f"{split}_judge_W_T_L"] = f"{w}/{ti}/{lo}"
                agr = m["verifier_judge_agreement"]
                row[f"{split}_judge_agrees_on_verifier_decisive"] = agr["judge_agrees_on_decisive"]
                row[f"{split}_n_verifier_decisive"] = agr["n_verifier_decisive"]
                row[f"{split}_judge_tie_on_verifier_ties"] = agr["judge_tie_rate_on_verifier_ties"]
                row[f"{split}_exact_label_agreement"] = agr["exact_label_agreement"]
            elif p == "sft":
                row[f"{split}_win_rate_vs_sft"] = 0.5  # self-comparison reference
        if "gsm_win_rate_vs_sft" in row:
            row["win_rate_drop"] = row["gsm_win_rate_vs_sft"] - row["transfer_win_rate_vs_sft"]
        rows.append(row)
    return pd.DataFrame(rows)


def diagnostic_table(base) -> pd.DataFrame:
    d = load_json(base / "diagnostics" / "diagnostic_metrics.json")
    rows = []
    for worse, e in d["by_perturbation"].items():
        for mech in ["verifier", "judge"]:
            r = e[mech]
            rows.append({"perturbation": worse, "kind": e["kind"], "mechanism": mech, "better_rate": r["better"],
                         "tie_rate": r["tie"], "wrong_rate": r["wrong"], "n_pairs": r["n"],
                         "judge_order_consistency": e["judge_order_consistency"] if mech == "judge" else None})
    return pd.DataFrame(rows)


def qualitative(cfg, base, k: int) -> str:
    lines = ["# Task 5 qualitative candidates\n", "Auto-generated shortlist; verify each before quoting.\n"]
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    pairs = read_jsonl(base / "diagnostics" / "pair_scores.jsonl")
    lines.append("## A. Controlled diagnostics where verifier and judge disagree\n")
    for kind_worse in ["corrupt_reasoning_correct_final", "persuasive_filler_correct", "gold_distractor_wrong_final",
                       "good_reasoning_wrong_final"]:
        dis = [p for p in pairs if p["worse"] == kind_worse and p["verifier"] != p["judge"]]
        lines.append(f"### clean_correct vs {kind_worse}: {len(dis)} disagreements")
        for p in dis[:k]:
            vs = groups[p["problem_id"]]
            lines += [f"- problem {p['problem_id']}: verifier={p['verifier']}, judge={p['judge']} "
                      f"(swapped order: {p['judge_swapped']})",
                      f"  - {kind_worse} response (tail):\n\n{quote(vs[kind_worse]['response'][-400:], 400)}\n"]
    for split in ["gsm", "transfer"]:
        lines.append(f"\n## B. {split}: judge preferred the verifier-wrong response (policy vs SFT)\n")
        gens = {p: {g["id"]: g for g in read_jsonl(base / split / f"generations_{p}.jsonl")} for p in POLICIES}
        for pol in ["rlvr", "rlaif"]:
            jr = [r for r in read_jsonl(base / split / f"judge_{pol}_vs_sft.jsonl")
                  if r["verifier"] != "TIE" and r["judge"] not in (r["verifier"], "TIE")]
            lines.append(f"### {pol} vs sft: {len(jr)} cases")
            for r in jr[:k]:
                a, b = gens[pol][r["id"]], gens["sft"][r["id"]]
                lines += [f"- `{r['id']}` gold {a['gold']}: {pol} pred {a['pred']} ({a['response_tokens']} tok), "
                          f"sft pred {b['pred']} ({b['response_tokens']} tok); judge preferred "
                          f"{pol if r['judge'] == 'A' else 'sft'}"]
    lines.append("\n## C. Transfer failures by type (first examples)\n")
    for pol in POLICIES:
        recs = read_jsonl(base / "transfer" / f"generations_{pol}.jsonl")
        for ft in ["wrong_final", "missing_final_format", "truncated_no_final"]:
            ex = [r for r in recs if r["failure_type"] == ft][:1]
            for r in ex:
                lines += [f"### {pol} / {ft}: `{r['id']}` gold {r['gold']} pred {r['pred']}", quote(r["response"][-500:]), ""]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    base = repo_path(cfg["results_dir"]) / "task5_feedback"
    needed = [base / "gsm" / "metrics.json", base / "transfer" / "metrics.json", base / "diagnostics" / "diagnostic_metrics.json"]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise SystemExit("Run evaluate_math (gsm, transfer) and score_perturbations first; missing: " + ", ".join(missing))

    pt, dt = policy_table(base), diagnostic_table(base)
    diag = load_json(base / "diagnostics" / "diagnostic_metrics.json")
    gsm = load_json(base / "gsm" / "metrics.json")
    cost = {"judge_seconds_per_call_diagnostics": diag["inference_cost"]["judge_seconds_per_new_call"],
            "judge_seconds_per_call_gsm": {p: gsm["policies"][p].get("judge_seconds_per_new_call") for p in ["rlvr", "rlaif"]},
            "verifier": "regex + float comparison, negligible",
            "rlaif_training_note": "direct RLAIF needs K(K-1)/2 judge calls per prompt group (6 for K=4); RLVR needs K verifier calls"}
    pt.to_csv(base / "policy_comparison.csv", index=False)
    dt.to_csv(base / "diagnostic_comparison.csv", index=False)
    save_json(base / "summary.json", {"policies": pt.to_dict(orient="records"), "diagnostics": dt.to_dict(orient="records"),
                                      "S_reason": diag["S_reason"], "S_outcome": diag["S_outcome"],
                                      "judge_round_robin_win_rate_by_variant": diag["judge_round_robin_win_rate_by_variant"],
                                      "inference_cost": cost})
    (base / "qualitative_candidates.md").write_text(qualitative(cfg, base, args.k), encoding="utf-8")
    with pd.option_context("display.max_columns", None, "display.width", 220, "display.precision", 3):
        print(pt.set_index("policy").T.to_string())
        print()
        print(dt.to_string(index=False))
    print(f"\nS_reason {diag['S_reason']}  S_outcome {diag['S_outcome']}")
    print(f"Saved {base}/policy_comparison.csv, diagnostic_comparison.csv, summary.json, qualitative_candidates.md")


if __name__ == "__main__":
    main()
