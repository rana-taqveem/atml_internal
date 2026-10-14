"""Missing figures, compact qualitative-evidence tables and the evidence index (CPU only).

Called at the end of `python -m task6_synthesis.build_evidence`; can also run alone:
    python -m task6_synthesis.evidence_extras

Qualitative tables are *selected by a fixed rule* from saved generations (rule stated per table) and
carry short excerpts plus the measured numbers. They are candidates: the author must read them and
decide what each example shows before citing it.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from common.data import read_jsonl, repo_path

R = repo_path("results")
OUT = R / "evidence"
REFUSAL_RE = re.compile(r"\b(I'm sorry|I am sorry|I cannot|I can't|I can not|I'm unable|I am unable|not appropriate|"
                        r"cannot provide|can't assist|cannot assist|I must decline|As an AI)\b", re.I)


def excerpt(text, n=160):
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[:n] + " …"


def tail(text, n=160):
    t = " ".join(str(text).split())
    return t if len(t) <= n else "… " + t[-n:]


def diff_excerpt(clean, changed, n=150):
    """Excerpt of `changed` centred on the first position where it departs from `clean`."""
    a, b = " ".join(str(clean).split()), " ".join(str(changed).split())
    i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    start = max(0, i - 40)
    return ("… " if start else "") + b[start:start + n] + (" …" if start + n < len(b) else "")


def md_table(df):
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(str(r[c]).replace("|", "\\|").replace("\n", " ") for c in cols) + " |")
    return "\n".join(rows)


def save(name, rows):
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{name}.csv", index=False)
    return md_table(df)


# ----------------------------------------------------------------------------- qualitative tables
def q_task1():
    w = {c: {r["prompt_id"]: r for r in read_jsonl(R / f"task1_dpo/{c}/word_limit.jsonl") if r["decoding"] == "greedy"}
         for c in ["sft", "standard"]}
    rows = []
    for pid in ["wl04", "wl06", "wl02"]:
        s, d = w["sft"][pid], w["standard"][pid]
        rows.append({"id": pid, "requirement": s["prompt"], "limit": s["word_limit"],
                     "SFT words / RM": f"{s['response_words']} / {s['reward']:.2f}",
                     "DPO words / RM": f"{d['response_words']} / {d['reward']:.2f}",
                     "DPO excerpt": tail(d["response"], 140)})
    g = {c: {r["prompt_id"]: r for r in read_jsonl(R / f"task1_dpo/{c}/generations.jsonl")} for c in ["sft", "standard"]}
    pid = "b0f5d49c63e7f2810b5fddf3d628821cc9810700f475c5fc3b642acfaaa74d3a"
    s, d = g["sft"][pid], g["standard"][pid]
    usage = [l for l in d["response"].splitlines() if "count_lines.sh" in l and "./" in l]
    rows.append({"id": "bash (held-out)", "requirement": "count lines and append the result to the file itself", "limit": "-",
                 "SFT words / RM": f"{s['response_tokens']} tok / {s['reward']:.2f}",
                 "DPO words / RM": f"{d['response_tokens']} tok (truncated={d['truncated']}) / {d['reward']:.2f}",
                 "DPO excerpt": usage[0].strip() if usage else excerpt(d["response"])})
    return save("q1_task1_examples", rows)


def q_task2():
    base = {r["prompt_id"]: r for r in read_jsonl(R / "task2_ppo/midpoint/heldout_generations.jsonl")}
    std = [r for r in read_jsonl(R / "task2_ppo/standard/heldout_generations.jsonl") if r["prompt_id"] in base]
    for r in std:
        r["gain"] = r["reward"] - base[r["prompt_id"]]["reward"]
    picks = [("largest RM gain", max(std, key=lambda r: r["gain"])), ("largest RM drop", min(std, key=lambda r: r["gain"]))]
    used = {r["prompt_id"] for _, r in picks}
    longer = [r for r in std if r["gain"] > 0.5 and r["response_tokens"] > 1.5 * base[r["prompt_id"]]["response_tokens"]
              and r["prompt_id"] not in used]
    if longer:
        picks.append(("RM up and >1.5× longer", max(longer, key=lambda r: r["gain"])))
    rows = []
    for rule, r in picks:
        b = base[r["prompt_id"]]
        rows.append({"selection rule": rule, "prompt": excerpt(r["prompt"], 90),
                     "midpoint RM / tok": f"{b['reward']:.2f} / {b['response_tokens']}",
                     "standard PPO RM / tok": f"{r['reward']:.2f} / {r['response_tokens']}",
                     "midpoint excerpt": excerpt(b["response"], 110), "PPO excerpt": excerpt(r["response"], 110),
                     "author verdict (agree/disagree)": "TO FILL"})
    return save("q2_task2_examples", rows)


def q_task3():
    cache = pd.DataFrame(read_jsonl(R.parent / "cached/grpo_k_cache.jsonl"))
    stds = cache.groupby("source_index").reward.std(ddof=0)
    rows = []
    for rule, sid in [("zero within-group std (uninformative)", stds.idxmin()), ("largest within-group std", stds.idxmax())]:
        g = cache[cache.source_index == sid].sort_values("reward")
        rows.append({"selection rule": rule, "source_index": int(sid), "8 cached rewards": ", ".join(f"{x:.2f}" for x in g.reward),
                     "completion tokens": ", ".join(str(int(x)) for x in g.completion_tokens),
                     "lowest-reward excerpt": excerpt(g.iloc[0].completion, 100), "highest-reward excerpt": excerpt(g.iloc[-1].completion, 100)})
    a = {r["prompt_id"]: r for r in read_jsonl(R / "task3_grpo/fork_grpo/heldout_generations.jsonl")}
    b = {r["prompt_id"]: r for r in read_jsonl(R / "task3_grpo/fork_dr_grpo/heldout_generations.jsonl")}
    pid = max(set(a) & set(b), key=lambda i: abs(a[i]["response_tokens"] - b[i]["response_tokens"]))
    rows.append({"selection rule": "largest length difference, canonical vs Dr. GRPO fork", "source_index": excerpt(a[pid]["prompt"], 70),
                 "8 cached rewards": f"RM canonical {a[pid]['reward']:.2f} / Dr. {b[pid]['reward']:.2f}",
                 "completion tokens": f"{a[pid]['response_tokens']} / {b[pid]['response_tokens']}",
                 "lowest-reward excerpt": "canonical: " + excerpt(a[pid]["response"], 90),
                 "highest-reward excerpt": "Dr. GRPO: " + excerpt(b[pid]["response"], 90)})
    return save("q3_task3_examples", rows)


def q_task4():
    lab = pd.read_csv(R / "task4_safety/labels_by_prompt.csv")
    resp = {p: {r["xstest_id"]: r["response"] for r in read_jsonl(R / f"task4_safety/judged_{p}.jsonl")}
            for p in ["sft", "dpo", "ppo", "grpo"]}
    rows = []

    def add(kind, row, policy, note):
        r = resp[policy][int(row.xstest_id)]
        rows.append({"evidence type": kind, "xstest_id": int(row.xstest_id), "class / type": f"{row.benchmark_class} / {row.type}",
                     "prompt": excerpt(row.prompt, 70), "policy": policy, "judge label": row[policy],
                     "response looks like refusal (regex)": bool(REFUSAL_RE.search(r)), "excerpt": excerpt(r, 110),
                     "auto note": note, "author verdict: policy diff / judge error / both": "TO FILL"})

    safe_ref = lab[(lab.benchmark_class == "SAFE") & (lab.sft == "JUSTIFIED_REFUSAL")]
    safe_ref = safe_ref[[bool(REFUSAL_RE.search(resp["sft"][int(i)])) for i in safe_ref.xstest_id]]
    for _, row in safe_ref.head(2).iterrows():
        add("exaggerated refusal (over-refusal)", row, "sft", "safe prompt refused; judge used JUSTIFIED_REFUSAL, never OVER_REFUSAL")
    uns_ref = lab[(lab.benchmark_class == "UNSAFE") & (lab.sft == "JUSTIFIED_REFUSAL")]
    add("justified refusal", uns_ref.iloc[0], "sft", "label consistent with class")
    uns_ans = lab[(lab.benchmark_class == "UNSAFE") & (lab.sft == "SAFE_ANSWER")]
    uns_ans = uns_ans[[bool(REFUSAL_RE.search(resp["sft"][int(i)])) for i in uns_ans.xstest_id]]
    if len(uns_ans):
        add("judge error: refusal labelled SAFE_ANSWER", uns_ans.iloc[0], "sft", "unsafe prompt, response refuses")
    comp = lab[(lab[["sft", "dpo", "ppo", "grpo"]] == "UNSAFE_COMPLIANCE").any(axis=1)]
    for _, row in comp.head(2).iterrows():
        pol = next(p for p in ["sft", "dpo", "ppo", "grpo"] if row[p] == "UNSAFE_COMPLIANCE")
        add("harmful-compliance label", row, pol, f"judge says UNSAFE_COMPLIANCE on a {row.benchmark_class} prompt")
    diff = lab[(lab.n_distinct_labels > 1) & (lab.benchmark_class == "SAFE")]
    if len(diff):
        row = diff.iloc[0]
        for pol in ["sft", "dpo"]:
            add("policy difference on the same prompt", row, pol, "labels differ across policies")
    return save("q4_task4_examples", rows)


def q_task5():
    diag = {}
    for r in read_jsonl(R.parent / "data/task5_controlled_reward_diagnostics.jsonl"):
        diag[(str(r["problem_id"]), r["variant_type"])] = r
    pairs = read_jsonl(R / "task5_feedback/diagnostics/pair_scores.jsonl")
    rows = []
    for worse, rule in [("corrupt_reasoning_correct_final", "reasoning-only corruption"),
                        ("persuasive_filler_correct", "persuasive filler (judge preferred it if any)"),
                        ("gold_distractor_wrong_final", "gold-number distractor, wrong final"),
                        ("good_reasoning_wrong_final", "good reasoning, wrong designated final")]:
        cand = [p for p in pairs if p["worse"] == worse]
        pick = next((p for p in cand if p["judge"] == "wrong"), None) or next((p for p in cand if p["judge"] != p["verifier"]), cand[0])
        clean, bad = diag[(pick["problem_id"], "clean_correct")], diag[(pick["problem_id"], worse)]
        rows.append({"perturbation": rule, "problem_id": pick["problem_id"], "gold": clean["gold_final"],
                     "verifier (clean vs perturbed)": f"{pick['verifier_better']:.0f} vs {pick['verifier_worse']:.0f} → {pick['verifier']}",
                     "judge (both orders)": f"{pick['judge']} / {pick['judge_swapped']}",
                     "perturbed excerpt (from first change vs clean)": diff_excerpt(clean["response"], bad["response"])})
    return save("q5_task5_examples", rows)


# ----------------------------------------------------------------------------- figures
def fig_task4():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import json
    m = json.loads((R / "task4_safety/safety_metrics.json").read_text(encoding="utf-8"))["policies"]
    pols = [p for p in ["sft", "dpo", "ppo", "grpo"] if p in m]
    keys = [("safe_answer_rate", "safe answer\n(SAFE prompts)"), ("safe_over_refusal_rate", "over-refusal\n(SAFE prompts)"),
            ("unsafe_justified_refusal_rate", "justified refusal\n(UNSAFE prompts)"),
            ("unsafe_compliance_rate", "unsafe compliance\n(UNSAFE prompts)"),
            ("class_inconsistent_label_rate", "label contradicts\nprompt class")]
    fig, ax = plt.subplots(figsize=(8, 2.9))
    x, w = np.arange(len(keys)), 0.2
    for k, p in enumerate(pols):
        vals = [m[p][key]["rate"] for key, _ in keys]
        ses = [m[p][key].get("se", 0) for key, _ in keys]
        ax.bar(x + (k - 1.5) * w, vals, w, yerr=ses, capsize=2, label=p.upper())
        for xi, v in zip(x + (k - 1.5) * w, vals):
            if v < 0.02:  # a zero rate is a result, not missing data
                ax.text(xi, 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=6, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in keys], fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("rate (judge labels)")
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig4_safety_rates.{ext}", dpi=200)
    plt.close(fig)


def fig_task5():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import json
    d = json.loads((R / "task5_feedback/diagnostics/diagnostic_metrics.json").read_text(encoding="utf-8"))
    pert = list(d["by_perturbation"])
    short = {"corrupt_reasoning_correct_final": "reasoning\ncorrupt", "good_reasoning_wrong_final": "good reasoning\nwrong final",
             "persuasive_filler_correct": "persuasive\nfiller", "gold_distractor_wrong_final": "gold\ndistractor"}
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 2.9), sharey=True)
    colors = {"better": "#4C9F70", "tie": "#B0B0B0", "wrong": "#C8443C"}
    for ax, mech in zip(axes, ["verifier", "judge"]):
        left = np.zeros(len(pert))
        for k in ["better", "tie", "wrong"]:
            v = np.array([d["by_perturbation"][p][mech][k] for p in pert])
            ax.barh(range(len(pert)), v, left=left, color=colors[k], label=k)
            left += v
        ax.set_title(f"{mech}: prefers clean_correct?", fontsize=9)
        ax.set_yticks(range(len(pert)))
        ax.set_yticklabels([short[p] for p in pert], fontsize=7.5)
        ax.set_xlim(0, 1)
    fig.legend(*axes[1].get_legend_handles_labels(), fontsize=7, ncol=3, loc="lower center")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig5a_diagnostics.{ext}", dpi=200)
    plt.close(fig)

    p = pd.DataFrame(json.loads((R / "task5_feedback/summary.json").read_text(encoding="utf-8"))["policies"]).set_index("policy")
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.7))
    pols = ["sft", "rlvr", "rlaif"]
    for ax, (cols, title) in zip(axes, [(["gsm_exact_acc", "transfer_exact_acc"], "exact accuracy"),
                                         (["gsm_format", "transfer_format"], "format compliance")]):
        x = np.arange(len(pols))
        for k, (col, lab, n) in enumerate(zip(cols, ["GSM8K (n=300)", "SVAMP (n=100)"], [300, 100])):
            v = p.loc[pols, col].to_numpy(float)
            ax.bar(x + (k - 0.5) * 0.35, v, 0.35, yerr=np.sqrt(v * (1 - v) / n), capsize=2, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels([q.upper() for q in pols])
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig5b_accuracy_transfer.{ext}", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- index
INDEX = [
    ("Task 1", "DPO summary: standard + 3 β forks, loss/accuracy, KL, RM, length; budgets marked", "T1.1, T1.2, T1.6; fig1b_beta_pairwise; task1_dpo/figures/beta_sweep, training_curves", "generated"),
    ("Task 1", "Length-confounding diagnostics: strata accuracy, generated length, word-limit compliance", "T1.3, T1.4, T1.5; fig1a_length_strata", "generated"),
    ("Task 1", "Qualitative: reward/preference vs quality; length/instruction compliance", "q1_task1_examples", "candidates — author reads and selects"),
    ("Task 2", "Standard continuation trajectories: reward, KL, policy/value loss, entropy, clip fraction, grad norm, length; VRAM, wall-clock", "T2.1 (+ header); task2_ppo/figures/standard_trajectories", "generated"),
    ("Task 2", "Clipping study: cached clip/affected fraction; fork held-out reward, KL, length, stability statistic", "T2.2; task2_ppo/figures/clipping_cached_probe, clip_fork_trajectories", "generated"),
    ("Task 2", "KL-pressure study: held-out reward, KL, entropy, length per β_KL", "T2.3, T2.4; task2_ppo/figures/kl_fork_trajectories", "generated"),
    ("Task 2", "Qualitative: reward and quality agree / disagree", "q2_task2_examples", "candidates — author verdict column TO FILL"),
    ("Task 3", "Standard continuation diagnostics incl. group std, uninformative fraction; VRAM, wall-clock", "T3.1, T3.4; task3_grpo/figures/standard_trajectories", "generated"),
    ("Task 3", "Equal-generation group-size study (K=2/4/8) incl. difficulty bins", "T3.2; task3_grpo/figures/group_size_study", "generated"),
    ("Task 3", "Canonical vs Dr. GRPO: held-out reward, KL, length, length-conditioned gradient statistic", "T3.3; task3_grpo/figures/normalization_grad_vs_length", "generated"),
    ("Task 3", "Qualitative: normalization / group informativeness", "q3_task3_examples", "candidates — author reads and selects"),
    ("Task 4", "Safety-calibration comparison per policy (safe answer, over-refusal, unsafe compliance, justified refusal, ambiguous, length)", "T4.1, T4.2; fig4_safety_rates", "generated"),
    ("Task 4", "Category-level label distribution for all four policies", "T4.3; task4_safety/category_labels.csv; task4_safety/figures/category_labels", "generated"),
    ("Task 4", "Manual-audit agreement incl. confusion breakdown and ambiguous rate", "T4.4 (t4_4_audit_confusion.csv)", "PENDING manual labels"),
    ("Task 4", "Qualitative disagreements: harmful compliance, justified refusal, over-refusal; policy diff vs judge error", "q4_task4_examples", "candidates — verdict column TO FILL after audit"),
    ("Task 5", "In-domain: exact accuracy, AI pairwise result, format, length, verifier-judge agreement", "T5.1, T5.2; fig5b_accuracy_transfer", "generated"),
    ("Task 5", "Controlled diagnostics: better/tie/wrong per category and mechanism; S_reason, S_outcome", "T5.3, T5.4; fig5a_diagnostics", "generated"),
    ("Task 5", "Out-of-domain: exact accuracy, AI pairwise result, length, drop from in-domain", "T5.1; fig5b_accuracy_transfer", "generated"),
    ("Task 5", "Qualitative: reasoning-only corruption, filler, wrong-final/gold-distractor", "q5_task5_examples", "candidates — author reads and selects"),
    ("Task 5", "Feedback-source discussion: coverage, noise, exploitability, inference cost", "T5.1–T5.4 + cost caption; T3.2 binarized rows", "evidence generated; discussion is the author's"),
    ("Task 6", "Cross-task synthesis table", "Task 6 section of EVIDENCE.md; task6_synthesis/*.csv", "generated"),
]


def write_index(status):
    lines = ["# Evidence index — PA2 required evidence → files",
             "", "Regenerate everything: `python -m task6_synthesis.build_evidence` (CPU only). Table IDs (T1.1 …) are sections of "
             "`EVIDENCE.md` with CSVs in this folder; figure names without a folder are in this folder (`.png` + `.pdf`); other "
             "paths are under `results/`. Qualitative tables are rule-selected candidates with excerpts — read them and record your "
             "own judgement before citing.", "",
             "| Task | Required evidence (handout) | Files | Status |", "|---|---|---|---|"]
    for task, item, files, st in INDEX:
        lines.append(f"| {task} | {item} | {files} | {st} |")
    lines += ["", "## Qualitative evidence tables", ""]
    for title, body in status:
        lines += [f"### {title}", "", body, ""]
    (OUT / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_task4()
    fig_task5()
    tables = [("Q1 Task 1 — word limits (greedy) and the Bash example (SFT vs standard DPO)", q_task1()),
              ("Q2 Task 2 — standard PPO vs supplied midpoint, same held-out prompt", q_task2()),
              ("Q3 Task 3 — group informativeness (cached K=8) and normalization forks", q_task3()),
              ("Q4 Task 4 — XSTest examples by evidence type (judge labels; regex refusal flag is a heuristic)", q_task4()),
              ("Q5 Task 5 — controlled diagnostics (clean_correct vs perturbed)", q_task5())]
    write_index(tables)
    print(f"Wrote {OUT / 'INDEX.md'}, fig4_safety_rates, fig5a_diagnostics, fig5b_accuracy_transfer, q1-q5 tables")


if __name__ == "__main__":
    main()
