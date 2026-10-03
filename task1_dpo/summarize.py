"""Collect all Task 1 result files into tables, figures, and qualitative-example candidates.

Reads only saved results (no GPU). Run after train/evaluate/ablate_beta/analyze_length.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.metrics import safe_corr

STRATA = ["preferred_longer", "length_matched", "rejected_longer"]
N_BOOT = 5000


def binom_se(p: float, n: int) -> float:
    return float(np.sqrt(p * (1 - p) / max(n, 1)))


def bootstrap_ci(values, rng, n_boot: int = N_BOOT):
    """Percentile 95% CI of the mean of `values` (resampling the evaluation items)."""
    v = np.asarray(values, float)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def condition_names(results_dir, betas):
    names = ["sft", "standard"] + [f"beta_{b:g}" for b in betas] + ["length_balanced"]
    return [n for n in names if (results_dir / n / "eval_metrics.json").exists()]


def summary_table(results_dir, names) -> pd.DataFrame:
    rows = []
    for name in names:
        m = load_json(results_dir / name / "eval_metrics.json")
        ts_path = results_dir / name / "train_summary.json"
        ts = load_json(ts_path) if ts_path.exists() else {}
        ps, strat = m["pairs_standard"], m["pairs_length_stratified"]
        row = {
            "condition": name,
            "train_pairs": ts.get("n_pairs", 0),
            "updates": ts.get("updates", 0),
            "train_beta": ts.get("beta"),
            "heldout_dpo_loss": ps["dpo_loss"],
            "heldout_pref_acc": ps["preference_accuracy"],
            "heldout_pref_acc_se": binom_se(ps["preference_accuracy"], ps["n_pairs"]),
            "heldout_margin_mean": ps["margin_mean"],
            "chosen_logratio_mean": ps["chosen_logratio_mean"],
            "rejected_logratio_mean": ps["rejected_logratio_mean"],
        }
        for s in STRATA:
            b = strat["by_stratum"][s]
            row[f"acc_{s}"] = b["preference_accuracy"]
            row[f"acc_{s}_se"] = binom_se(b["preference_accuracy"], b["n_pairs"])
        if "generation" in m:
            g = m["generation"]
            row.update({
                "kl_token_mean": g["kl_token_mean"], "kl_seq_mean": g["kl_seq_mean"],
                "entropy_token_mean": g["entropy_token_mean"],
                "rm_reward_mean": g["reward_mean"], "rm_reward_std": g["reward_std"],
                "rm_reward_se": g["reward_std"] / np.sqrt(g["n_prompts"]),
                "gen_tokens_mean": g["response_tokens"]["mean"], "gen_tokens_std": g["response_tokens"]["std"],
                "gen_tokens_median": g["response_tokens"]["median"], "gen_tokens_iqr": g["response_tokens"]["iqr"],
                "truncation_rate": g["truncation_rate"],
                "truncation_rate_se": binom_se(g["truncation_rate"], g["n_prompts"]),
                "wordlimit_greedy_compliance": m["word_limit"]["greedy"]["compliance_rate"],
                "wordlimit_sampled_compliance": m["word_limit"].get("sampled", {}).get("compliance_rate"),
                "wordlimit_greedy_words_mean": m["word_limit"]["greedy"]["words"]["mean"],
            })
            gens = read_jsonl(results_dir / name / "generations.jsonl")
            row["corr_reward_vs_length"] = safe_corr([r["reward"] for r in gens], [r["response_tokens"] for r in gens])
            sft_gens = results_dir / "sft" / "generations.jsonl"
            if name != "sft" and sft_gens.exists():
                # Paired on prompt: each condition and SFT answer the same held-out prompts.
                base = {r["prompt_id"]: r for r in read_jsonl(sft_gens)}
                rng = np.random.default_rng(0)
                for key, col in [("reward", "rm_reward"), ("response_tokens", "gen_tokens")]:
                    d = [r[key] - base[r["prompt_id"]][key] for r in gens if r["prompt_id"] in base]
                    lo, hi = bootstrap_ci(d, rng)
                    row.update({f"{col}_diff_vs_sft": float(np.mean(d)), f"{col}_diff_ci_lo": lo,
                                f"{col}_diff_ci_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def length_mechanism(results_dir, names) -> dict:
    """Is the per-stratum asymmetry produced by summing log-ratios over tokens?

    For every trained condition: correlation/slope of each response's log-ratio
    (log pi - log ref) against its token count, and preference accuracy when the
    margin uses per-token (length-normalised) log-ratios instead of sums. Plus paired
    bootstrap CIs for length-balanced vs standard, and the within-model gap between
    the rejected-longer and preferred-longer strata.
    """
    rng = np.random.default_rng(0)
    out = {"conditions": {}}
    for name in [n for n in names if n != "sft"]:
        out["conditions"][name] = {}
        for split in ["standard", "length_stratified"]:
            pairs = read_jsonl(results_dir / name / f"eval_pairs_{split}.jsonl")
            lr = np.array([p["policy_chosen_logp"] - p["ref_chosen_logp"] for p in pairs]
                          + [p["policy_rejected_logp"] - p["ref_rejected_logp"] for p in pairs])
            tok = np.array([p["chosen_tokens"] for p in pairs] + [p["rejected_tokens"] for p in pairs], float)
            norm = np.array([(p["policy_chosen_logp"] - p["ref_chosen_logp"]) / p["chosen_tokens"]
                             - (p["policy_rejected_logp"] - p["ref_rejected_logp"]) / p["rejected_tokens"]
                             for p in pairs])
            entry = {
                "corr_logratio_vs_tokens": safe_corr(lr, tok),
                "slope_logratio_per_100_tokens": float(np.polyfit(tok, lr, 1)[0] * 100),
                "accuracy_summed_margin": float(np.mean([p["correct"] for p in pairs])),
                "accuracy_per_token_margin": float(np.mean(norm > 0)),
            }
            if split == "length_stratified":
                entry["by_stratum"] = {s: {
                    "accuracy_summed_margin": float(np.mean([p["correct"] for p in pairs if p["length_stratum"] == s])),
                    "accuracy_per_token_margin": float(np.mean([n > 0 for n, p in zip(norm, pairs)
                                                                if p["length_stratum"] == s])),
                } for s in STRATA}
                acc = {s: np.array([p["correct"] for p in pairs if p["length_stratum"] == s], float) for s in STRATA}
                a, b = acc["rejected_longer"], acc["preferred_longer"]
                gaps = (a[rng.integers(0, len(a), (N_BOOT, len(a)))].mean(1)
                        - b[rng.integers(0, len(b), (N_BOOT, len(b)))].mean(1))
                entry["gap_rejected_minus_preferred_longer"] = {
                    "value": float(a.mean() - b.mean()),
                    "ci95": [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))]}
            out["conditions"][name][split] = entry

    if {"standard", "length_balanced"} <= set(names):
        def key(p):
            return p["prompt_id"], p["length_stratum"]
        a = {key(p): p for p in read_jsonl(results_dir / "standard" / "eval_pairs_length_stratified.jsonl")}
        b = {key(p): p for p in read_jsonl(results_dir / "length_balanced" / "eval_pairs_length_stratified.jsonl")}
        paired = {}
        for s in STRATA + ["all"]:
            ks = [k for k in a if s == "all" or k[1] == s]
            d = np.array([b[k]["correct"] - a[k]["correct"] for k in ks], float)
            lo, hi = bootstrap_ci(d, rng)
            paired[s] = {"accuracy_diff": float(d.mean()), "ci95": [lo, hi],
                         "discordant_pairs": int((d != 0).sum()), "n_pairs": len(d)}
        out["length_balanced_minus_standard"] = paired
    return out


def plot_all(results_dir, fig_dir, df: pd.DataFrame, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)

    # (a) Beta sweep: four panels, short forks as a line, standard 1-epoch run as a separate marker.
    forks = df[df["condition"].str.startswith("beta_")].sort_values("train_beta")
    if len(forks) and "kl_token_mean" in df:
        panels = [("heldout_pref_acc", "Held-out pref. accuracy"), ("kl_token_mean", "KL to ref (per token)"),
                  ("rm_reward_mean", "RM score"), ("gen_tokens_mean", "Generated tokens (mean)")]
        fig, axes = plt.subplots(1, 4, figsize=(13, 2.8))
        std = df[df["condition"] == "standard"]
        sft = df[df["condition"] == "sft"]
        for ax, (col, title) in zip(axes, panels):
            se = {"heldout_pref_acc": "heldout_pref_acc_se", "rm_reward_mean": "rm_reward_se"}.get(col)
            ax.errorbar(forks["train_beta"], forks[col], yerr=forks[se] if se else None, fmt="o-", capsize=3,
                        label="short fork (600 pairs)")
            if len(std):
                ax.errorbar(std["train_beta"], std[col], yerr=std[se] if se else None, fmt="s", color="C3",
                            capsize=3, label="standard (1 epoch, 1500 pairs)")
            if len(sft) and col != "heldout_pref_acc":
                ax.axhline(float(sft[col].iloc[0]), ls="--", color="gray", lw=1, label="SFT reference")
            ax.set_xscale("log")
            ax.set_xticks(forks["train_beta"])
            ax.set_xticklabels([f"{b:g}" for b in forks["train_beta"]])
            ax.set_xlabel(r"$\beta$")
            ax.set_title(title, fontsize=10)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "beta_sweep.pdf")
        fig.savefig(fig_dir / "beta_sweep.png", dpi=150)
        plt.close(fig)

    # (b) Per-stratum accuracy: standard vs length-balanced.
    pair = df[df["condition"].isin(["standard", "length_balanced"])]
    if len(pair):
        fig, ax = plt.subplots(figsize=(5, 3))
        x = np.arange(len(STRATA))
        w = 0.8 / len(pair)
        for i, (_, r) in enumerate(pair.iterrows()):
            ax.bar(x + i * w - 0.4 + w / 2, [r[f"acc_{s}"] for s in STRATA], w, label=r["condition"],
                   yerr=[r[f"acc_{s}_se"] for s in STRATA], capsize=3)
        ax.axhline(0.5, ls="--", color="gray", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in STRATA])
        ax.set_ylabel("Held-out preference accuracy")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "length_strata.pdf")
        fig.savefig(fig_dir / "length_strata.png", dpi=150)
        plt.close(fig)

    # (c) Training curves.
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.8))
    for name in names:
        log = results_dir / name / "train_log.jsonl"
        if not log.exists():
            continue
        tl = pd.DataFrame(read_jsonl(log))
        for ax, col in zip(axes, ["loss", "preference_accuracy", "margin_mean"]):
            ax.plot(tl["examples_seen"], tl[col], label=name, lw=1.2)
            ax.set_xlabel("training pairs seen")
            ax.set_title(f"train {col}", fontsize=10)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "training_curves.pdf")
    fig.savefig(fig_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def clip(text: str, n: int = 600) -> str:
    text = text.strip()
    return text if len(text) <= n else text[:n] + " [...]"


def qualitative_candidates(results_dir, target: str, k: int) -> str:
    """Shortlist (not a selection) of examples for the report; the author picks and trims."""
    lines = [f"# Task 1 qualitative candidates ({target} vs sft)\n",
             "Auto-generated shortlist. Read the responses and choose; nothing here is a judgement of quality.\n"]
    sft_dir, tgt_dir = results_dir / "sft", results_dir / target
    if not (sft_dir / "generations.jsonl").exists() or not (tgt_dir / "generations.jsonl").exists():
        return "\n".join(lines + ["(needs generations for both `sft` and the target condition)"])

    base = {r["prompt_id"]: r for r in read_jsonl(sft_dir / "generations.jsonl")}
    tgt = [r for r in read_jsonl(tgt_dir / "generations.jsonl") if r["prompt_id"] in base]
    for r in tgt:
        r["reward_gain"] = r["reward"] - base[r["prompt_id"]]["reward"]
        r["token_gain"] = r["response_tokens"] - base[r["prompt_id"]]["response_tokens"]

    lines.append("## A. Higher RM score than SFT *and* much longer (check: is it actually better, or just longer?)\n")
    for r in sorted([r for r in tgt if r["reward_gain"] > 0], key=lambda r: -r["token_gain"])[:k]:
        b = base[r["prompt_id"]]
        lines += [f"### prompt_id `{r['prompt_id']}`", f"**Prompt:** {clip(r['prompt'], 400)}\n",
                  f"- SFT: reward {b['reward']:.3f}, {b['response_tokens']} tokens, truncated={b['truncated']}",
                  f"- {target}: reward {r['reward']:.3f}, {r['response_tokens']} tokens, truncated={r['truncated']}\n",
                  f"**SFT response:**\n\n> {clip(b['response']).replace(chr(10), chr(10) + '> ')}\n",
                  f"**{target} response:**\n\n> {clip(r['response']).replace(chr(10), chr(10) + '> ')}\n"]

    lines.append("## B. Truncated at the generation cap but scored higher than SFT\n")
    for r in sorted([r for r in tgt if r["truncated"] and r["reward_gain"] > 0], key=lambda r: -r["reward_gain"])[:k]:
        lines += [f"- `{r['prompt_id']}`: reward gain {r['reward_gain']:+.3f}, tokens {r['response_tokens']} "
                  f"(SFT {base[r['prompt_id']]['response_tokens']})"]

    lines.append("\n## C. Word-limit prompts (greedy): word count vs limit and RM score\n")
    wl_t = {r["prompt_id"]: r for r in read_jsonl(tgt_dir / "word_limit.jsonl") if r["decoding"] == "greedy"}
    wl_b = {r["prompt_id"]: r for r in read_jsonl(sft_dir / "word_limit.jsonl") if r["decoding"] == "greedy"}
    lines.append("| prompt | limit | SFT words | SFT RM | " + target + " words | " + target + " RM |")
    lines.append("|---|---|---|---|---|---|")
    for pid, r in wl_t.items():
        b = wl_b.get(pid, {})
        lines.append(f"| {pid} | {r['word_limit']} | {b.get('response_words')} | {b.get('reward', float('nan')):.3f} | "
                     f"{r['response_words']} | {r['reward']:.3f} |")
    violators = sorted([r for r in wl_t.values() if not r["compliant"]], key=lambda r: -r["reward"])[:k]
    for r in violators:
        lines += [f"\n### `{r['prompt_id']}` ({r['response_words']} words, limit {r['word_limit']}, RM {r['reward']:.3f})",
                  f"**Prompt:** {r['prompt']}\n", f"> {clip(r['response']).replace(chr(10), chr(10) + '> ')}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--k", type=int, default=5, help="candidates per qualitative category")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"])
    names = condition_names(results_dir, [float(b) for b in cfg["betas"]])
    if not names:
        raise SystemExit(f"No eval_metrics.json found under {results_dir}")

    df = summary_table(results_dir, names)
    df.to_csv(results_dir / "summary.csv", index=False)
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.precision", 4):
        print(df.T.to_string(header=False))
    plot_all(results_dir, results_dir / "figures", df, names)
    save_json(results_dir / "length_mechanism.json", length_mechanism(results_dir, names))
    for target in ["standard", "length_balanced"]:
        if (results_dir / target / "generations.jsonl").exists():
            (results_dir / f"qualitative_candidates_{target}.md").write_text(
                qualitative_candidates(results_dir, target, args.k), encoding="utf-8")
    print(f"\nWrote {results_dir / 'summary.csv'}, length_mechanism.json, figures/, qualitative_candidates_*.md")


if __name__ == "__main__":
    main()
