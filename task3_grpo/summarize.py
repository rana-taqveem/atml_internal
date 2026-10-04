"""Collect all Task 3 result files into tables, figures and qualitative candidates (CPU only)."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json

TRAJ = [("reward", "learned reward"), ("kl_token_mean", "KL to ref (token)"),
        ("group_reward_std", "within-group reward std"), ("uninformative_group_fraction", "uninformative groups"),
        ("policy_loss", "policy loss"), ("grad_norm", "grad norm"), ("entropy", "entropy"),
        ("response_tokens", "response tokens"), ("truncated_fraction", "truncated (masked) fraction")]
CONDITIONS = ["sft", "midpoint", "standard", "fork_grpo", "fork_dr_grpo"]


def bootstrap_ci(values, seed=0, n_boot=5000):
    v = np.asarray(values, float)
    means = v[np.random.default_rng(seed).integers(0, len(v), (n_boot, len(v)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def condition_table(results_dir) -> pd.DataFrame:
    base_path = results_dir / "midpoint" / "heldout_generations.jsonl"
    base = {r["prompt_id"]: r for r in read_jsonl(base_path)} if base_path.exists() else {}
    rows = []
    for name in CONDITIONS:
        mpath = results_dir / name / "eval_metrics.json"
        if not mpath.exists():
            continue
        m = load_json(mpath)
        row = {"condition": name, "heldout_reward": m["reward_mean"], "heldout_reward_se": m["reward_se"],
               "heldout_reward_raw": m["reward_raw_mean"], "heldout_kl_token": m["kl_token_mean"],
               "heldout_kl_seq": m["kl_seq_mean"], "heldout_entropy": m["entropy_token_mean"],
               "heldout_tokens_mean": m["response_tokens"]["mean"], "heldout_tokens_std": m["response_tokens"]["std"],
               "heldout_tokens_iqr": m["response_tokens"]["iqr"], "heldout_truncation": m["truncation_rate"]}
        if base and name != "midpoint":
            gens = read_jsonl(results_dir / name / "heldout_generations.jsonl")
            for key, col in [("reward", "reward"), ("response_tokens", "tokens")]:
                d = [g[key] - base[g["prompt_id"]][key] for g in gens if g["prompt_id"] in base]
                row[f"{col}_diff_vs_midpoint"] = float(np.mean(d))
                row[f"{col}_diff_ci_lo"], row[f"{col}_diff_ci_hi"] = bootstrap_ci(d)
        log_path = results_dir / name / "train_log.jsonl"
        if log_path.exists():
            log = pd.DataFrame(read_jsonl(log_path))
            row.update({"train_updates": len(log), "tokens_generated": int(log["tokens_generated"].iloc[-1]),
                        "train_reward_mean": float(log["reward"].mean()),
                        "train_uninformative_fraction": float(log["uninformative_group_fraction"].mean()),
                        "train_group_std_mean": float(log["group_reward_std"].mean()),
                        "train_truncated_fraction": float(log["truncated_fraction"].mean()),
                        "grad_norm_mean": float(log["grad_norm"].mean()), "grad_norm_std": float(log["grad_norm"].std(ddof=0))})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_all(results_dir, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "standard" / "train_log.jsonl"
    if log_path.exists():
        log = pd.DataFrame(read_jsonl(log_path))
        fig, axes = plt.subplots(2, 5, figsize=(16, 5.5))
        for ax, (col, title) in zip(axes.flat, TRAJ):
            ax.plot(log["update"], log[col], "o-", ms=3)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("update")
        axes.flat[-1].plot(log["update"], log["corr_seq_grad_norm_vs_length"], "o-", ms=3)
        axes.flat[-1].set_title("corr(per-completion grad, length)", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / "standard_trajectories.pdf")
        fig.savefig(fig_dir / "standard_trajectories.png", dpi=150)
        plt.close(fig)

    gpath = results_dir / "group_size_study.json"
    if gpath.exists():
        g = load_json(gpath)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        for name, ls in [("learned_reward", "-"), ("binarized_reward", "--")]:
            byk = g[name]["by_k"]
            ks = sorted(int(k) for k in byk)
            for b, c in [("hard", "C3"), ("medium", "C1"), ("easy", "C2")]:
                axes[0].plot(ks, [byk[str(k)]["by_difficulty"].get(b, {}).get("informative_rate", np.nan) for k in ks],
                             ls, marker="o", color=c, label=f"{b} ({name.split('_')[0]})")
            axes[1].plot(ks, [byk[str(k)]["all"]["mean_within_group_std"] for k in ks], ls, marker="o", label=name)
            axes[2].plot(ks, [byk[str(k)]["instability"]["mean_partition_variance_of_advantage"] for k in ks], ls,
                         marker="o", label=name)
        for ax, t in zip(axes, ["informative-group rate by difficulty", "mean within-group reward std",
                                "advantage variance across regroupings"]):
            ax.set_xticks([2, 4, 8])
            ax.set_xlabel("K (equal 192 generations)")
            ax.set_title(t, fontsize=10)
        axes[0].legend(fontsize=6)
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "group_size_study.pdf")
        fig.savefig(fig_dir / "group_size_study.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.4))
    found = False
    for lt, c in [("grpo", "C0"), ("dr_grpo", "C3")]:
        p = results_dir / f"fork_{lt}" / "train_log.jsonl"
        if not p.exists():
            continue
        found = True
        L, Y = [], []
        for row in read_jsonl(p):
            for n, gn, a in zip(row["seq_lengths"], row["seq_grad_norms"], row["seq_advantages"]):
                if gn > 0:
                    L.append(n), Y.append(gn / max(abs(a), 1e-6))
        ax.scatter(L, Y, s=10, alpha=0.6, color=c, label=lt)
    if found:
        ax.set_xlabel("completion length (tokens)")
        ax.set_ylabel("grad norm / |advantage|")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "normalization_grad_vs_length.pdf")
        fig.savefig(fig_dir / "normalization_grad_vs_length.png", dpi=150)
    plt.close(fig)


def clip(text, n=600):
    text = text.strip()
    return text if len(text) <= n else text[:n] + " [...]"


def qualitative_candidates(results_dir, cfg, k: int) -> str:
    lines = ["# Task 3 qualitative candidates\n", "Auto-generated shortlist; read and judge yourself.\n"]
    cache = read_jsonl(cfg["group_cache"])
    groups = {}
    for r in cache:
        groups.setdefault(r["source_index"], []).append(r)
    lines.append("## A. Group informativeness (cached K=8 groups): smallest and largest within-group reward std\n")
    ranked = sorted(groups.items(), key=lambda kv: np.std([c["reward"] for c in kv[1]]))
    for sid, comps in ranked[:2] + ranked[-1:]:
        rs = [c["reward"] for c in comps]
        lines.append(f"### source_index {sid}: rewards {np.round(rs, 2).tolist()} (std {np.std(rs):.3f})")
        best, worst = max(comps, key=lambda c: c["reward"]), min(comps, key=lambda c: c["reward"])
        lines += [f"- best ({best['reward']:+.2f}, {best['completion_tokens']} tok): {clip(best['completion'], 300)}",
                  f"- worst ({worst['reward']:+.2f}, {worst['completion_tokens']} tok): {clip(worst['completion'], 300)}\n"]
    lines.append("## B. Normalisation forks: same held-out prompt, canonical GRPO vs Dr. GRPO (largest length differences)\n")
    pa, pb = results_dir / "fork_grpo" / "heldout_generations.jsonl", results_dir / "fork_dr_grpo" / "heldout_generations.jsonl"
    if pa.exists() and pb.exists():
        a = {r["prompt_id"]: r for r in read_jsonl(pa)}
        b = {r["prompt_id"]: r for r in read_jsonl(pb)}
        common_ids = sorted(set(a) & set(b), key=lambda i: -abs(a[i]["response_tokens"] - b[i]["response_tokens"]))
        for pid in common_ids[:k]:
            lines += [f"### `{pid}` grpo {a[pid]['response_tokens']} tok / reward {a[pid]['reward']:+.2f} vs "
                      f"dr_grpo {b[pid]['response_tokens']} tok / reward {b[pid]['reward']:+.2f}",
                      f"**Prompt:** {clip(a[pid]['prompt'], 300)}\n",
                      f"**grpo:**\n\n> {clip(a[pid]['response']).replace(chr(10), chr(10) + '> ')}\n",
                      f"**dr_grpo:**\n\n> {clip(b[pid]['response']).replace(chr(10), chr(10) + '> ')}\n"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"])
    df = condition_table(results_dir)
    if not df.empty:
        df.to_csv(results_dir / "summary.csv", index=False)
        save_json(results_dir / "summary.json", df.to_dict(orient="records"))
        with pd.option_context("display.max_columns", None, "display.width", 220, "display.precision", 4):
            print(df.set_index("condition").T.to_string())
    s = results_dir / "standard" / "train_summary.json"
    if s.exists():
        s = load_json(s)
        print(f"\nstandard continuation: wall-clock {s['wall_clock_s']} s, peak VRAM {s['peak_vram_gib']} GiB on {s['gpu']}")
    plot_all(results_dir, results_dir / "figures")
    (results_dir / "qualitative_candidates.md").write_text(qualitative_candidates(results_dir, cfg, args.k), encoding="utf-8")
    print(f"\nWrote {results_dir}/summary.csv, figures/, qualitative_candidates.md")


if __name__ == "__main__":
    main()
