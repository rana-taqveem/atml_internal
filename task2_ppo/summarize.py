"""Collect all Task 2 result files into tables, figures and qualitative candidates (CPU only)."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from task2_ppo.forks import fork_name

TRAJ = [("reward", "learned reward"), ("kl_token_mean", "KL to ref (token)"), ("policy_loss", "policy loss"),
        ("value_loss", "value loss"), ("entropy", "entropy"), ("clip_fraction", "clip fraction"),
        ("policy_grad_norm", "policy grad norm"), ("response_tokens", "response tokens"),
        ("critic_explained_variance", "critic explained var.")]


def bootstrap_ci(values, seed=0, n_boot=5000):
    v = np.asarray(values, float)
    means = v[np.random.default_rng(seed).integers(0, len(v), (n_boot, len(v)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def stability(log: pd.DataFrame) -> dict:
    """Stability statistics of one continuation, from its per-update log."""
    return {
        "updates": int(len(log)),
        "tokens_generated": int(log["tokens_generated"].iloc[-1]),
        "mean_step_kl_k3": float(log["step_kl_old_new_k3"].mean()),   # policy change per update
        "max_step_kl_k3": float(log["step_kl_old_new_k3"].max()),
        "mean_clip_fraction": float(log["clip_fraction"].mean()),
        "mean_affected_fraction": float(log["affected_fraction"].mean()),
        "max_ratio": float(log["ratio_max"].max()),
        "min_ratio": float(log["ratio_min"].min()),
        "grad_norm_mean": float(log["policy_grad_norm"].mean()),
        "grad_norm_std": float(log["policy_grad_norm"].std(ddof=0)),
        "policy_loss_std": float(log["policy_loss"].std(ddof=0)),
        "nonfinite_steps": int(log["nonfinite_steps"].sum()),
        "train_reward_mean": float(log["reward"].mean()),
        "train_kl_last": float(log["kl_token_mean"].iloc[-1]),
    }


def condition_table(results_dir, cfg) -> pd.DataFrame:
    eps0, beta0 = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    conds = [("sft", None, None), ("midpoint", None, None), ("standard", eps0, beta0)]
    conds += [(fork_name(float(e), beta0), float(e), beta0) for e in cfg["clip_values"]]
    conds += [(fork_name(eps0, float(b)), eps0, float(b)) for b in cfg["kl_values"] if float(b) != beta0]
    base_path = results_dir / "midpoint" / "heldout_generations.jsonl"
    base = {r["prompt_id"]: r for r in read_jsonl(base_path)} if base_path.exists() else {}
    rows = []
    for name, eps, beta in conds:
        mpath = results_dir / name / "eval_metrics.json"
        if not mpath.exists():
            continue
        m = load_json(mpath)
        row = {"condition": name, "clip_epsilon": eps, "kl_beta": beta,
               "heldout_reward": m["reward_mean"], "heldout_reward_se": m["reward_se"],
               "heldout_reward_raw": m["reward_raw_mean"], "heldout_kl_token": m["kl_token_mean"],
               "heldout_kl_seq": m["kl_seq_mean"], "heldout_entropy": m["entropy_token_mean"],
               "heldout_tokens_mean": m["response_tokens"]["mean"], "heldout_tokens_std": m["response_tokens"]["std"],
               "heldout_tokens_iqr": m["response_tokens"]["iqr"], "heldout_truncation": m["truncation_rate"]}
        gens = read_jsonl(results_dir / name / "heldout_generations.jsonl")
        if base and name != "midpoint":
            d = [g["reward"] - base[g["prompt_id"]]["reward"] for g in gens if g["prompt_id"] in base]
            row["reward_diff_vs_midpoint"] = float(np.mean(d))
            row["reward_diff_ci_lo"], row["reward_diff_ci_hi"] = bootstrap_ci(d)
        log_path = results_dir / name / "train_log.jsonl"
        if log_path.exists():
            row.update(stability(pd.DataFrame(read_jsonl(log_path))))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_all(results_dir, cfg, fig_dir):
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
        axes.flat[-1].plot(log["update"], log["step_kl_old_new_k3"], "o-", ms=3)
        axes.flat[-1].set_title("policy change per update (k3)", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / "standard_trajectories.pdf")
        fig.savefig(fig_dir / "standard_trajectories.png", dpi=150)
        plt.close(fig)

    eps0, beta0 = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    for study, key, values in [("kl", "kl_beta", cfg["kl_values"]), ("clip", "clip_epsilon", cfg["clip_values"])]:
        cols = [("reward", "learned reward"), ("kl_token_mean", "KL to ref"), ("entropy", "entropy"),
                ("response_tokens", "response tokens"), ("step_kl_old_new_k3", "policy change / update"),
                ("clip_fraction", "clip fraction")]
        fig, axes = plt.subplots(1, len(cols), figsize=(19, 2.9))
        found = False
        for v in values:
            name = fork_name(eps0, float(v)) if key == "kl_beta" else fork_name(float(v), beta0)
            p = results_dir / name / "train_log.jsonl"
            if not p.exists():
                continue
            found = True
            log = pd.DataFrame(read_jsonl(p))
            for ax, (col, title) in zip(axes, cols):
                ax.plot(log["update"], log[col], "o-", ms=3, label=f"{key}={float(v):g}")
                ax.set_title(title, fontsize=10)
                ax.set_xlabel("update")
        if found:
            axes[0].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(fig_dir / f"{study}_fork_trajectories.pdf")
            fig.savefig(fig_dir / f"{study}_fork_trajectories.png", dpi=150)
        plt.close(fig)

    cpath = results_dir / "clipping_cached.json"
    if cpath.exists():
        c = load_json(cpath)
        fig, axes = plt.subplots(1, 3, figsize=(11, 2.9))
        for eps, entry in c["eps"].items():
            pr = pd.DataFrame(entry["probe"])
            for ax, col in zip(axes, ["clip_fraction", "affected_fraction", "step_kl_k3"]):
                ax.plot(pr["step"], pr[col], "o-", ms=3, label=f"eps={eps}")
                ax.set_title(col.replace("_", " ") + " (cached batch)", fontsize=10)
                ax.set_xlabel("probe step")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "clipping_cached_probe.pdf")
        fig.savefig(fig_dir / "clipping_cached_probe.png", dpi=150)
        plt.close(fig)


def clip(text, n=700):
    text = text.strip()
    return text if len(text) <= n else text[:n] + " [...]"


def qualitative_candidates(results_dir, names, k: int) -> str:
    lines = ["# Task 2 qualitative candidates (vs supplied midpoint, same held-out prompt)\n",
             "Auto-generated shortlist ranked by learned-reward change; read the responses and judge quality yourself.\n"]
    base_path = results_dir / "midpoint" / "heldout_generations.jsonl"
    if not base_path.exists():
        return "\n".join(lines + ["(needs midpoint held-out generations)"])
    base = {r["prompt_id"]: r for r in read_jsonl(base_path)}
    for name in names:
        path = results_dir / name / "heldout_generations.jsonl"
        if not path.exists():
            continue
        gens = [g for g in read_jsonl(path) if g["prompt_id"] in base]
        for g in gens:
            g["gain"] = g["reward"] - base[g["prompt_id"]]["reward"]
        lines.append(f"\n## {name}\n")
        picks = [("largest reward gains", sorted(gens, key=lambda g: -g["gain"])[:k]),
                 ("largest reward drops", sorted(gens, key=lambda g: g["gain"])[:max(1, k // 2)])]
        for title, chosen in picks:
            lines.append(f"### {title}\n")
            for g in chosen:
                b = base[g["prompt_id"]]
                lines += [f"#### `{g['prompt_id']}` reward {b['reward']:+.2f} -> {g['reward']:+.2f} "
                          f"({b['response_tokens']} -> {g['response_tokens']} tokens, truncated {b['truncated']} -> {g['truncated']})",
                          f"**Prompt:** {clip(g['prompt'], 300)}\n",
                          f"**midpoint:**\n\n> {clip(b['response']).replace(chr(10), chr(10) + '> ')}\n",
                          f"**{name}:**\n\n> {clip(g['response']).replace(chr(10), chr(10) + '> ')}\n"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"])
    df = condition_table(results_dir, cfg)
    if df.empty:
        raise SystemExit(f"No eval_metrics.json under {results_dir}")
    df.to_csv(results_dir / "summary.csv", index=False)
    with pd.option_context("display.max_columns", None, "display.width", 220, "display.precision", 4):
        print(df.set_index("condition").T.to_string())
    std_summary = results_dir / "standard" / "train_summary.json"
    if std_summary.exists():
        s = load_json(std_summary)
        print(f"\nstandard continuation: wall-clock {s['wall_clock_s']} s, peak VRAM {s['peak_vram_gib']} GiB on {s['gpu']}")
    plot_all(results_dir, cfg, results_dir / "figures")
    beta0 = float(cfg["kl_beta"])
    names = ["standard"] + [fork_name(float(cfg["clip_epsilon"]), float(b)) for b in cfg["kl_values"]]
    names += [fork_name(float(e), beta0) for e in cfg["clip_values"] if float(e) != float(cfg["clip_epsilon"])]
    (results_dir / "qualitative_candidates.md").write_text(qualitative_candidates(results_dir, names, args.k),
                                                            encoding="utf-8")
    save_json(results_dir / "summary.json", df.to_dict(orient="records"))
    print(f"\nWrote {results_dir}/summary.csv, summary.json, figures/, qualitative_candidates.md")


if __name__ == "__main__":
    main()
