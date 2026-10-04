from __future__ import annotations

import argparse

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.metrics import safe_corr
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import evaluate_policy

LOSS_TYPES = ["grpo", "dr_grpo"]


def fork_name(loss_type: str) -> str:
    return f"fork_{loss_type}"


def length_conditioned_stats(log_rows) -> dict:
    """Per-completion gradient norm vs completion length, pooled over all updates of one fork.

    Uses the gradient each completion contributed on its own (logged by grpo_update). Dividing by
    |advantage| removes the reward-driven part, leaving the weight the normalisation gives to a
    completion of that length.
    """
    L, G, A = [], [], []
    for row in log_rows:
        for n, g, a in zip(row["seq_lengths"], row["seq_grad_norms"], row["seq_advantages"]):
            if g > 0:  # completions with no loss tokens (masked/truncated) are excluded
                L.append(n), G.append(g), A.append(abs(a))
    L, G, A = np.array(L, float), np.array(G, float), np.array(A, float)
    if len(L) < 2:
        return {}
    per_adv = G / np.maximum(A, 1e-6)
    med = float(np.median(L))
    short, long_ = L <= med, L > med
    return {
        "n_completions": int(len(L)),
        "median_length": med,
        "corr_grad_norm_vs_length": safe_corr(G, L),
        "corr_grad_per_abs_adv_vs_length": safe_corr(per_adv, L),
        "mean_grad_norm_short": float(G[short].mean()),
        "mean_grad_norm_long": float(G[long_].mean()) if long_.any() else float("nan"),
        "mean_grad_per_abs_adv_short": float(per_adv[short].mean()),
        "mean_grad_per_abs_adv_long": float(per_adv[long_].mean()) if long_.any() else float("nan"),
        "share_of_grad_norm_from_long": float(G[long_].sum() / G.sum()),
        "share_of_tokens_from_long": float(L[long_].sum() / L.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--skip-existing", action="store_true", help="reuse adapters/metrics that already exist")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"])
    print("Fork updates:", cfg["fork_updates"])
    print("Compare loss_type='grpo' vs loss_type='dr_grpo' from the identical supplied midpoint.")

    # Same midpoint, prompts, seeds, K, reward, beta, eps and generation cap; only the normaliser differs.
    for lt in LOSS_TYPES:
        name = fork_name(lt)
        output = cfg["fork_output_template"].format(name=name)
        if args.skip_existing and (repo_path(output) / "adapter_config.json").exists():
            print(f"[{name}] reusing existing adapter at {output}")
        else:
            run_grpo(args.config, output=output, updates=int(cfg["fork_updates"]), loss_type=lt, run_name=name)
        if not (args.skip_existing and (results_dir / name / "eval_metrics.json").exists()):
            evaluate_policy(args.config, output, name)

    report = {}
    for lt in LOSS_TYPES:
        name = fork_name(lt)
        log = read_jsonl(results_dir / name / "train_log.jsonl")
        m = load_json(results_dir / name / "eval_metrics.json")
        report[lt] = {
            "heldout_reward": m["reward_mean"], "heldout_reward_se": m["reward_se"],
            "heldout_kl_token": m["kl_token_mean"], "heldout_entropy": m["entropy_token_mean"],
            "heldout_tokens": m["response_tokens"], "heldout_truncation": m["truncation_rate"],
            "train_reward_mean": float(np.mean([r["reward"] for r in log])),
            "train_tokens_mean": float(np.mean([r["response_tokens"] for r in log])),
            "tokens_generated": int(log[-1]["tokens_generated"]),
            "length_conditioned": length_conditioned_stats(log),
        }
    save_json(results_dir / "normalization_study.json", report)
    for lt, e in report.items():
        lc = e["length_conditioned"]
        print(f"[{lt}] held-out reward={e['heldout_reward']:+.3f}+-{e['heldout_reward_se']:.3f} "
              f"kl={e['heldout_kl_token']:.4f} len={e['heldout_tokens']['mean']:.1f} | "
              f"corr(grad/|A|, len)={lc.get('corr_grad_per_abs_adv_vs_length', float('nan')):+.3f} "
              f"long share of grad={lc.get('share_of_grad_norm_from_long', float('nan')):.2f} "
              f"(of tokens {lc.get('share_of_tokens_from_long', float('nan')):.2f})")
    print(f"Saved {results_dir / 'normalization_study.json'}")


if __name__ == "__main__":
    main()
