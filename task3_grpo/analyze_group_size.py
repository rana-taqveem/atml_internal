from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json

TOL = 1e-6  # same std tolerance as task3_grpo.grpo.group_reward_stats


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int, order: dict | None = None):
    """Return K-sized groups while keeping total cached completions fixed.

    Every prompt's 8 completions are split into 8/K disjoint groups of size K (in generation_index
    order, or in the permutation given by `order[prompt]`). All K therefore use the same 24 prompts x
    8 = 192 generations: K=2 -> 96 groups, K=4 -> 48, K=8 -> 24. Equal *generations*, not equal prompts.
    """
    groups = []
    for pid, comps in by_prompt.items():
        n = (len(comps) // k) * k
        idx = order[pid] if order is not None else np.arange(len(comps))
        for start in range(0, n, k):
            groups.append((pid, [comps[i] for i in idx[start:start + k]]))
    return groups


def group_metrics(groups, reward_key: str = "reward") -> dict:
    stds, informative, centered, norm_adv = [], [], [], []
    for _, comps in groups:
        r = np.array([c[reward_key] for c in comps], float)
        s = r.std()
        stds.append(s)
        informative.append(s > TOL)
        centered.extend(r - r.mean())
        norm_adv.extend((r - r.mean()) / (s + TOL))
    return {
        "n_groups": len(groups),
        "informative_rate": float(np.mean(informative)),
        "uninformative_fraction": float(1 - np.mean(informative)),
        "mean_within_group_std": float(np.mean(stds)),
        # Group-relative signal before / after std scaling, pooled over all completions.
        "var_centered_signal": float(np.var(centered)),
        "var_normalized_advantage": float(np.var(norm_adv)),
        "mean_abs_normalized_advantage": float(np.mean(np.abs(norm_adv))),
    }


def advantage_instability(by_prompt, k: int, n_partitions: int, rng, reward_key: str = "reward") -> dict:
    """How much one completion's advantage depends on which K-1 group-mates it was sampled with.

    Over random partitions of each prompt's 8 completions into K-groups, compute each completion's
    normalised advantage; report the mean (over completions) of its variance across partitions, and
    how often its sign flips relative to the full-group (K=8) advantage.
    """
    advs = defaultdict(list)
    for _ in range(n_partitions):
        order = {pid: rng.permutation(len(c)) for pid, c in by_prompt.items()}
        for pid, comps in regroup_equal_generation_budget(by_prompt, k, order):
            r = np.array([c[reward_key] for c in comps], float)
            a = (r - r.mean()) / (r.std() + TOL)
            for c, ai in zip(comps, a):
                advs[(pid, c["generation_index"])].append(ai)
    full = {}
    for pid, comps in by_prompt.items():
        r = np.array([c[reward_key] for c in comps], float)
        for c, ai in zip(comps, (r - r.mean()) / (r.std() + TOL)):
            full[(pid, c["generation_index"])] = ai
    var = [np.var(v) for v in advs.values()]
    flips = [np.mean(np.sign(v) != np.sign(full[key])) for key, v in advs.items() if abs(full[key]) > TOL]
    return {"mean_partition_variance_of_advantage": float(np.mean(var)),
            "sign_disagreement_with_k8": float(np.mean(flips)) if flips else float("nan")}


def difficulty_bins(by_prompt, reward_key: str = "reward"):
    """Binning rule (defined once): tertiles of each prompt's mean reward over all 8 cached completions.

    'hard' = lowest third of mean learned reward, 'easy' = highest third.
    """
    means = {pid: float(np.mean([c[reward_key] for c in comps])) for pid, comps in by_prompt.items()}
    lo, hi = np.percentile(list(means.values()), [100 / 3, 200 / 3])
    label = lambda m: "hard" if m <= lo else ("easy" if m > hi else "medium")
    return {pid: label(m) for pid, m in means.items()}, {"tertile_cutoffs": [float(lo), float(hi)], "prompt_mean_reward": means}


def study(by_prompt, ks, n_partitions, seed, reward_key="reward", bins=None):
    """Equal-generation metrics for every K, overall and per difficulty bin (bins from the learned reward)."""
    rng = np.random.default_rng(seed)
    bins_learned, bin_info = difficulty_bins(by_prompt)
    bins = bins or bins_learned
    out = {"total_generations": int(sum(len(c) for c in by_prompt.values())), "n_prompts": len(by_prompt),
           "difficulty": bin_info, "bins": bins, "by_k": {}}
    for k in ks:
        groups = regroup_equal_generation_budget(by_prompt, k)
        entry = {"all": group_metrics(groups, reward_key),
                 "instability": advantage_instability(by_prompt, k, n_partitions, rng, reward_key),
                 "by_difficulty": {}}
        # Expected informative rate over random partitions (less dependent on generation order).
        rates = []
        for _ in range(n_partitions):
            order = {pid: rng.permutation(len(c)) for pid, c in by_prompt.items()}
            rates.append(group_metrics(regroup_equal_generation_budget(by_prompt, k, order), reward_key)["informative_rate"])
        entry["all"]["informative_rate_random_partitions"] = float(np.mean(rates))
        for b in ["hard", "medium", "easy"]:
            sub = [g for g in groups if bins[g[0]] == b]
            if sub:
                entry["by_difficulty"][b] = group_metrics(sub, reward_key)
        out["by_k"][str(k)] = entry
    return out


def binarize(by_prompt, threshold: float):
    """Low-resolution reward variant: 1 if learned reward > threshold else 0 (diagnostic only)."""
    return {pid: [dict(c, reward_bin=float(c["reward"] > threshold)) for c in comps] for pid, comps in by_prompt.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    print("Cached prompts:", len(by_prompt))
    print("Group sizes to analyze:", cfg["group_sizes"])
    first = next(iter(by_prompt.values()))
    print("Cache row keys:", sorted(first[0].keys()))

    ks = [int(k) for k in cfg["group_sizes"]]
    n_part = int(cfg.get("group_study_partitions", 200))
    result = {"reward": "learned reward model score (as cached)", "tolerance": TOL,
              "partitioning": "each prompt's 8 completions split into disjoint K-groups; equal 192 generations for every K",
              "learned_reward": study(by_prompt, ks, n_part, int(cfg["seed"]))}
    # Supplementary: same analysis with a binary reward (threshold = median cached reward).
    median = float(np.median([c["reward"] for comps in by_prompt.values() for c in comps]))
    result["binarized_reward"] = {"threshold": median,
                                  **study(binarize(by_prompt, median), ks, n_part, int(cfg["seed"]), "reward_bin",
                                          bins=result["learned_reward"]["bins"])}
    save_json(repo_path(cfg["results_dir"]) / "group_size_study.json", result)

    for name in ["learned_reward", "binarized_reward"]:
        print(f"\n{name}:")
        print(f"{'K':>3}{'groups':>8}{'inform.':>9}{'grp std':>9}{'var(r-mu)':>11}{'adv var/part':>14}{'sign flip':>11}"
              "   informative rate hard/medium/easy")
        for k, e in result[name]["by_k"].items():
            a, ins = e["all"], e["instability"]
            bd = " / ".join(f"{e['by_difficulty'][b]['informative_rate']:.2f}" for b in ["hard", "medium", "easy"]
                            if b in e["by_difficulty"])
            print(f"{k:>3}{a['n_groups']:>8}{a['informative_rate']:>9.3f}{a['mean_within_group_std']:>9.3f}"
                  f"{a['var_centered_signal']:>11.3f}{ins['mean_partition_variance_of_advantage']:>14.3f}"
                  f"{ins['sign_disagreement_with_k8']:>11.3f}   {bd}")
    print(f"\nSaved {repo_path(cfg['results_dir']) / 'group_size_study.json'}")


if __name__ == "__main__":
    main()
