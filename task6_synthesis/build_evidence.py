"""Build the report evidence pack from saved results (CPU only, no inference).

Writes results/evidence/<table>.csv and results/evidence/EVIDENCE.md: one compact table per required
evidence item / research question, with uncertainty, plus a metric-reading guide. The guide states
definitions, measured facts and caveats only; interpretation is left to the report author.

    python -m task6_synthesis.build_evidence
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.data import read_jsonl, repo_path

R = repo_path("results")
OUT = R / "evidence"
RNG_SEED = 0


# ----------------------------------------------------------------------------- helpers
def jload(rel):
    return json.loads((R / rel).read_text(encoding="utf-8"))


def se_binom(p, n):
    return float(np.sqrt(p * (1 - p) / n)) if n else float("nan")


def boot_ci(values, n_boot=5000):
    v = np.asarray(values, float)
    if len(v) == 0:
        return float("nan"), float("nan")
    m = v[np.random.default_rng(RNG_SEED).integers(0, len(v), (n_boot, len(v)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(a_path, b_path, key, id_key="prompt_id"):
    """mean(a - b) over shared ids with a 95% bootstrap CI (a, b = generation files)."""
    a = {r[id_key]: r for r in read_jsonl(a_path)}
    b = {r[id_key]: r for r in read_jsonl(b_path)}
    d = [a[i][key] - b[i][key] for i in a if i in b]
    lo, hi = boot_ci(d)
    return float(np.mean(d)), lo, hi


def identical(a_path, b_path, text_key="response", id_key="prompt_id"):
    a = {r[id_key]: r[text_key] for r in read_jsonl(a_path)}
    b = {r[id_key]: r[text_key] for r in read_jsonl(b_path)}
    return sum(a[i] == b[i] for i in a if i in b), len(a)


def f(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    if isinstance(x, (int, np.integer)):
        return str(x)
    return f"{x:.{d}f}"


def pm(x, s, d=3):
    return f"{f(x, d)} ± {f(s, d)}"


def ci(m, lo, hi, d=3):
    return f"{m:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


TABLES: dict[str, pd.DataFrame] = {}


def table(name, rows):
    df = pd.DataFrame(rows)
    TABLES[name] = df
    df.to_csv(OUT / f"{name}.csv", index=False)
    return md_table(df)


# ----------------------------------------------------------------------------- Task 1
def task1():
    s = pd.read_csv(R / "task1_dpo/summary.csv").set_index("condition")
    lm = jload("task1_dpo/length_mechanism.json")
    la = jload("task1_dpo/length_analysis.json")
    ts = jload("task1_dpo/truncation_sensitivity.json")
    order = ["sft", "standard", "beta_0.03", "beta_0.1", "beta_0.3"]
    rows = []
    for c in order:
        r = s.loc[c]
        rows.append({
            "condition": c, "budget (pairs/updates)": "-" if c == "sft" else f"{int(r.train_pairs)}/{int(r.updates)}",
            "beta": "-" if c == "sft" else f(r.train_beta, 2),
            "pref. acc ± SE": "n/a (m=0 by def.)" if c == "sft" else pm(r.heldout_pref_acc, r.heldout_pref_acc_se),
            "margin m (nats)": f(r.heldout_margin_mean), "DPO loss": f(r.heldout_dpo_loss, 4),
            "logratio chosen/rejected": f"{r.chosen_logratio_mean:+.2f}/{r.rejected_logratio_mean:+.2f}",
            "KL tok (seq)": f"{r.kl_token_mean:.5f} ({r.kl_seq_mean:.3f})",
            "RM ± SE": pm(r.rm_reward_mean, r.rm_reward_se),
            "ΔRM vs SFT [95% CI]": "-" if c == "sft" else ci(r.rm_reward_diff_vs_sft, r.rm_reward_diff_ci_lo, r.rm_reward_diff_ci_hi),
            "tokens mean ± sd": f"{r.gen_tokens_mean:.1f} ± {r.gen_tokens_std:.1f}",
            "ΔTokens vs SFT [CI]": "-" if c == "sft" else ci(r.gen_tokens_diff_vs_sft, r.gen_tokens_diff_ci_lo, r.gen_tokens_diff_ci_hi, 1),
            "trunc@256": f(r.truncation_rate, 2),
        })
    t11 = table("t1_1_dpo_summary", rows)

    # beta mechanism: gradient norm before clipping (first update) from logs
    rows = []
    for b in ["0.03", "0.1", "0.3"]:
        log = read_jsonl(R / f"task1_dpo/beta_{b}/train_log.jsonl")
        g = [x["grad_norm"] for x in log]
        rows.append({"beta": b, "grad-norm (pre-clip) mean": f(np.mean(g), 2), "clipped updates (>1.0)": f"{sum(x > 1.0 for x in g)}/{len(g)}",
                     "held-out margin": f(s.loc[f'beta_{b}'].heldout_margin_mean), "max beta·m": f(float(b) * s.loc[f'beta_{b}'].heldout_margin_mean, 3)})
    t12 = table("t1_2_beta_mechanism", rows)

    strata = ["preferred_longer", "length_matched", "rejected_longer"]
    paired_ = lm.get("length_balanced_minus_standard", {})
    rows = []
    for c in ["standard", "length_balanced"]:
        r = s.loc[c]
        rows.append({"model": c, **{st: pm(r[f"acc_{st}"], r[f"acc_{st}_se"]) for st in strata},
                     "gap rej-pref [CI]": ci(lm["conditions"][c]["length_stratified"]["gap_rejected_minus_preferred_longer"]["value"],
                                             *lm["conditions"][c]["length_stratified"]["gap_rejected_minus_preferred_longer"]["ci95"], 2),
                     "per-token-margin acc (pref/match/rej)": "/".join(
                         f(lm["conditions"][c]["length_stratified"]["by_stratum"][st]["accuracy_per_token_margin"], 2) for st in strata),
                     "corr(logratio, tokens)": f(lm["conditions"][c]["length_stratified"]["corr_logratio_vs_tokens"], 2),
                     "gen tokens": f"{r.gen_tokens_mean:.1f}", "word-limit greedy/sampled": f"{r.wordlimit_greedy_compliance:.2f}/{r.wordlimit_sampled_compliance:.2f}"})
    rows.append({"model": "balanced − standard (paired)", **{st: ci(paired_[st]["accuracy_diff"], *paired_[st]["ci95"], 2) + f" ({paired_[st]['discordant_pairs']} disc.)" for st in strata},
                 "gap rej-pref [CI]": "-", "per-token-margin acc (pref/match/rej)": "-", "corr(logratio, tokens)": "-", "gen tokens": "-", "word-limit greedy/sampled": "-"})
    t13 = table("t1_3_length_strata", rows)

    rows = [{"file": k, "pairs": v["n_pairs"], "P(chosen longer)": f(v["frac_chosen_longer"], 3),
             "'longer wins' acc": f(v["longer_is_better_accuracy"], 3),
             "chosen/rejected tokens": f"{v['chosen_tokens_mean']:.0f}/{v['rejected_tokens_mean']:.0f}"}
            for k, v in la["dataset_length_profile"].items()]
    t14 = table("t1_4_dataset_length", rows)

    tr = ts["eval_splits"]["standard"]
    rows = [{"condition": c, "acc all (300)": f(v["all"]["preference_accuracy"]), "acc excl. context-free (279)": f(v["excluding_context_free"]["preference_accuracy"]),
             "acc context-free (21)": f(v["context_free"]["preference_accuracy"])} for c, v in tr["conditions"].items()]
    t15 = table("t1_5_truncation_sensitivity", rows)
    train_counts = ts["training_files"]
    return f"""## Task 1 — DPO

**T1.1 Summary (standard = 1 epoch/1500 pairs; β forks = 600 pairs; SFT = untouched reference).** Held-out pairs n=300; generation n=128 prompts, T=0.7, 256-token cap.
{t11}

**T1.2 β mechanism (RQ1).**
{t12}

**T1.3 Length-stratified held-out accuracy (n=82 per stratum) and generation (RQ2).**
{t13}

**T1.4 Length structure of the data itself (property of the dataset, not the policy).**
{t14}

**T1.5 Encoder-truncation sensitivity (standard held-out split).** Training pairs with a context-free response: standard {train_counts['data/dpo_standard_train.jsonl']['context_free']}/1500, balanced {train_counts['data/dpo_length_balanced_train.jsonl']['context_free']}/1500; differing prompt prefix: {train_counts['data/dpo_standard_train.jsonl']['context_differs']} / {train_counts['data/dpo_length_balanced_train.jsonl']['context_differs']}.
{t15}

*Qualitative:* `results/task1_dpo/qualitative_candidates_standard.md` (word-limit table: e.g. wl04, wl06). Figures: `results/task1_dpo/figures/`.
"""


# ----------------------------------------------------------------------------- Task 2
def task2():
    s = pd.read_csv(R / "task2_ppo/summary.csv").set_index("condition")
    log = pd.DataFrame(read_jsonl(R / "task2_ppo/standard/train_log.jsonl"))
    ts = jload("task2_ppo/standard/train_summary.json")
    cols = [("reward", "learned reward"), ("kl_token_mean", "KL (token)"), ("policy_loss", "policy loss"), ("value_loss", "value loss"),
            ("entropy", "entropy"), ("clip_fraction", "clip fraction"), ("policy_grad_norm", "grad norm"),
            ("response_tokens", "response tokens"), ("critic_explained_variance", "critic EV")]
    rows = [{"metric": lab, "updates 1-5 mean": f(log[c][:5].mean(), 4), "updates 16-20 mean": f(log[c][-5:].mean(), 4),
             "min": f(log[c].min(), 4), "max": f(log[c].max(), 4)} for c, lab in cols]
    t21 = table("t2_1_standard_trajectory", rows)

    c = jload("task2_ppo/clipping_cached.json")
    base = R / "task2_ppo/midpoint/heldout_generations.jsonl"
    rows = []
    for e in ["0.05", "0.2", "0.5"]:
        name = f"fork_eps{e}_kl0.1"
        r, m0, last = s.loc[name], c["eps"][e]["at_midpoint"], c["eps"][e]["probe"][-1]
        rows.append({"eps": e, "cached clip frac": f(m0["clip_fraction"], 4), "cached affected frac": f(m0["affected_fraction"], 4),
                     "clipped surrogate": f(m0["clipped_surrogate"], 4), "fork max ratio": f(r.max_ratio, 3),
                     "fork mean affected frac": f(r.mean_affected_fraction, 4), "fork mean step-KL (k3)": f"{r.mean_step_kl_k3:.1e}",
                     "held-out RM ± SE": pm(r.heldout_reward, r.heldout_reward_se), "KL tok": f(r.heldout_kl_token, 5),
                     "tokens": f"{r.heldout_tokens_mean:.0f}"})
    t22 = table("t2_2_clipping", rows)

    rows, logs = [], {}
    for b in ["0", "0.1", "0.2"]:
        name = f"fork_eps0.2_kl{b}"
        r = s.loc[name]
        logs[b] = pd.DataFrame(read_jsonl(R / f"task2_ppo/{name}/train_log.jsonl"))
        d, lo, hi = paired(R / f"task2_ppo/{name}/heldout_generations.jsonl", base, "reward")
        rows.append({"beta_KL": b, "held-out RM ± SE": pm(r.heldout_reward, r.heldout_reward_se), "ΔRM vs midpoint [CI]": ci(d, lo, hi),
                     "KL tok (seq)": f"{r.heldout_kl_token:.5f} ({r.heldout_kl_seq:.3f})", "entropy": f(r.heldout_entropy),
                     "tokens mean ± sd": f"{r.heldout_tokens_mean:.0f} ± {r.heldout_tokens_std:.0f}", "trunc@768": f(r.heldout_truncation, 2),
                     "train tokens (realized)": int(r.tokens_generated)})
    t23 = table("t2_3_kl_pressure", rows)
    # first update at which beta=0 departs from beta=0.1, per logged quantity
    first = {}
    for col in ["reward", "kl_token_mean", "entropy", "response_tokens", "policy_grad_norm"]:
        dif = np.abs(logs["0"][col].to_numpy() - logs["0.1"][col].to_numpy())
        hit = np.where(dif > 1e-6 * (1 + np.abs(logs["0.1"][col].to_numpy())))[0]
        first[col] = int(hit[0]) + 1 if len(hit) else None
    t24 = table("t2_4_kl_first_divergence", [{"quantity": k, "first update β=0 ≠ β=0.1": v} for k, v in first.items()])

    rows = []
    for name in ["sft", "midpoint", "standard", "fork_eps0.2_kl0", "fork_eps0.2_kl0.2"]:
        g = read_jsonl(R / f"task2_ppo/{name}/heldout_generations.jsonl")
        rr = np.array([x["reward"] for x in g]); ll = np.array([x["response_tokens"] for x in g])
        rows.append({"condition": name, "held-out RM ± SE": pm(rr.mean(), rr.std() / np.sqrt(len(rr))),
                     "corr(RM, length)": f(float(np.corrcoef(rr, ll)[0, 1]), 2)})
    d, lo, hi = paired(R / "task2_ppo/standard/heldout_generations.jsonl", R / "task2_ppo/sft/heldout_generations.jsonl", "reward")
    t25 = table("t2_5_reward_informativeness", rows)
    return f"""## Task 2 — PPO (all runs start from the supplied midpoint policy + critic)

Standard continuation: 20 updates, 1 prompt/update, 2 PPO epochs; wall-clock {ts['wall_clock_s']} s, peak VRAM {ts['peak_vram_gib']} GiB ({ts['gpu']}); non-finite steps 0. Held-out: 64 prompts, 768-token cap, seeded sampling.

**T2.1 Standard trajectory (per-update values; 1 rollout per update, so single updates are noisy).** Figure: `results/task2_ppo/figures/standard_trajectories.png`.
{t21}

**T2.2 Clipping study (RQ1).** Cached batch: 32 rollouts, ratios of the supplied midpoint vs cached old log-probs. Forks: 8 updates, β_KL 0.10, matched prompts/seeds.
{t22}

**T2.3 KL-pressure study (RQ2, RQ3).** ε 0.20, 8 updates; equal updates/prompts/seeds/cap, realized tokens differ (EOS is policy-dependent).
{t23}

**T2.4 Which logged quantity departs first (β=0 vs β=0.1 fork, same seeds).** Figure: `results/task2_ppo/figures/kl_fork_trajectories.png`.
{t24}

**T2.5 Learned-reward informativeness (RQ3).** Standard − SFT on the same prompts: ΔRM {ci(d, lo, hi)}.
{t25}

*Qualitative:* `results/task2_ppo/qualitative_candidates.md` (largest reward gains/drops vs midpoint).
"""


# ----------------------------------------------------------------------------- Task 3
def task3():
    s = pd.read_csv(R / "task3_grpo/summary.csv").set_index("condition")
    log = pd.DataFrame(read_jsonl(R / "task3_grpo/standard/train_log.jsonl"))
    ts = jload("task3_grpo/standard/train_summary.json")
    # Updates whose policy-surrogate term carried signal: informative group and >=1 unmasked completion.
    eff = int(((log.uninformative_group_fraction < 1) & (log.loss_masked_fraction < 1)).sum())
    n_masked = int((log.loss_masked_fraction == 1).sum())
    n_uninf = int((log.uninformative_group_fraction == 1).sum())
    cols = [("reward", "learned reward"), ("kl_token_mean", "KL (token)"), ("group_reward_std", "within-group reward std"),
            ("uninformative_group_fraction", "uninformative group (0/1 per update)"), ("truncated_fraction", "truncated (masked) fraction"),
            ("policy_loss", "policy loss"), ("grad_norm", "grad norm"), ("entropy", "entropy"), ("response_tokens", "response tokens")]
    rows = [{"metric": lab, "mean over 20 updates": f(log[c].mean(), 4), "sd": f(log[c].std(ddof=0), 4),
             "min": f(log[c].min(), 4), "max": f(log[c].max(), 4)} for c, lab in cols]
    t31 = table("t3_1_standard_trajectory", rows)

    g = jload("task3_grpo/group_size_study.json")
    rows = []
    for name in ["learned_reward", "binarized_reward"]:
        for k, e in g[name]["by_k"].items():
            a, ins, bd = e["all"], e["instability"], e["by_difficulty"]
            rows.append({"reward": name.split("_")[0], "K": k, "groups": a["n_groups"], "informative rate": f(a["informative_rate"], 3),
                         "(random partitions)": f(a["informative_rate_random_partitions"], 3),
                         "mean within-group std": f(a["mean_within_group_std"], 3), "var(r − group mean)": f(a["var_centered_signal"], 3),
                         "adv. variance across regroupings": f(ins["mean_partition_variance_of_advantage"], 3),
                         "sign flip vs K=8": f(ins["sign_disagreement_with_k8"], 3),
                         "informative hard/medium/easy": "/".join(f(bd[b]["informative_rate"], 2) for b in ["hard", "medium", "easy"] if b in bd)})
    t32 = table("t3_2_group_size", rows)
    cut = g["learned_reward"]["difficulty"]["tertile_cutoffs"]

    n = jload("task3_grpo/normalization_study.json")
    base = R / "task3_grpo/midpoint/heldout_generations.jsonl"
    rows = []
    for lt in ["grpo", "dr_grpo"]:
        e, lc = n[lt], n[lt]["length_conditioned"]
        d, lo, hi = paired(R / f"task3_grpo/fork_{lt}/heldout_generations.jsonl", base, "reward")
        dt, tlo, thi = paired(R / f"task3_grpo/fork_{lt}/heldout_generations.jsonl", base, "response_tokens")
        rows.append({"normalizer": "1/T_k" if lt == "grpo" else "1/L_max", "held-out RM ± SE": pm(e["heldout_reward"], e["heldout_reward_se"]),
                     "ΔRM vs midpoint [CI]": ci(d, lo, hi), "KL tok": f(e["heldout_kl_token"], 5),
                     "tokens (Δ vs midpoint [CI])": f"{e['heldout_tokens']['mean']:.0f} ({ci(dt, tlo, thi, 1)})",
                     "token weight short/long": f"{lc['mean_policy_token_weight_short']:.4f}/{lc['mean_policy_token_weight_long']:.4f}",
                     "corr(grad/|A|, length)": f(lc["corr_grad_per_abs_adv_vs_length"], 2),
                     "long share: grad / tokens": f"{lc['share_of_grad_norm_from_long']:.2f} / {lc['share_of_tokens_from_long']:.2f}",
                     "n completions": lc["n_completions"], "train tokens (realized)": e["tokens_generated"]})
    t33 = table("t3_3_normalization", rows)
    rows = []
    for name in ["sft", "midpoint", "standard"]:
        r = s.loc[name]
        rows.append({"condition": name, "held-out RM ± SE": pm(r.heldout_reward, r.heldout_reward_se),
                     "ΔRM vs midpoint [CI]": "-" if name == "midpoint" else ci(r.reward_diff_vs_midpoint, r.reward_diff_ci_lo, r.reward_diff_ci_hi),
                     "KL tok": f(r.heldout_kl_token, 5), "entropy": f(r.heldout_entropy), "tokens": f"{r.heldout_tokens_mean:.0f}",
                     "trunc@768": f(r.heldout_truncation, 2)})
    t34 = table("t3_4_standard_heldout", rows)
    return f"""## Task 3 — GRPO (all runs start from the supplied GRPO midpoint)

Standard continuation: 20 updates, K=4, 1 prompt/update, max-length completions masked; wall-clock {ts['wall_clock_s']} s, peak VRAM {ts['peak_vram_gib']} GiB; tokens generated {ts['tokens_generated']}; updates with a usable advantage signal {eff}/20 (uninformative group: {n_uninf}; all completions masked: {n_masked}).

**T3.1 Standard trajectory.** Figure: `results/task3_grpo/figures/standard_trajectories.png`.
{t31}

**T3.4 Standard held-out (64 prompts, 768 cap; same protocol as Task 2).**
{t34}

**T3.2 Equal-generation group-size study (RQ1).** 24 prompts × 8 cached completions = 192 generations for every K. Difficulty bins = tertiles of a prompt's mean cached reward (cut-offs {cut[0]:.2f}, {cut[1]:.2f}). Binarized = reward > median (diagnostic).
{t32}

**T3.3 Canonical GRPO vs Dr. GRPO (RQ2).** 8-update forks, identical prompts/seeds/K/β/ε/cap.
{t33}

*Qualitative:* `results/task3_grpo/qualitative_candidates.md` (uninformative group example; length-differing fork pairs).
"""


# ----------------------------------------------------------------------------- Task 4
REFUSE = {"JUSTIFIED_REFUSAL", "OVER_REFUSAL"}


def task4():
    m = jload("task4_safety/safety_metrics.json")
    rows, rows2 = [], []
    sft = {r["xstest_id"]: r["response"] for r in read_jsonl(R / "task4_safety/generated_sft.jsonl")}
    for p, v in m["policies"].items():
        rows.append({"policy": p, "safe answer": pm(v["safe_answer_rate"]["rate"], v["safe_answer_rate"]["se"]),
                     "over-refusal": pm(v["safe_over_refusal_rate"]["rate"], v["safe_over_refusal_rate"]["se"]),
                     "unsafe compliance": pm(v["unsafe_compliance_rate"]["rate"], v["unsafe_compliance_rate"]["se"]),
                     "justified refusal": pm(v["unsafe_justified_refusal_rate"]["rate"], v["unsafe_justified_refusal_rate"]["se"]),
                     "ambiguous": f(v["ambiguous_rate"]["rate"]), "class-inconsistent label": f(v["class_inconsistent_label_rate"]["rate"]),
                     "tokens mean (safe/unsafe)": f"{v['response_tokens_mean']:.0f} ({v['response_tokens_mean_safe']:.0f}/{v['response_tokens_mean_unsafe']:.0f})"})
        j = pd.DataFrame(read_jsonl(R / f"task4_safety/judged_{p}.jsonl"))
        same = sum(sft[i] == t for i, t in zip(j.xstest_id, j.response))
        safe, uns = j[j.benchmark_class == "SAFE"], j[j.benchmark_class == "UNSAFE"]
        rows2.append({"policy": p, "judge 'refusal' label on SAFE": f(safe.ai_label.isin(REFUSE).mean()),
                      "judge 'answer' label on UNSAFE": f(uns.ai_label.isin({"SAFE_ANSWER", "UNSAFE_COMPLIANCE"}).mean()),
                      "responses identical to SFT": f"{same}/450"})
    t41 = table("t4_1_safety_rates", rows)
    t42 = table("t4_2_refusal_view_supplementary", rows2)
    cat = pd.read_csv(R / "task4_safety/category_labels.csv")
    cat["refusal-type label"] = cat["JUSTIFIED_REFUSAL"] + cat["OVER_REFUSAL"]
    piv = cat.pivot_table(index=["benchmark_class", "type"], columns="policy", values="refusal-type label", sort=False).reset_index()
    piv = piv[["benchmark_class", "type"] + [p for p in ["sft", "dpo", "ppo", "grpo"] if p in piv]]
    for p in ["sft", "dpo", "ppo", "grpo"]:
        piv[p] = piv[p].map(lambda x: f"{x:.2f}")
    t43 = table("t4_3_category_refusal_labels", piv.to_dict(orient="records"))
    a = m.get("manual_audit") or {}
    if "agreement" in a:
        conf = pd.DataFrame(a["confusion_manual_rows_ai_cols"]).T
        audit = (f"Labeled {a['n_labeled']}/{a['n_rows']}; agreement {a['agreement']:.3f}; Cohen's κ {a['cohen_kappa']:.3f}; "
                 f"manual ambiguous {a['manual_ambiguous_rate']:.3f} vs AI {a['ai_ambiguous_rate_on_audit']:.3f}.\n\n"
                 + md_table(conf.reset_index().rename(columns={"index": "manual \\ AI"})))
        conf.to_csv(OUT / "t4_4_audit_confusion.csv")
    else:
        audit = "**PENDING** — fill `results/task4_safety/manual_audit_sheet.csv`, then rerun `python -m task4_safety.evaluate_safety` and this script."
    return f"""## Task 4 — Safety calibration (XSTest: 250 safe, 200 unsafe; greedy, 256 cap; fixed 3B categorical judge)

**T4.1 Required rates (judge labels; rates conditional on benchmark class, ± binomial SE).**
{t41}

**T4.2 Supplementary refusal view (judge label collapsed to refuse/answer, conditioned on benchmark class; inherits judge errors).**
{t42}

**T4.3 Category-level: share of refusal-type labels (JUSTIFIED_REFUSAL + OVER_REFUSAL) per XSTest type.** Full 5-label distribution: `results/task4_safety/category_labels.csv`; figure `results/task4_safety/figures/category_labels.png`.
{t43}

**T4.4 Manual audit vs judge.** {audit}
"""


# ----------------------------------------------------------------------------- Task 5
def task5():
    p = pd.DataFrame(jload("task5_feedback/summary.json")["policies"]).set_index("policy")
    d = jload("task5_feedback/diagnostics/diagnostic_metrics.json")
    rows = []
    for pol in ["sft", "rlvr", "rlaif"]:
        r = p.loc[pol]
        row = {"policy": pol}
        for sp, n in [("gsm", 300), ("transfer", 100)]:
            row[f"{sp} exact acc ± SE"] = pm(r[f"{sp}_exact_acc"], se_binom(r[f"{sp}_exact_acc"], n))
            row[f"{sp} format"] = f(r[f"{sp}_format"])
            row[f"{sp} tokens"] = f"{r[f'{sp}_tokens_mean']:.0f}"
            row[f"{sp} win vs SFT (W/T/L)"] = "0.500 (ref.)" if pol == "sft" else f"{r[f'{sp}_win_rate_vs_sft']:.3f} ({r[f'{sp}_judge_W_T_L']})"
        row["acc gsm − transfer (neg. = higher on SVAMP)"] = f(r["exact_acc_drop"])
        if pol != "sft":
            same_g = identical(R / f"task5_feedback/gsm/generations_{pol}.jsonl", R / "task5_feedback/gsm/generations_sft.jsonl", id_key="id")
            row["identical to SFT (gsm)"] = f"{same_g[0]}/{same_g[1]}"
            row["judge=verifier on decisive (n)"] = f"{r['gsm_judge_agrees_on_verifier_decisive']:.3f} ({int(r['gsm_n_verifier_decisive'])})"
        else:
            row["identical to SFT (gsm)"] = "-"
            row["judge=verifier on decisive (n)"] = "-"
        rows.append(row)
    t51 = table("t5_1_in_and_out_of_domain", rows)
    rows = [{"policy": pol, **{f"{sp} {ft}": int(p.loc[pol][f"{sp}_{ft}"]) for sp in ["gsm", "transfer"]
                                for ft in ["correct", "wrong_final", "missing_final_format", "truncated_no_final"]}} for pol in ["sft", "rlvr", "rlaif"]]
    t52 = table("t5_2_failure_types", rows)
    rows = []
    for worse, e in d["by_perturbation"].items():
        rows.append({"pair (clean_correct vs …)": worse, "changes": e["kind"],
                     "verifier better/tie/wrong": "/".join(f(e["verifier"][k], 2) for k in ["better", "tie", "wrong"]),
                     "judge better/tie/wrong": "/".join(f(e["judge"][k], 2) for k in ["better", "tie", "wrong"]),
                     "judge order-consistency": f(e["judge_order_consistency"], 2)})
    t53 = table("t5_3_controlled_diagnostics", rows)
    rr = d["judge_round_robin_win_rate_by_variant"]; pv = d["pointwise_verifier_reward_by_variant"]
    t54 = table("t5_4_variant_scores", [{"variant": v, "verifier reward": f(pv[v], 2), "judge round-robin win rate": f(rr[v], 3)} for v in rr])
    det = jload("task5_feedback/judge_cache.details.json")
    echo = sum("," in x["raw_output"] for x in det.values())
    return f"""## Task 5 — RLVR vs RLAIF (supplied adapters; greedy; 512 cap; verifier = last `#### n`)

**T5.1 In-domain (GSM8K, n=300) and out-of-domain (SVAMP, n=100).** Win rate = judge(policy vs SFT): win 1, tie 0.5, loss 0.
{t51}

**T5.2 Failure types (counts).**
{t52}

**T5.3 Controlled reward diagnostics (20 problems; each pair judged in both orders).** S_reason: verifier {d['S_reason']['verifier']:.2f}, judge {d['S_reason']['judge']:.2f}. S_outcome: verifier {d['S_outcome']['verifier']:.2f}, judge {d['S_outcome']['judge']:.3f} (good-reasoning-wrong-final {d['S_outcome']['judge_good_reasoning_wrong_final_only']:.2f}, gold-distractor {d['S_outcome']['judge_gold_distractor_only']:.2f}).
{t53}

**T5.4 Per-variant scores (all five categories).**
{t54}

*Cost:* judge {d['inference_cost']['judge_seconds_per_new_call']:.2f} s/call on a T4 (3B, 4-bit) vs negligible verifier; direct RLAIF needs K(K−1)/2 = 6 judge calls per K=4 group. *Judge parse caveat:* {echo}/{len(det)} raw judge outputs echoed the option list (e.g. "A, B"); the fixed parser takes the first letter. *Qualitative:* `results/task5_feedback/qualitative_candidates.md`.
"""


# ----------------------------------------------------------------------------- guide
GUIDE = """# PA2 evidence pack

Generated by `python -m task6_synthesis.build_evidence` from saved results only. Every number below is copied
or recomputed from a file under `results/`; CSV versions of each table are next to this file. **Interpretation and
report wording are the author's.** "Facts" bullets state what the numbers show, not why.

## How to read the metrics (one line each)
| Metric | Definition | Read as / caveat |
|---|---|---|
| Preference accuracy | share of held-out pairs with m = [log π/π_ref](y+) − [log π/π_ref](y−) > 0 | 0.5 ≈ no reference-relative preference; SFT is 0 by definition (m = 0) |
| DPO loss | −log σ(β m) | depends on β; never compare across β |
| Log-ratio chosen/rejected | mean log π − log π_ref for y+ / y− | both negative = both pushed down, rejected more |
| KL (token / seq) | mean over sampled response tokens of log π − log π_ref (course estimator); seq = summed per response | drift from the reference; single-sample estimate can be slightly negative |
| RM score | fixed reward-model scalar; Δ vs baseline paired on prompts, bootstrap 95% CI | only within one protocol; CI containing 0 = no demonstrated change |
| Truncation @cap | share of generations hitting the token cap | high truncation censors length comparisons |
| Clip fraction / affected fraction | share of tokens with ratio outside [1−ε, 1+ε] / where the clipped branch is active (gradient zeroed) | affected ≤ clip fraction; only affected tokens are actually constrained |
| Step-KL (k3) | E[r − 1 − log r], r = π_new/π_old on the update's own tokens | size of one update; stability statistic |
| Critic explained variance | 1 − Var(G − V)/Var(G) | < 0: the critic predicts returns worse than a constant |
| Informative group | within-group reward std > 1e-6 | uninformative group ⇒ all advantages 0 ⇒ no gradient |
| Advantage variance across regroupings | variance of one completion's normalized advantage when its K−1 group-mates are resampled | baseline noise of the group-relative signal |
| Token weight 1/T_k vs 1/L_max | per-token surrogate weight a completion receives | isolates the normalizer; grad/|A| also contains KL/content |
| Safety rates | P(label \\| benchmark class), judge labels | read with the class-inconsistent rate and the manual audit |
| Exact accuracy / format | verifier match of the last `#### n` / share with a parsable `#### n` | accuracy ≤ format; format failures count as wrong |
| Win rate vs SFT | judge pairwise, win 1 / tie 0.5 / loss 0 | ties include identical responses |
| S_reason / S_outcome | P(R(better) > R(worse)) on reasoning-only / outcome-changing pairs | verifier ties on reasoning-only pairs by design |
"""


def facts():
    """Measured facts with exact numbers (no interpretation)."""
    t1 = pd.read_csv(R / "task1_dpo/summary.csv").set_index("condition")
    t2 = pd.read_csv(R / "task2_ppo/summary.csv").set_index("condition")
    t3 = pd.read_csv(R / "task3_grpo/summary.csv").set_index("condition")
    s4 = jload("task4_safety/safety_metrics.json")["policies"]
    p5 = pd.DataFrame(jload("task5_feedback/summary.json")["policies"]).set_index("policy")
    lm = jload("task1_dpo/length_mechanism.json")
    la = jload("task1_dpo/length_analysis.json")["dataset_length_profile"]
    n3 = jload("task3_grpo/normalization_study.json")
    g3 = jload("task3_grpo/group_size_study.json")
    d5 = jload("task5_feedback/diagnostics/diagnostic_metrics.json")
    L = ["## Facts per research question (numbers only; see tables)"]
    L += ["### Task 1",
          f"- RQ1: held-out accuracy for β 0.03/0.10/0.30 = {t1.loc['beta_0.03'].heldout_pref_acc:.3f}/{t1.loc['beta_0.1'].heldout_pref_acc:.3f}/{t1.loc['beta_0.3'].heldout_pref_acc:.3f} (SE ≈ 0.028); margins {t1.loc['beta_0.03'].heldout_margin_mean:.3f}/{t1.loc['beta_0.1'].heldout_margin_mean:.3f}/{t1.loc['beta_0.3'].heldout_margin_mean:.3f}; KL tok {t1.loc['beta_0.03'].kl_token_mean:.5f}/{t1.loc['beta_0.1'].kl_token_mean:.5f}/{t1.loc['beta_0.3'].kl_token_mean:.5f}; all ΔRM-vs-SFT CIs contain 0 (T1.1). Pre-clip gradient norm scales with β and is clipped for β ≥ 0.1; max β·m ≤ 0.04 (T1.2).",
          f"- RQ2: data: 'longer wins' accuracy {la['standard_train']['longer_is_better_accuracy']:.3f} (standard train) vs {la['length_balanced_train']['longer_is_better_accuracy']:.3f} (balanced). Model: accuracy rises from preferred-longer to rejected-longer in both models; gap CIs exclude 0; per-token margins reverse the ordering; balanced − standard differs clearly only on rejected-longer ({lm['length_balanced_minus_standard']['rejected_longer']['accuracy_diff']:+.3f}, {lm['length_balanced_minus_standard']['rejected_longer']['discordant_pairs']} discordant pairs). Generated length and word-limit compliance ≈ SFT (T1.1, T1.3).",
          "- RQ3: preference accuracy 0.637 with ΔRM vs SFT ≈ 0 and ΔTokens ≈ 0; word-limit candidates where RM rises while the limit is violated more (qualitative file).",
          "### Task 2",
          f"- RQ1: cached clip fraction 0.154/0.041/0.021 and affected 0.076/0.020/0.010 for ε 0.05/0.20/0.50; in the forks max ratio ≤ {t2.loc[['fork_eps0.05_kl0.1','fork_eps0.2_kl0.1','fork_eps0.5_kl0.1']].max_ratio.max():.3f} and the three ε forks produce identical held-out results ({t2.loc['fork_eps0.2_kl0.1'].heldout_reward:.3f}).",
          f"- RQ2: held-out RM β 0/0.1/0.2 = {t2.loc['fork_eps0.2_kl0'].heldout_reward:.3f}/{t2.loc['fork_eps0.2_kl0.1'].heldout_reward:.3f}/{t2.loc['fork_eps0.2_kl0.2'].heldout_reward:.3f} (SE ≈ 0.15); KL tok ≈ 0 for all; first-divergence update per quantity in T2.4.",
          f"- RQ3: held-out RM SFT {t2.loc['sft'].heldout_reward:.3f} > standard {t2.loc['standard'].heldout_reward:.3f} > midpoint {t2.loc['midpoint'].heldout_reward:.3f}; critic EV < 0 throughout (T2.1, T2.5).",
          "### Task 3",
          f"- RQ1: informative rate K=2/4/8 = {g3['learned_reward']['by_k']['2']['all']['informative_rate']:.3f}/{g3['learned_reward']['by_k']['4']['all']['informative_rate']:.3f}/{g3['learned_reward']['by_k']['8']['all']['informative_rate']:.3f} (learned reward) vs {g3['binarized_reward']['by_k']['2']['all']['informative_rate']:.3f}/{g3['binarized_reward']['by_k']['4']['all']['informative_rate']:.3f}/{g3['binarized_reward']['by_k']['8']['all']['informative_rate']:.3f} (binarized); hard bin lowest in both (T3.2).",
          f"- RQ2: corr(grad/|A|, length) {n3['grpo']['length_conditioned']['corr_grad_per_abs_adv_vs_length']:+.2f} (1/T_k) vs {n3['dr_grpo']['length_conditioned']['corr_grad_per_abs_adv_vs_length']:+.2f} (1/L_max); long completions' share of gradient {n3['grpo']['length_conditioned']['share_of_grad_norm_from_long']:.2f} vs {n3['dr_grpo']['length_conditioned']['share_of_grad_norm_from_long']:.2f} (tokens {n3['grpo']['length_conditioned']['share_of_tokens_from_long']:.2f}); held-out RM {n3['grpo']['heldout_reward']:.3f} vs {n3['dr_grpo']['heldout_reward']:.3f}, tokens {n3['grpo']['heldout_tokens']['mean']:.0f} vs {n3['dr_grpo']['heldout_tokens']['mean']:.0f} (T3.3).",
          f"- RQ3: standard run: uninformative updates {int((pd.DataFrame(read_jsonl(R / 'task3_grpo/standard/train_log.jsonl')).uninformative_group_fraction == 1).sum())}/20, mean truncated(masked) fraction {pd.DataFrame(read_jsonl(R / 'task3_grpo/standard/train_log.jsonl')).truncated_fraction.mean():.2f}; held-out RM SFT {t3.loc['sft'].heldout_reward:.3f}, midpoint {t3.loc['midpoint'].heldout_reward:.3f}, standard {t3.loc['standard'].heldout_reward:.3f} (T3.1, T3.4).",
          "### Task 4",
          f"- RQ1/RQ2: safe-answer {', '.join(f'{p} {v['safe_answer_rate']['rate']:.3f}' for p, v in s4.items())}; over-refusal 0.000 and unsafe-compliance ≈ 0 for all; class-inconsistent labels {', '.join(f'{v['class_inconsistent_label_rate']['rate']:.3f}' for v in s4.values())} (T4.1). Refusal-type labels on SAFE prompts ≈ 0.41–0.43 for every policy (T4.2); highest categories in T4.3.",
          "- RQ3: judge never emits OVER_REFUSAL or AMBIGUOUS; confidence is 1.0 for every label; manual-audit agreement in T4.4.",
          "### Task 5",
          f"- RQ1/RQ2: S_reason verifier 0.00 (tie by design) vs judge {d5['S_reason']['judge']:.2f}; filler pair: judge prefers the filler response in {d5['by_perturbation']['persuasive_filler_correct']['judge']['wrong']:.2f} of pairs; filler variant has the highest round-robin win rate ({d5['judge_round_robin_win_rate_by_variant']['persuasive_filler_correct']:.3f}) (T5.3, T5.4).",
          f"- RQ3: S_outcome verifier 1.00 vs judge {d5['S_outcome']['judge']:.3f}; judge prefers the wrong-final response in {d5['by_perturbation']['gold_distractor_wrong_final']['judge']['wrong']:.2f} (gold-distractor) and {d5['by_perturbation']['good_reasoning_wrong_final']['judge']['wrong']:.2f} (good-reasoning) of pairs (T5.3).",
          f"- RQ4: exact acc gsm→transfer SFT {p5.loc['sft'].gsm_exact_acc:.3f}→{p5.loc['sft'].transfer_exact_acc:.3f}, RLVR {p5.loc['rlvr'].gsm_exact_acc:.3f}→{p5.loc['rlvr'].transfer_exact_acc:.3f}, RLAIF {p5.loc['rlaif'].gsm_exact_acc:.3f}→{p5.loc['rlaif'].transfer_exact_acc:.3f}; format compliance gsm {p5.loc['sft'].gsm_format:.2f}/{p5.loc['rlvr'].gsm_format:.2f}/{p5.loc['rlaif'].gsm_format:.2f}; most RLVR/RLAIF responses are identical to SFT (T5.1)."]
    return "\n".join(L)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = [GUIDE, facts(), task1(), task2(), task3(), task4(), task5()]
    syn = R / "task6_synthesis/drift_reward_compute.csv"
    if syn.exists():
        d = pd.read_csv(syn).drop(columns=["training_signal", "eval_protocol"], errors="ignore")
        parts.append("## Task 6 — cross-task table\n`rm_reward_vs_start`: DPO vs SFT (Task 1 protocol); PPO/GRPO vs their "
                     "supplied midpoint (Task 2/3 protocol). RM values are comparable only within one protocol.\n\n"
                     + md_table(d.round(4).fillna("-")))
    (OUT / "EVIDENCE.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"Wrote {OUT / 'EVIDENCE.md'} and {len(TABLES)} CSV tables")


if __name__ == "__main__":
    main()
