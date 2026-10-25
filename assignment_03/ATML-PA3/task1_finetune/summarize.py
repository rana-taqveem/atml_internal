"""Task 1 summary: tables (CSV + LaTeX), figures and a fact sheet for the report.

Reads only saved result files, so every reported number traces to a file in results/task1.

    python -m task1_finetune.summarize
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.plotting import T1_COLORS, T1_LABELS, TEXT_WIDTH, save, setup
from common.store import Store
from common.utils import ensure_dir, load_config, read_json, read_jsonl, rel

OUT = "results/task1/summary"
FIG = "report/figures/task1"
VARIANTS = ["full", "l2sp", "lora4", "lora16", "oplora16"]
BENCH = [("gsm8k", "flexible_extract", "GSM8K"), ("hellaswag", "acc_norm", "HellaSwag"),
         ("boolq", "acc", "BoolQ"), ("arc_easy", "acc_norm", "ARC-Easy")]


def benchmark_table(cfg):
    rows = []
    for cond in cfg["eval"]["conditions"]:
        p = rel(f"results/task1/eval/{cond}/scores.json")
        if not p.exists():
            continue
        s = read_json(p)
        r = {"condition": cond, "label": T1_LABELS[cond]}
        for task, metric, _ in BENCH:
            r[task] = 100 * s[task][metric]
        r["gsm8k_strict"] = 100 * s["gsm8k"]["strict_match"]
        r["hellaswag_acc"] = 100 * s["hellaswag"]["acc"]
        r["arc_easy_acc"] = 100 * s["arc_easy"]["acc"]
        r["gsm8k_truncation_rate"] = 100 * s["gsm8k"]["truncation_rate"]
        r["gsm8k_extractable_rate"] = 100 * s["gsm8k"]["extractable_rate"]
        r["gsm8k_mean_tokens"] = s["gsm8k"]["mean_response_tokens"]
        r["merge_max_abs_diff_fp32"] = (s.get("merge_check", {}).get("fp32", {}) or {}).get("max_abs_diff")
        rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty or "baseline" not in set(df["condition"]):
        return df
    base = df.set_index("condition").loc["baseline"]
    for task, _, _ in BENCH:
        df[f"{task}_delta_pp"] = df[task] - base[task]
    nonmath = [t for t, _, _ in BENCH[1:]]
    df["nonmath_mean_delta_pp"] = df[[f"{t}_delta_pp" for t in nonmath]].mean(axis=1)
    return df


def gsm_transitions(cfg):
    """Per-question GSM8K changes vs baseline: correctness flips and extraction/truncation."""
    base = {r["doc_id"]: r for r in read_jsonl("results/task1/eval/baseline/samples_gsm8k.jsonl")}
    rows = []
    for cond in cfg["eval"]["conditions"]:
        if cond == "baseline":
            continue
        cur = read_jsonl(f"results/task1/eval/{cond}/samples_gsm8k.jsonl")
        if not cur or not base:
            continue
        c = {"condition": cond, "wrong_to_right": 0, "right_to_wrong": 0}
        for r in cur:
            b = base[r["doc_id"]]
            bc, cc = b["correct_flexible-extract"], r["correct_flexible-extract"]
            c["wrong_to_right"] += int(cc > bc)
            c["right_to_wrong"] += int(cc < bc)
        rows.append(c)
    return pd.DataFrame(rows)


def training_tables():
    res, logs = [], {}
    for v in VARIANTS:
        p = rel(f"results/task1/train/{v}/summary.json")
        if not p.exists():
            continue
        s = read_json(p)
        logs[v] = pd.DataFrame(read_jsonl(f"results/task1/train/{v}/log.jsonl"))
        sizes = s.get("checkpoint_sizes", {})
        res.append({
            "variant": v, "label": T1_LABELS[v], "trainable_params": s["trainable_params"],
            "train_hours": s["train_hours"], "peak_gpu_gb": s.get("peak_allocated_gb"),
            "final_weights_mb": sizes.get("weights_final", np.nan) / 1e6,
            "recovery_ckpt_mb": sizes.get("recovery", np.nan) / 1e6,
            "skipped_updates": s.get("skipped_updates"),
            "displacement_final": s.get("displacement", {}).get("final"),
            "displacement_mid": s.get("displacement", {}).get("u0625"),
            "l2sp_reference_mb": s.get("l2sp_reference_bytes", np.nan) / 1e6,
            "oplora_basis_mb": s.get("basis_bytes", np.nan) / 1e6 if v == "oplora16" else np.nan,
            "oplora_basis_seconds": s.get("basis_seconds") if v == "oplora16" else np.nan,
            "final_loss_sft_last50": logs[v]["loss_sft"].tail(50).mean(),
        })
    return pd.DataFrame(res), logs


def plot_losses(logs):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.55, 2.2))
    for v, df in logs.items():
        sm = df["loss_sft"].rolling(25, min_periods=1).mean()
        ax.plot(df["update"], df["loss_sft"], color=T1_COLORS[v], alpha=0.18, lw=0.6)
        ax.plot(df["update"], sm, color=T1_COLORS[v], label=T1_LABELS[v])
    ax.set_xlabel("Optimizer update")
    ax.set_ylabel("SFT loss (nats/token)")
    ax.legend(ncol=1)
    save(fig, f"{FIG}/sft_loss")


def plot_l2sp(logs):
    import matplotlib.pyplot as plt

    if "l2sp" not in logs:
        return
    df = logs["l2sp"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 1.9))
    axes[0].plot(df["update"], df["omega"], color=T1_COLORS["l2sp"])
    axes[0].set_xlabel("Optimizer update"); axes[0].set_ylabel(r"$\Omega(\theta,\theta_0)$")
    axes[1].plot(df["update"], df["loss_sft"].rolling(25, min_periods=1).mean(),
                 color=T1_COLORS["l2sp"], label="SFT loss")
    axes[1].plot(df["update"], df["lambda_omega"], color=T1_COLORS["l2sp"], ls="--",
                 label=r"$\lambda\Omega$")
    axes[1].set_xlabel("Optimizer update"); axes[1].set_ylabel("nats/token")
    axes[1].legend()
    save(fig, f"{FIG}/l2sp_penalty")


def plot_diagnostics():
    import matplotlib.pyplot as plt

    w = rel("results/task1/diagnostics/weight_space.csv")
    a = rel("results/task1/diagnostics/activation_summary.csv")
    if not (w.exists() and a.exists()):
        return
    wdf, adf = pd.read_csv(w), pd.read_csv(a)
    conds = [c for c in T1_LABELS if c in set(wdf["condition"])]
    layers = sorted(wdf["layer"].unique())
    from matplotlib.ticker import NullFormatter

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.1), gridspec_kw={"wspace": 0.6})
    x = np.arange(len(layers))
    width = 0.8 / len(conds)
    for i, c in enumerate(conds):
        d = wdf[wdf["condition"] == c].set_index("layer").loc[layers]
        axes[0].bar(x + i * width, d["rho_k"], width * 0.92, color=T1_COLORS[c], label=T1_LABELS[c])
        axes[1].bar(x + i * width, d["eta"], width * 0.92, color=T1_COLORS[c])
        m = adf[(adf["condition"] == c) & (adf["set"] == "math")].set_index("layer").loc[layers]["mean"]
        n = adf[(adf["condition"] == c) & (adf["set"] == "nonmath")].set_index("layer").loc[layers]["mean"]
        axes[2].scatter(n, m, color=T1_COLORS[c], s=14, zorder=3)
    for ax, t in zip(axes[:2], [r"$\rho_k$ (k=128)", r"$\eta=\|\Delta W\|_F/\|W_0\|_F$"]):
        ax.set_xticks(x + 0.4 - width / 2, [f"L{l}" for l in layers]); ax.set_title(t)
    axes[1].set_yscale("log")
    lim = [min(axes[2].get_xlim()[0], axes[2].get_ylim()[0]), max(axes[2].get_xlim()[1], axes[2].get_ylim()[1])]
    axes[2].plot(lim, lim, color="#bbb", lw=0.8, ls=":")
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    for ax in axes[1:]:
        ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    axes[2].set_xlabel("mean R(z), HellaSwag"); axes[2].set_ylabel("mean R(z), GSM8K")
    axes[2].set_title("Mean R(z) per layer")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 1.12))
    save(fig, f"{FIG}/diagnostics")


def plot_benchmarks(df):
    import matplotlib.pyplot as plt

    if df.empty or "gsm8k_delta_pp" not in df:
        return
    d = df[df["condition"] != "baseline"]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.45, 2.2))
    for _, r in d.iterrows():
        ax.scatter(r["nonmath_mean_delta_pp"], r["gsm8k_delta_pp"], color=T1_COLORS[r["condition"]],
                   s=22, zorder=3)
        ax.annotate(r["label"], (r["nonmath_mean_delta_pp"], r["gsm8k_delta_pp"]),
                    fontsize=6, xytext=(3, 2), textcoords="offset points", color="#52514e")
    ax.axhline(0, color="#bbb", lw=0.8); ax.axvline(0, color="#bbb", lw=0.8)
    ax.set_xlabel("Mean non-math change (pp)")
    ax.set_ylabel("GSM8K change (pp)")
    save(fig, f"{FIG}/tradeoff")


def latex_benchmarks(df):
    cols = ["label"] + [b[0] for b in BENCH]
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Condition & GSM8K & HellaSwag & BoolQ & ARC-Easy \\", r"\midrule"]
    for _, r in df.iterrows():
        cells = []
        for t, _, _ in BENCH:
            if r["condition"] == "baseline":
                cells.append(f"{r[t]:.2f}")
            else:
                cells.append(f"{r[t]:.2f} ({r[f'{t}_delta_pp']:+.2f})")
        lines.append(f"{r['label']} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def latex_resources(df):
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Method & Trainable params & Train time (h) & Peak mem. (GB) & Final ckpt (MB) \\",
             r"\midrule"]
    for _, r in df.iterrows():
        lines.append(f"{r['label']} & {r['trainable_params']:,} & {r['train_hours']:.2f} & "
                     f"{r['peak_gpu_gb']:.2f} & {r['final_weights_mb']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def run(cfg):
    setup()
    out = ensure_dir(OUT)
    store = Store()
    for p in ["results/task1/train", "results/task1/eval", "results/task1/diagnostics",
              "results/task1/qualitative", "results/task1/data"]:
        store.pull(p)
    bench = benchmark_table(cfg)
    bench.to_csv(out / "benchmarks.csv", index=False)
    trans = gsm_transitions(cfg)
    trans.to_csv(out / "gsm8k_transitions.csv", index=False)
    res, logs = training_tables()
    res.to_csv(out / "resources.csv", index=False)
    if not bench.empty and "gsm8k_delta_pp" in bench:
        (out / "benchmarks_table.tex").write_text(latex_benchmarks(bench), encoding="utf-8")
    if not res.empty:
        (out / "resources_table.tex").write_text(latex_resources(res), encoding="utf-8")
    if logs:
        plot_losses(logs)
        plot_l2sp(logs)
    plot_benchmarks(bench)
    plot_diagnostics()
    store.push(OUT, "task1 summary")
    pd.set_option("display.width", 200)
    print(bench.round(2).to_string()); print(res.round(3).to_string()); print(trans.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml")
    run(load_config(ap.parse_args().config))


if __name__ == "__main__":
    main()
