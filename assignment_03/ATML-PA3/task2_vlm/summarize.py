"""Task 2 summary: learning curves, final tables (CSV + LaTeX), resources.

    python -m task2_vlm.summarize
"""
from __future__ import annotations

import argparse

import pandas as pd

from common.plotting import INK2, NEUTRAL, SERIES, TEXT_WIDTH, save, setup
from common.store import Store
from common.utils import ensure_dir, load_config, read_json, read_jsonl, rel, write_json
from task2_vlm.train import run_dirs, stage_bounds

COND_COLORS = {"correct": SERIES[0], "complementary": SERIES[1], "none": NEUTRAL, "mismatched": SERIES[2]}


def curves(cfg, rdir, fig_dir):
    import matplotlib.pyplot as plt

    log = pd.DataFrame(read_jsonl(f"{rdir}/log.jsonl"))
    dev = pd.DataFrame(read_jsonl(f"{rdir}/dev_log.jsonl"))
    if log.empty:
        return
    bounds = [b[1] for b in stage_bounds(cfg)][:-1]
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 3.4), gridspec_kw={"wspace": 0.3, "hspace": 0.6})
    axes = axes.flatten()
    axes[0].plot(log["update"], log["loss"], color=SERIES[0], alpha=0.2, lw=0.5)
    axes[0].plot(log["update"], log["loss"].rolling(50, min_periods=1).mean(), color=SERIES[0])
    axes[0].set_title("Training loss")
    axes[1].plot(dev["update"], dev["dev_caption_loss"], color=SERIES[0], marker="o", ms=3)
    axes[1].set_title("Dev caption loss")
    for ax, suite in zip(axes[2:], ("grounding", "vqa")):
        for cond in ("correct", "complementary", "none"):
            ax.plot(dev["update"], dev[f"{suite}_nll_{cond}"], color=COND_COLORS[cond], marker="o", ms=3,
                    label={"correct": "correct image", "complementary": "partner image", "none": "no image"}[cond])
        ax.set_title(f"Dev answer NLL ({'grounding' if suite == 'grounding' else 'VQAv2'})")
    fig.legend(*axes[3].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    for ax in axes:
        for b in bounds:
            ax.axvline(b, color=INK2, lw=0.6, ls=":")
        ax.set_xlabel("Update")
    save(fig, f"{fig_dir}/learning_curves")
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.4, 1.8))
    for suite, col in (("grounding", SERIES[0]), ("vqa", SERIES[1])):
        ax.plot(dev["update"], dev[f"{suite}_swap_penalty"], color=col, marker="o", ms=3,
                label="grounding" if suite == "grounding" else "VQAv2")
    for b in bounds:
        ax.axvline(b, color=INK2, lw=0.6, ls=":")
    ax.axhline(0, color="#bbb", lw=0.8)
    ax.set_xlabel("Update"); ax.set_ylabel(r"$\Delta_{NLL}$ (partner $-$ correct)")
    ax.legend(fontsize=6)
    save(fig, f"{fig_dir}/swap_penalty")


def latex_final(m: pd.DataFrame, boot: dict) -> str:
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r"Suite & Image & Acc. & Pair acc. & Identical & Empty & Ref. NLL & s/answer \\", r"\midrule"]
    for suite in ("grounding", "vqa"):
        for cond in ("correct", "complementary", "mismatched", "none"):
            r = m[(m["suite"] == suite) & (m["condition"] == cond)]
            if r.empty:
                continue
            r = r.iloc[0]
            b = boot.get(suite, {}).get(f"score_{cond}", {})
            acc = f"{100 * r['accuracy']:.1f} [{100 * b.get('lo', float('nan')):.1f}, {100 * b.get('hi', float('nan')):.1f}]"
            lines.append(f"{'Grounding' if suite == 'grounding' else 'VQAv2'} & {cond} & {acc} & "
                         f"{100 * r['pair_accuracy']:.1f} & {100 * r['identical_answer_rate']:.1f} & "
                         f"{100 * r['empty_answer_rate']:.1f} & {r['mean_ref_nll']:.3f} & {r['gen_seconds_per_answer']:.3f} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def run(cfg):
    setup()
    store = Store()
    sfx = cfg.get("run_suffix", "")
    _, rdir = run_dirs(cfg)
    edir = f"results/task2/eval{sfx}"
    for p in (rdir, edir, "results/task2/resources_feature_cache.json"):
        store.pull(p)
    out = ensure_dir(f"results/task2/summary{sfx}")
    fig_dir = f"report/figures/task2{sfx}"
    curves(cfg, rdir, fig_dir)
    m = pd.read_csv(rel(f"{edir}/metrics.csv"))
    boot = read_json(f"{edir}/bootstrap.json")["bootstrap"]
    m.to_csv(out / "final_metrics.csv", index=False)
    (out / "final_table.tex").write_text(latex_final(m, boot), encoding="utf-8")
    s = read_json(f"{rdir}/summary.json")
    fc = read_json("results/task2/resources_feature_cache.json") if rel("results/task2/resources_feature_cache.json").exists() else {}
    res = {"train_hours": s["train_hours"], "stage_seconds": s["stage_seconds"],
           "stage_peak_gb": s["stage_peak_gb"], "trainable_by_stage": s["trainable_by_stage"],
           "feature_cache": fc, "gen_seconds_per_answer": m.groupby("suite")["gen_seconds_per_answer"].mean().to_dict(),
           "correct_image_advantage": {k: {kk: v[kk] for kk in ("score_advantage", "nll_advantage")} for k, v in boot.items()}}
    write_json(res, out / "resources.json")
    store.push(out, "task2 summary")
    print(m.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2.yaml")
    run(load_config(ap.parse_args().config))


if __name__ == "__main__":
    main()
