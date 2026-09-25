"""Create Task 4 score-distribution, ROC, and model-comparison figures."""

import argparse
import json
import os

import numpy as np

from assignment_01.figure_style import print_ready
from assignment_01.task4.config import task_config
from assignment_01.task4.evaluation.metrics import auroc

SCORES = ["msp", "mls", "mahalanobis"]
LABELS = {"msp": "MSP", "mls": "MLS", "mahalanobis": "Mahalanobis"}


def roc_curve(known_scores, unknown_scores):
    """False/true positive rates over all thresholds, unknown = positive.

    Written out rather than imported so the figure depends only on numpy.
    """
    known = np.asarray(known_scores, dtype=np.float64)
    unknown = np.asarray(unknown_scores, dtype=np.float64)

    thresholds = np.unique(np.concatenate([known, unknown]))[::-1]
    true_positive = np.empty(len(thresholds) + 2)
    false_positive = np.empty(len(thresholds) + 2)
    true_positive[0] = false_positive[0] = 0.0
    for index, threshold in enumerate(thresholds, start=1):
        true_positive[index] = (unknown >= threshold).mean()
        false_positive[index] = (known >= threshold).mean()
    true_positive[-1] = false_positive[-1] = 1.0
    return false_positive, true_positive


def model_comparison_figure(results_dir, dpi=200):
    """Plot closed-set accuracy against open-set performance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = os.path.join(results_dir, "osr_results.json")
    if not os.path.isfile(path):
        print(f"  (no {path}: skipping model-comparison figure)")
        return None
    with open(path, encoding="utf-8") as file:
        rows = json.load(file)["table2"]
    if not rows:
        return None

    labels = [f"{r['model'].upper()}\n({r['score']})" for r in rows]
    positions = np.arange(len(rows))
    colors = {"vanilla": "#4C72B0", "gcsc": "#DD8452", "proser": "#55A868"}

    figure, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))

    width = 0.2
    series = [("csa_pct", "CSA (%)", "#8FA8CF", 1.0),
              ("auroc_near", "AUROC near", "#DD8452", 100.0),
              ("auroc_far", "AUROC far", "#55A868", 100.0),
              ("auroc_all", "AUROC all", "#C44E52", 100.0)]
    for index, (key, label, color, scale) in enumerate(series):
        values = [r[key] * scale for r in rows]
        offset = (index - (len(series) - 1) / 2) * width
        bars = axes[0].bar(positions + offset, values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.8,
                         f"{value:.1f}", ha="center", fontsize=6.5)

    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("percent  /  AUROC x 100")
    axes[0].set_ylim(0, 108)
    axes[0].set_title("(a) Closed-set accuracy and open-set AUROC", fontsize=10.5)
    axes[0].legend(fontsize=7.5, ncol=2, loc="lower right")

    for row in rows:
        axes[1].scatter(row["auroc_near"], row["auroc_far"], s=95, zorder=3,
                        color=colors.get(row["model"], "grey"),
                        marker="o" if row["score"] == "mls" else "^",
                        label=f"{row['model']} ({row['score']})")
    low = min(min(r["auroc_near"] for r in rows), min(r["auroc_far"] for r in rows)) - 0.02
    high = max(max(r["auroc_near"] for r in rows), max(r["auroc_far"] for r in rows)) + 0.02
    axes[1].plot([low, high], [low, high], color="grey", linestyle=":", linewidth=1.0)
    axes[1].text(low + 0.005, low + 0.002, "equal difficulty", fontsize=7.5, color="grey")
    axes[1].set_xlabel("AUROC, known vs near unknown")
    axes[1].set_ylabel("AUROC, known vs far unknown")
    axes[1].set_title("(b) Near unknowns are harder for every model", fontsize=10.5)
    axes[1].legend(fontsize=7.5, loc="lower right")

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(alpha=0.25, linewidth=0.6)

    figure.tight_layout()
    output = os.path.join(results_dir, "figure_model_comparison.png")
    print_ready(figure)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    figure.savefig(output.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output}")
    return output


def run(results_dir=None, output_path=None, dpi=200):
    import matplotlib
    matplotlib.use("Agg")                       # no display on Colab/Kaggle
    import matplotlib.pyplot as plt

    results_dir = results_dir or task_config.TASK_RESULTS_DIR
    scores_path = os.path.join(results_dir, "figure_scores.json")
    if not os.path.isfile(scores_path):
        raise SystemExit(f"No {scores_path}. Run evaluate_osr.py first.")

    with open(scores_path, encoding="utf-8") as file:
        data = json.load(file)

    # Thresholds come from the saved table so the figure and the table cannot
    # disagree about where the operating point is.
    thresholds = {}
    table_path = os.path.join(results_dir, "osr_results.json")
    if os.path.isfile(table_path):
        with open(table_path, encoding="utf-8") as file:
            for row in json.load(file)["table1"]:
                thresholds[row["score"]] = row["threshold"]

    available = [name for name in SCORES if name in data]
    if not available:
        raise SystemExit(f"figure_scores.json has none of {SCORES}.")

    figure, axes = plt.subplots(2, len(available), figsize=(4.2 * len(available), 7.0))
    if len(available) == 1:
        axes = axes.reshape(2, 1)

    colors = {"test": "#4C72B0", "near": "#DD8452", "far": "#55A868"}
    names = {"test": "known (CIFAR-10 test)", "near": "near unknown", "far": "far unknown"}

    for column, score_name in enumerate(available):
        known = np.asarray(data[score_name]["test"])
        near = np.asarray(data[score_name]["near"])
        far = np.asarray(data[score_name]["far"])

        # Top: distributions on a shared range, trimmed to the 1st-99th
        # percentile so one outlier cannot flatten the whole panel.
        combined = np.concatenate([known, near, far])
        low, high = np.percentile(combined, [1, 99])
        bins = np.linspace(low, high, 60)

        top = axes[0, column]
        for split, values in (("test", known), ("near", near), ("far", far)):
            top.hist(values, bins=bins, density=True, alpha=0.55,
                     color=colors[split], label=names[split])
        if score_name in thresholds:
            top.axvline(thresholds[score_name], color="black", linestyle="--",
                        linewidth=1.2,
                        label=f"threshold ({task_config.ACCEPT_PERCENTILE:.0f}th pct)")
        top.set_title(LABELS[score_name])
        top.set_xlabel("unknownness  u(x)")
        top.set_ylabel("density" if column == 0 else "")
        if column == 0:
            top.legend(fontsize=7, loc="upper right")

        # Bottom: ROC, unknown treated as the positive class.
        bottom = axes[1, column]
        for split, values, color in (("near", near, colors["near"]),
                                     ("far", far, colors["far"])):
            false_positive, true_positive = roc_curve(known, values)
            bottom.plot(false_positive, true_positive, color=color, linewidth=1.6,
                        label=f"{split}  AUROC {auroc(known, values):.3f}")
        bottom.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1.0)
        bottom.set_xlabel("false positive rate (known accepted as unknown)")
        bottom.set_ylabel("true positive rate" if column == 0 else "")
        bottom.set_xlim(0, 1)
        bottom.set_ylim(0, 1)
        bottom.legend(fontsize=7, loc="lower right")

    figure.suptitle("Post-hoc novelty scores on the frozen Vanilla model", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    print_ready(figure, min_height_in=4.2)

    output_path = output_path or os.path.join(results_dir, "figure_osr_scores.png")
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    figure.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    run(results_dir=args.results_dir, output_path=args.output, dpi=args.dpi)
    model_comparison_figure(args.results_dir or task_config.TASK_RESULTS_DIR, dpi=args.dpi)


if __name__ == "__main__":
    main()
