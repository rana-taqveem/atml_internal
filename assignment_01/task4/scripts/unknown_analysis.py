"""Per-unknown-class rejection behaviour, for the semantic-similarity question.

The failure table lists a handful of individual mistakes. This answers the
wider version of the same question: across all sixteen CIFAR-100 classes used
as unknowns, which ones does the model accept as known, and which CIFAR-10
label absorbs them.

Two artifacts:

  unknown_class_analysis.csv   one row per unknown class: acceptance rate under
                               the validation-calibrated threshold, mean
                               unknownness, the CIFAR-10 class that absorbs the
                               most of its images, and how many
  figure_unknown_classes.png   the same result as a figure, near and far shown
                               together and ordered by acceptance rate

Reads the cached outputs written by extract_outputs.py, so no model is loaded
and every number matches the tables produced by evaluate_osr.py.

    python -m assignment_01.task4.scripts.unknown_analysis --method vanilla
"""

import argparse
import collections
import csv
import os

import numpy as np

from assignment_01.task4.config import task_config
from assignment_01.task4.evaluation.metrics import calibrate_threshold
from assignment_01.task4.scores.novelty import compute_scores, fit_mahalanobis
from assignment_01.task4.scripts.extract_outputs import cache_path


def per_class_rows(cache, method, score_name="mls"):
    """Acceptance rate and dominant absorbing class for each unknown class."""
    known_columns = (slice(None), slice(0, task_config.NUM_CLASSES))
    fitted = fit_mahalanobis(cache["train_features"], cache["train_labels"])

    def score_of(split):
        logits = cache[f"{split}_logits"][known_columns]
        return compute_scores(logits, cache[f"{split}_features"],
                              fitted=fitted, names=[score_name])[score_name]

    # Threshold from CIFAR-10 validation only: no unknown influences it.
    threshold = calibrate_threshold(score_of("val"), task_config.ACCEPT_PERCENTILE)

    rows = []
    for group in ("near", "far"):
        scores = score_of(group)
        logits = cache[f"{group}_logits"][known_columns]
        predictions = logits.argmax(axis=1)
        names = cache[f"{group}_names"]

        for class_name in sorted(set(names.tolist())):
            mask = names == class_name
            accepted = scores[mask] <= threshold
            absorbed = collections.Counter(
                task_config.CIFAR10_CLASSES[p] for p in predictions[mask][accepted])
            top = absorbed.most_common(1)
            rows.append({
                "method": method,
                "group": group,
                "unknown_class": class_name,
                "n_images": int(mask.sum()),
                "accepted": int(accepted.sum()),
                "acceptance_rate_pct": 100.0 * accepted.mean(),
                "mean_unknownness": float(scores[mask].mean()),
                "absorbed_by": top[0][0] if top else "",
                "absorbed_count": top[0][1] if top else 0,
                "score_name": score_name,
                "threshold": float(threshold),
            })
    return rows


def figure(rows, results_dir, method, score_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda r: r["acceptance_rate_pct"], reverse=True)
    names = [r["unknown_class"] for r in ordered]
    values = [r["acceptance_rate_pct"] for r in ordered]
    colors = ["#DD8452" if r["group"] == "near" else "#55A868" for r in ordered]

    fig, axis = plt.subplots(figsize=(11.2, 4.6))
    bars = axis.bar(names, values, color=colors)

    for bar, row in zip(bars, ordered):
        if row["absorbed_by"]:
            axis.text(bar.get_x() + bar.get_width() / 2, row["acceptance_rate_pct"] + 1.2,
                      f"-> {row['absorbed_by']}", ha="center", fontsize=7, rotation=90,
                      color="#444")

    # Knowns are accepted at ~95% by construction; unknowns should be far below.
    axis.axhline(5.0, color="grey", linestyle=":", linewidth=1.0)
    axis.text(len(names) - 0.6, 6.0, "5%", fontsize=7.5, color="grey", ha="right")

    handles = [plt.Rectangle((0, 0), 1, 1, color="#DD8452"),
               plt.Rectangle((0, 0), 1, 1, color="#55A868")]
    axis.legend(handles, ["near unknown", "far unknown"], fontsize=8)

    axis.set_ylabel(f"Accepted as known (%), {score_name.upper()} at the 95th-percentile threshold")
    axis.set_title(f"Which unknown classes slip through: {method}", fontsize=11)
    axis.set_ylim(0, max(values) * 1.35 if values else 1)
    axis.tick_params(axis="x", labelrotation=45, labelsize=8.5)
    for label in axis.get_xticklabels():
        label.set_ha("right")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(alpha=0.25, linewidth=0.6, axis="y")

    fig.tight_layout()
    path = os.path.join(results_dir, "figure_unknown_classes.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def run(method="vanilla", score_name="mls", results_dir=None, cache_dir=None):
    task_config.init_env()
    results_dir = results_dir or task_config.TASK_RESULTS_DIR

    # cache_dir lets the analysis read caches produced elsewhere (a Kaggle or
    # Colab run downloaded into the repository) without copying them.
    path = (os.path.join(cache_dir, f"{method}_outputs.npz") if cache_dir
            else cache_path(method))
    if not os.path.isfile(path):
        raise SystemExit(f"No cached outputs for '{method}' at {path}.")
    cache = np.load(path, allow_pickle=True)

    rows = per_class_rows(cache, method, score_name)

    print(f"\n{'group':<6}{'unknown class':<15}{'accepted':>9}{'rate':>8}   absorbed by")
    for row in sorted(rows, key=lambda r: r["acceptance_rate_pct"], reverse=True):
        print(f"{row['group']:<6}{row['unknown_class']:<15}"
              f"{row['accepted']:>4}/{row['n_images']:<4}{row['acceptance_rate_pct']:>7.1f}%"
              f"   {row['absorbed_by']} ({row['absorbed_count']})")

    for group in ("near", "far"):
        subset = [r for r in rows if r["group"] == group]
        total = sum(r["accepted"] for r in subset)
        count = sum(r["n_images"] for r in subset)
        print(f"  {group}: {total}/{count} accepted = {100.0 * total / count:.1f}%")

    out = os.path.join(results_dir, "unknown_class_analysis.csv")
    with open(out, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {out}")

    figure(rows, results_dir, method, score_name)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="vanilla")
    parser.add_argument("--score", default="mls")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--cache-dir", default=None,
                        help="directory holding <method>_outputs.npz, if not the task cache")
    args = parser.parse_args()
    run(method=args.method, score_name=args.score, results_dir=args.results_dir,
        cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
