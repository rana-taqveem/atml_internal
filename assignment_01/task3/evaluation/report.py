"""Generate Task 3 report tables and figures from saved results."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from assignment_01.figure_style import TEXT_WIDTH_IN, print_ready

# Printed width (inches) of figures the report shows narrower than the text block.
PRINT_WIDTH = {"task3_fig3_alignment_study": 3.6}

# Main comparison first, then the strength study.
METHOD_ORDER = ["erm", "dan_dg", "sam", "dan_dg_lambda0.1", "dan_dg_lambda10"]
LABELS = {
    "erm": "ERM",
    "dan_dg": r"DAN-DG ($\lambda$=1)",
    "sam": r"SAM ($\rho$=0.05)",
    "dan_dg_lambda0.1": r"DAN-DG ($\lambda$=0.1)",
    "dan_dg_lambda10": r"DAN-DG ($\lambda$=10)",
}
COLORS = {
    "erm": "#4C72B0",
    "dan_dg": "#DD8452",
    "sam": "#55A868",
    "dan_dg_lambda0.1": "#C44E52",
    "dan_dg_lambda10": "#8172B3",
}


def read_csv_stripped(path):
    """Read CSV rows with whitespace trimmed from headers and values."""
    with Path(path).open(encoding="utf-8") as file:
        return [{(k or "").strip(): (v.strip() if isinstance(v, str) else v)
                 for k, v in record.items()}
                for record in csv.DictReader(file)]


def load_histories(results_dir):
    histories = {}
    for path in sorted(Path(results_dir).glob("*_history.json")):
        name = path.name[: -len("_history.json")]
        with path.open(encoding="utf-8") as file:
            histories[name] = json.load(file)
    return histories


def load_final(results_dir):
    for name in ("final_results.json", "task3_final_results.json"):
        path = Path(results_dir) / name
        if path.is_file():
            with path.open(encoding="utf-8") as file:
                return json.load(file)
    return {}


def ordered(names):
    known = [m for m in METHOD_ORDER if m in names]
    return known + [m for m in names if m not in METHOD_ORDER]


def tidy(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(alpha=0.25, linewidth=0.6)


def save(figure, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print_ready(figure, width_in=PRINT_WIDTH.get(name, TEXT_WIDTH_IN))
    for suffix in ("png", "pdf"):
        figure.savefig(out_dir / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"  wrote {name}.pdf / .png")


def figure_training_curves(histories, out_dir):
    """Classification loss, MMD penalty, and source validation macro-F1."""
    names = ordered(histories)
    if not names:
        return

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

    for name in names:
        history = histories[name]
        epochs = np.arange(1, len(history["classification_loss"]) + 1)
        style = dict(color=COLORS.get(name, "grey"), linewidth=1.6,
                     label=LABELS.get(name, name))
        axes[0].plot(epochs, history["classification_loss"], **style)
        alignment = history.get("alignment_loss") or []
        # SAM logs zeros: it has no discrepancy penalty, so it is left out.
        if any(value for value in alignment):
            axes[1].plot(epochs, alignment, **style)
        axes[2].plot(epochs, history["val_macro_f1"], **style)

    axes[0].set_title("Classification loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy")
    axes[0].set_yscale("log")

    axes[1].set_title("Pairwise source MMD penalty")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel(r"$\lambda_{DG}\cdot$MMD$^2$")
    axes[1].set_yscale("log")

    axes[2].set_title("Source validation macro-F1")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("%")

    for axis in axes:
        tidy(axis)
    axes[0].legend(fontsize=7.5)
    figure.tight_layout()
    save(figure, out_dir, "task3_fig1_training_curves")


def figure_method_comparison(final, out_dir):
    """Mean-source, worst-source and Sketch macro-F1 as grouped bars."""
    names = ordered(final)
    if not names:
        return

    mean_source = [final[m]["mean_source_macro_f1_pct"] for m in names]
    worst_source = [final[m]["worst_source_macro_f1_pct"] for m in names]
    target = [final[m]["target_macro_f1_pct"] for m in names]

    positions = np.arange(len(names))
    width = 0.27

    figure, axis = plt.subplots(figsize=(1.9 * len(names) + 2.2, 4.2))
    axis.bar(positions - width, mean_source, width, label="mean source", color="#4C72B0")
    axis.bar(positions, worst_source, width, label="worst source", color="#8FA8CF")
    axis.bar(positions + width, target, width, label="Sketch (target)", color="#DD8452")

    for position, value in zip(positions + width, target):
        axis.text(position, value + 1.2, f"{value:.1f}", ha="center", fontsize=7.5)

    axis.set_xticks(positions)
    axis.set_xticklabels([LABELS.get(m, m) for m in names], fontsize=8)
    axis.set_ylabel("macro-F1 (%)")
    axis.set_ylim(0, 105)
    axis.set_title("Source validation performance does not predict Sketch")
    axis.legend(fontsize=8)
    tidy(axis)
    figure.tight_layout()
    save(figure, out_dir, "task3_fig2_method_comparison")


def figure_alignment_study(final, out_dir):
    """lambda_DG against source, separability and target on one axis."""
    points = []
    for name, entry in final.items():
        settings = entry.get("settings") or {}
        if "lambda_dg" in settings:
            points.append((float(settings["lambda_dg"]), entry))
    if len(points) < 2:
        print("  (fewer than two lambda settings: skipping study figure)")
        return None

    points.sort(key=lambda pair: pair[0])
    lambdas = [p[0] for p in points]
    source = [p[1]["mean_source_macro_f1_pct"] for p in points]
    target = [p[1]["target_macro_f1_pct"] for p in points]
    separability = [p[1]["source_domain_separability_pct"] for p in points]
    chance = points[0][1].get("separability_chance_pct", 33.3)

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot(lambdas, source, "o-", color="#4C72B0", label="mean source macro-F1")
    axis.plot(lambdas, target, "s-", color="#DD8452", label="Sketch macro-F1")
    axis.plot(lambdas, separability, "^--", color="#55A868", label="source separability")
    axis.axhline(chance, color="grey", linestyle=":", linewidth=1.0,
                 label=f"separability chance ({chance:.1f}%)")

    axis.set_xscale("log")
    axis.set_xticks(lambdas)
    axis.set_xticklabels([f"{value:g}" for value in lambdas])
    axis.set_xlabel(r"$\lambda_{DG}$  (alignment strength)")
    axis.set_ylabel("%")
    axis.set_ylim(0, 105)
    axis.set_title("Increasing source alignment: too much removes class structure")
    axis.legend(fontsize=8)
    tidy(axis)
    figure.tight_layout()
    save(figure, out_dir, "task3_fig3_alignment_study")

    return [{"lambda_dg": lam, "mean_source_macro_f1_pct": s,
             "source_domain_separability_pct": sep, "target_macro_f1_pct": t}
            for lam, s, sep, t in zip(lambdas, source, separability, target)]


def figure_diagnostics(final, out_dir):
    """Plot source separability and sharpness against Sketch performance."""
    names = ordered(final)
    if not names:
        return

    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))

    separability = [final[m]["source_domain_separability_pct"] for m in names]
    sharpness = [final[m]["sharpness"]["delta_sharp"] for m in names]
    target = [final[m]["target_macro_f1_pct"] for m in names]
    chance = final[names[0]].get("separability_chance_pct", 33.3)

    for axis, values, label, title in (
        (axes[0], separability, "Source-domain separability (%)",
         "(a) Source invariance vs transfer"),
        (axes[1], sharpness, r"$\Delta_{\mathrm{sharp}}$ (loss increase)",
         "(b) Local stability vs transfer"),
    ):
        # A legend rather than per-point labels: the two DAN-DG settings sit
        # almost on top of each other, so annotations collide.
        for name, x, y in zip(names, values, target):
            axis.scatter(x, y, s=95, color=COLORS.get(name, "grey"), zorder=3,
                         label=LABELS.get(name, name))
        axis.set_xlabel(label)
        axis.set_ylabel("Sketch macro-F1 (%)")
        axis.set_title(title, fontsize=10.5)
        tidy(axis)

    axes[0].axvline(chance, color="grey", linestyle=":", linewidth=1.0)
    axes[0].text(chance + 1.0, max(target) - 2, f"chance ({chance:.1f}%)",
                 fontsize=7.5, color="grey")
    axes[0].legend(fontsize=8, loc="center left", bbox_to_anchor=(0.08, 0.45),
                   framealpha=0.9)

    figure.tight_layout()
    save(figure, out_dir, "task3_fig5_diagnostics")


def figure_target_access(final, task2_results_dir, out_dir):
    """Compare target-aware DAN with target-free DAN-DG."""
    diagnostics_path = Path(task2_results_dir) / "diagnostics.json"
    if not diagnostics_path.is_file():
        print("  (no Task 2 diagnostics.json: skipping target-access figure)")
        return
    with diagnostics_path.open(encoding="utf-8") as file:
        task2 = json.load(file)
    if "dan" not in task2 or "erm" not in task2 or "dan_dg" not in final:
        print("  (missing DAN or DAN-DG: skipping target-access figure)")
        return

    baseline = task2["erm"]["target_macro_f1_pct"]
    entries = [
        ("Source-only ERM\n(shared baseline)", baseline, "#4C72B0"),
        ("DAN, Task 2\n(sees unlabelled Sketch)", task2["dan"]["target_macro_f1_pct"], "#DD8452"),
        ("DAN-DG, Task 3\n(never sees Sketch)", final["dan_dg"]["target_macro_f1_pct"], "#55A868"),
    ]

    figure, axis = plt.subplots(figsize=(6.6, 4.3))
    positions = np.arange(len(entries))
    values = [value for _, value, _ in entries]
    axis.bar(positions, values, 0.55, color=[c for _, _, c in entries])

    for position, (_, value, _) in zip(positions, entries):
        change = value - baseline
        text = f"{value:.2f}" if change == 0 else f"{value:.2f}\n({change:+.2f} pp)"
        axis.text(position, value + 1.2, text, ha="center", fontsize=8.5)

    axis.axhline(baseline, color="grey", linestyle=":", linewidth=1.0)
    axis.set_xticks(positions)
    axis.set_xticklabels([name for name, _, _ in entries], fontsize=8)
    axis.set_ylabel("Sketch macro-F1 (%)")
    axis.set_ylim(0, max(values) * 1.22)
    axis.set_title("What was unlabelled target data worth?", fontsize=10.5)
    tidy(axis)
    figure.tight_layout()
    save(figure, out_dir, "task3_fig6_target_access")


def figure_per_class(results_dir, out_dir, baseline="erm"):
    """Per-class Sketch accuracy change against ERM."""
    path = None
    for name in ("per_class_sketch.csv", "task3_per_class_sketch.csv"):
        candidate = Path(results_dir) / name
        if candidate.is_file():
            path = candidate
            break
    if path is None:
        print("  (no per-class CSV: skipping per-class figure)")
        return

    rows = read_csv_stripped(path)
    if not rows:
        return

    methods = [m for m in ordered({r["method"] for r in rows}) if m != baseline]
    classes = []
    for row in rows:
        if row["class"] not in classes:
            classes.append(row["class"])

    positions = np.arange(len(classes))
    width = 0.8 / max(len(methods), 1)

    figure, axis = plt.subplots(figsize=(1.35 * len(classes) + 3.0, 4.4))
    for index, method in enumerate(methods):
        changes = []
        for class_name in classes:
            match = [r for r in rows if r["method"] == method and r["class"] == class_name]
            changes.append(float(match[0]["change_pp"]) if match else 0.0)
        offset = (index - (len(methods) - 1) / 2) * width
        axis.bar(positions + offset, changes, width,
                 label=LABELS.get(method, method), color=COLORS.get(method, None))

    axis.axhline(0, color="black", linewidth=0.9)
    axis.set_xticks(positions)
    axis.set_xticklabels(classes, fontsize=8.5)
    axis.set_ylabel("Sketch accuracy change vs ERM (pp)")
    axis.set_title("An aggregate gain can hide class-level negative transfer")
    axis.legend(fontsize=8)
    tidy(axis)
    figure.tight_layout()
    save(figure, out_dir, "task3_fig4_per_class")


def figure_per_class_vs_task2(results_dir, task2_results_dir, out_dir):
    """Compare per-class Sketch changes for DAN and DAN-DG."""
    task3_path = None
    for name in ("per_class_sketch.csv", "task3_per_class_sketch.csv"):
        candidate = Path(results_dir) / name
        if candidate.is_file():
            task3_path = candidate
            break
    task2_path = Path(task2_results_dir) / "per_class_target.csv"
    if task3_path is None or not task2_path.is_file():
        print("  (missing a per-class CSV: skipping Task 2 comparison figure)")
        return

    task3_rows = [r for r in read_csv_stripped(task3_path) if r["method"] == "dan_dg"]
    task2_rows = [r for r in read_csv_stripped(task2_path) if r["method"] == "dan"]
    if not task3_rows or not task2_rows:
        print("  (no DAN or DAN-DG rows: skipping Task 2 comparison figure)")
        return

    classes = [r["class"] for r in task3_rows]
    task2_changes, task3_changes = [], []
    for class_name in classes:
        left = [r for r in task2_rows if r["class"] == class_name]
        right = [r for r in task3_rows if r["class"] == class_name]
        task2_changes.append(float(left[0]["change_pp"]) if left else 0.0)
        task3_changes.append(float(right[0]["change_pp"]) if right else 0.0)

    positions = np.arange(len(classes))
    width = 0.38

    figure, axis = plt.subplots(figsize=(1.45 * len(classes) + 3.0, 4.4))
    axis.bar(positions - width / 2, task2_changes, width, color="#DD8452",
             label="DAN, Task 2 (sees unlabelled Sketch)")
    axis.bar(positions + width / 2, task3_changes, width, color="#55A868",
             label="DAN-DG, Task 3 (never sees Sketch)")

    axis.axhline(0, color="black", linewidth=0.9)
    axis.set_xticks(positions)
    axis.set_xticklabels(classes, fontsize=8.5)
    axis.set_ylabel("Sketch accuracy change vs shared ERM (pp)")
    axis.set_title("Same discrepancy measure, different information", fontsize=10.5)
    axis.legend(fontsize=8)
    tidy(axis)
    figure.tight_layout()
    save(figure, out_dir, "task3_fig7_per_class_vs_task2")


def save_rows(rows, path):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {Path(path).name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir")
    parser.add_argument("--out", default=None, help="default: <results_dir>/analysis")
    parser.add_argument("--task2-results", default="assignment_01/task2/results",
                        help="for the target-aware vs target-free comparison")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else Path(args.results_dir) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    histories = load_histories(args.results_dir)
    # ERM is the reused Task 2 checkpoint, so its curve lives with Task 2.
    erm_history = Path(args.task2_results) / "erm_history.json"
    if "erm" not in histories and erm_history.is_file():
        with erm_history.open(encoding="utf-8") as file:
            histories["erm"] = json.load(file)
    final = load_final(args.results_dir)
    print(f"Loaded {len(histories)} histories, {len(final)} evaluated methods")
    if not final:
        print("  (no final_results.json: run evaluate_sketch.py first)")

    figure_training_curves(histories, out_dir)
    if final:
        figure_method_comparison(final, out_dir)
        figure_diagnostics(final, out_dir)
        figure_target_access(final, args.task2_results, out_dir)
        study = figure_alignment_study(final, out_dir)
        if study:
            save_rows(study, out_dir / "task3_alignment_study.csv")
    figure_per_class(args.results_dir, out_dir)
    figure_per_class_vs_task2(args.results_dir, args.task2_results, out_dir)

    print(f"\nArtifacts in {out_dir}")


if __name__ == "__main__":
    main()
