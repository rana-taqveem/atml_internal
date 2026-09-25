"""Build Task 2 tables and figures from saved results and histories."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from assignment_01.figure_style import print_ready

TEXT_WIDTH = 5.5
# Printed width (inches) of figures the report shows narrower than the text block.
PRINT_WIDTH = {
    "task2_fig4_per_class": 4.4,
    "task2_fig5_separability": 3.0,
    "task2_fig3_alignment_study": 4.4,
}

# Order used in every table and figure.
METHOD_ORDER = ["erm", "dan", "dann", "cdan"]
STUDY_ORDER = ["dann_alpha0.25", "dann_alpha0.5", "dann"]

LABELS = {
    "erm": "Source-only ERM",
    "dan": "DAN (MMD)",
    "dann": "DANN",
    "cdan": "CDAN",
    "dann_alpha0.25": "DANN a=0.25",
    "dann_alpha0.5": "DANN a=0.5",
}
STYLE = {
    "erm": ("#2a78d6", "o"),
    "dan": ("#eb6834", "s"),
    "dann": ("#1baf7a", "^"),
    "cdan": ("#eda100", "D"),
    "dann_alpha0.25": ("#e87ba4", "v"),
    "dann_alpha0.5": ("#4a3aa7", "P"),
}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_runs(results_dir):
    """Every *_results.json in the folder, keyed by method name."""
    runs = {}
    for path in sorted(Path(results_dir).glob("*_results.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        runs[data.get("method", path.stem.replace("_results", ""))] = data
    return runs


def load_diagnostics(results_dir):
    path = Path(results_dir) / "diagnostics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def tidy(ax):
    ax.grid(True, color="#e8e8e8", linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#999999")
    ax.tick_params(colors="#444444", length=2.5)


def save(fig, out_dir, name):
    print_ready(fig, width_in=PRINT_WIDTH.get(name, TEXT_WIDTH))
    for suffix in ("pdf", "png"):
        fig.savefig(Path(out_dir) / f"{name}.{suffix}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def comparison_rows(runs, diagnostics, order=None, baseline="erm"):
    """One row per method: per-domain source, means, target, change, separability."""
    order = [m for m in (order or METHOD_ORDER) if m in runs]
    baseline_target = (runs[baseline]["results"]["target"]["accuracy_pct"]
                       if baseline in runs else None)
    baseline_f1 = (runs[baseline]["results"]["target"]["macro_f1_pct"]
                   if baseline in runs else None)

    rows = []
    for method in order:
        result = runs[method]["results"]
        source = result["source_validation"]
        accuracies = [d["accuracy_pct"] for d in source.values()]
        row = {
            "method": LABELS.get(method, method),
            "key": method,
        }
        for domain, scores in source.items():
            row[f"{domain}_accuracy_pct"] = scores["accuracy_pct"]
            row[f"{domain}_macro_f1_pct"] = scores["macro_f1_pct"]
        row["mean_source_accuracy_pct"] = float(np.mean(accuracies))
        row["mean_source_macro_f1_pct"] = result["mean_source_validation_macro_f1_pct"]
        row["worst_source_macro_f1_pct"] = min(d["macro_f1_pct"] for d in source.values())
        row["target_accuracy_pct"] = result["target"]["accuracy_pct"]
        row["target_macro_f1_pct"] = result["target"]["macro_f1_pct"]
        row["target_accuracy_change_pp"] = (
            None if baseline_target is None else result["target"]["accuracy_pct"] - baseline_target)
        row["target_macro_f1_change_pp"] = (
            None if baseline_f1 is None else result["target"]["macro_f1_pct"] - baseline_f1)
        row["domain_separability_pct"] = (
            diagnostics.get(method, {}).get("domain_separability_pct"))
        rows.append(row)
    return rows


def training_health_rows(runs):
    """Evidence that each method trained as intended (or did not)."""
    rows = []
    for method, data in runs.items():
        history = data.get("history") or {}
        alignment = history.get("alignment_loss") or []
        domain_accuracy = [v for v in (history.get("domain_accuracy") or []) if v is not None]
        rows.append({
            "method": LABELS.get(method, method),
            "key": method,
            "epochs_run": len(history.get("val_macro_f1", [])),
            "best_epoch": history.get("best_epoch"),
            "final_classification_loss": (history.get("classification_loss") or [None])[-1],
            "first_alignment_loss": alignment[0] if alignment else None,
            "final_alignment_loss": alignment[-1] if alignment else None,
            "max_alignment_loss": max(alignment) if alignment else None,
            "first_discriminator_accuracy_pct": domain_accuracy[0] if domain_accuracy else None,
            "final_discriminator_accuracy_pct": domain_accuracy[-1] if domain_accuracy else None,
            "diverged": bool(alignment and max(alignment) > 10),
            "grad_clip": history.get("grad_clip"),
            "mixed_precision": history.get("mixed_precision"),
            "settings": json.dumps(data.get("settings", {})),
        })
    return rows


def figure_erm_failures(results_dir, out_dir, baseline="erm"):
    """Plot ERM target accuracy and error contribution by class."""
    path = Path(results_dir) / "per_class_target.csv"
    if not path.is_file():
        print("  (no per_class_target.csv: skipping ERM failure figure)")
        return

    with path.open(encoding="utf-8") as file:
        rows = [{(k or "").strip(): (v.strip() if isinstance(v, str) else v)
                 for k, v in record.items()}
                for record in csv.DictReader(file)]
    rows = [r for r in rows if r["method"] == baseline]
    if not rows:
        return

    for row in rows:
        row["support_n"] = int(float(row["support"]))
        row["accuracy"] = float(row["baseline_accuracy_pct"])
        # Errors implied by accuracy and support, so the panel cannot drift
        # away from the accuracy it is drawn beside.
        row["errors"] = int(round(row["support_n"] * (1 - row["accuracy"] / 100)))

    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.3))

    # (a) accuracy, weakest first
    by_accuracy = sorted(rows, key=lambda r: r["accuracy"])
    names = [r["class"] for r in by_accuracy]
    values = [r["accuracy"] for r in by_accuracy]
    bars = axes[0].bar(names, values, color="#4C72B0")
    for bar, row in zip(bars, by_accuracy):
        axes[0].text(bar.get_x() + bar.get_width() / 2, row["accuracy"] + 1.5,
                     f"n={row['support_n']}", ha="center", fontsize=7.5, color="#444")
    axes[0].set_ylim(0, 108)
    axes[0].set_ylabel("Target accuracy (%)")
    axes[0].set_title("(a) Source-only ERM, per class on Sketch")
    axes[0].tick_params(axis="x", labelsize=8.5)

    # (b) error counts with cumulative share
    by_errors = sorted(rows, key=lambda r: r["errors"], reverse=True)
    names = [r["class"] for r in by_errors]
    counts = [r["errors"] for r in by_errors]
    total = sum(counts)
    cumulative = np.cumsum(counts) / total * 100

    axes[1].bar(names, counts, color="#DD8452")
    for index, row in enumerate(by_errors):
        if row.get("confused_with"):
            axes[1].text(index, counts[index] + total * 0.012,
                         f"-> {row['confused_with']}\n({row['confused_count']})",
                         ha="center", fontsize=7, color="#444")

    twin = axes[1].twinx()
    twin.plot(names, cumulative, "o-", color="#55A868", linewidth=1.5, markersize=4)
    twin.set_ylim(0, 105)
    twin.set_ylabel("cumulative share of errors (%)", color="#55A868")
    twin.tick_params(axis="y", labelcolor="#55A868")
    twin.axhline(cumulative[2], color="#55A868", linestyle=":", linewidth=1.0)
    twin.text(len(names) - 1.4, cumulative[2] + 2.5,
              f"top 3 = {cumulative[2]:.1f}%", fontsize=8, color="#55A868", ha="right")

    axes[1].set_ylim(0, max(counts) * 1.28)
    axes[1].set_ylabel("Misclassified target images")
    axes[1].set_title("(b) Error contribution and dominant confusion")
    axes[1].tick_params(axis="x", labelsize=8.5)

    for axis in axes:
        tidy(axis)
    figure.tight_layout()
    save(figure, out_dir, "task2_fig0_erm_failures")


def save_rows(rows, path):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path.name}")


def figure_training_curves(runs, out_dir, order=None):
    """(a) classification loss, (b) alignment loss, (c) source validation macro-F1."""
    order = [m for m in (order or METHOD_ORDER) if m in runs]
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.0))

    panels = [
        ("classification_loss", "Classification loss", "(a) Classification", True),
        ("alignment_loss", "Alignment / domain loss", "(b) Alignment", True),
        ("val_macro_f1", "Source val macro-F1 (%)", "(c) Source validation", False),
    ]
    for ax, (key, ylabel, title, log_scale) in zip(axes, panels):
        plotted = False
        for method in order:
            history = runs[method].get("history") or {}
            values = history.get(key) or []
            if not values or (key == "alignment_loss" and max(values) == 0):
                continue
            colour, marker = STYLE.get(method, ("#555555", "o"))
            ax.plot(range(1, len(values) + 1), values, color=colour, marker=marker,
                    markersize=2.8, linewidth=1.1, label=LABELS.get(method, method))
            plotted = True
        if log_scale and plotted:
            # Log scale: the diverged run reaches 1e6 and would flatten the rest.
            ax.set_yscale("log")
            reference = np.log(7) if key == "classification_loss" else np.log(2)
            note = " ln7 (random)" if key == "classification_loss" else " ln2 (healthy)"
            ax.axhline(reference, color="#999999", linestyle=":", linewidth=0.9)
            ax.text(0.98, reference, note, transform=ax.get_yaxis_transform(),
                    fontsize=6, color="#777777", ha="right", va="bottom")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        tidy(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.10), handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, out_dir, "task2_fig1_training_curves")


def figure_method_comparison(rows, out_dir):
    """(a) source vs target performance, (b) target change against ERM."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.2))

    x = np.arange(len(rows))
    width = 0.38
    left.bar(x - width/2, [r["mean_source_macro_f1_pct"] for r in rows], width,
             color="#2a78d6", label="mean source val")
    left.bar(x + width/2, [r["target_macro_f1_pct"] for r in rows], width,
             color="#eb6834", label="target (Sketch)")
    left.set_xticks(x)
    left.set_xticklabels([r["method"].replace(" ", "\n", 1) for r in rows], fontsize=6.5)
    left.set_ylabel("Macro-F1 (%)")
    left.set_title("(a) Source vs target", loc="left")
    left.legend(frameon=False, fontsize=6.5)
    tidy(left)

    changes = [r["target_macro_f1_change_pp"] or 0.0 for r in rows]
    colours = ["#1baf7a" if c > 0 else "#e34948" for c in changes]
    right.bar(x, changes, 0.6, color=colours)
    right.axhline(0, color="#777777", linewidth=0.8)
    right.set_xticks(x)
    right.set_xticklabels([r["method"].replace(" ", "\n", 1) for r in rows], fontsize=6.5)
    right.set_ylabel("Target macro-F1 change (pp)")
    right.set_title("(b) Change vs ERM", loc="left")
    tidy(right)

    fig.tight_layout()
    save(fig, out_dir, "task2_fig2_method_comparison")


def figure_alignment_study(runs, diagnostics, out_dir):
    """Controlled study: alignment strength vs source, target and stability."""
    points = []
    for method, alpha in [("dann_alpha0.25", 0.25), ("dann_alpha0.5", 0.5), ("dann", 1.0)]:
        if method not in runs:
            continue
        result = runs[method]["results"]
        history = runs[method].get("history") or {}
        alignment = history.get("alignment_loss") or [0]
        points.append({
            "alpha": alpha,
            "source": result["mean_source_validation_macro_f1_pct"],
            "target": result["target"]["macro_f1_pct"],
            "max_alignment": max(alignment),
            "separability": diagnostics.get(method, {}).get("domain_separability_pct"),
        })
    if not points:
        return None
    points.sort(key=lambda p: p["alpha"])

    fig, (left, right) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.1))
    alphas = [p["alpha"] for p in points]

    left.plot(alphas, [p["source"] for p in points], color="#2a78d6", marker="o",
              markersize=4, linewidth=1.3, label="mean source val")
    left.plot(alphas, [p["target"] for p in points], color="#eb6834", marker="s",
              markersize=4, linewidth=1.3, label="target (Sketch)")
    if any(p["separability"] is not None for p in points):
        left.plot(alphas, [p["separability"] for p in points], color="#1baf7a", marker="^",
                  markersize=4, linewidth=1.3, label="domain separability")
    left.set_xlabel("Max gradient-reversal strength $\\alpha$")
    left.set_ylabel("Percent")
    left.set_title("(a) Alignment strength", loc="left")
    left.set_xticks(alphas)
    left.legend(frameon=False, fontsize=6.5)
    tidy(left)

    right.plot(alphas, [p["max_alignment"] for p in points], color="#e34948", marker="D",
               markersize=4, linewidth=1.3)
    right.set_yscale("log")
    right.axhline(np.log(2), color="#999999", linestyle=":", linewidth=0.9)
    right.text(0.98, np.log(2), " ln2 (healthy)", transform=right.get_yaxis_transform(),
               fontsize=6, color="#777777", ha="right", va="bottom")
    right.set_xlabel("Max gradient-reversal strength $\\alpha$")
    right.set_ylabel("Peak domain loss (log scale)")
    right.set_title("(b) Training stability", loc="left")
    right.set_xticks(alphas)
    tidy(right)

    fig.tight_layout()
    save(fig, out_dir, "task2_fig3_alignment_study")
    return points


def figure_per_class(results_dir, out_dir, baseline="erm", order=None):
    """Plot per-class target accuracy changes against ERM."""
    path = Path(results_dir) / "per_class_target.csv"
    if not path.is_file():
        print("  (no per_class_target.csv yet: skipping per-class figure)")
        return None

    with path.open(encoding="utf-8") as file:
        # Accept column-aligned CSVs by stripping surrounding whitespace.
        rows = [{(k or "").strip(): (v.strip() if isinstance(v, str) else v)
                 for k, v in record.items()}
                for record in csv.DictReader(file)]
    if not rows:
        return None

    methods = [m for m in (order or METHOD_ORDER)
               if m != baseline and any(r["method"] == m for r in rows)]
    classes = []
    for row in rows:
        if row["class"] not in classes:
            classes.append(row["class"])

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.3))
    width = 0.8 / max(len(methods), 1)
    x = np.arange(len(classes))

    for index, method in enumerate(methods):
        by_class = {r["class"]: r for r in rows if r["method"] == method}
        changes = [float(by_class[c]["change_pp"]) if by_class.get(c, {}).get("change_pp") else 0.0
                   for c in classes]
        colour, _ = STYLE.get(method, ("#555555", "o"))
        ax.bar(x + (index - (len(methods) - 1) / 2) * width, changes, width,
               color=colour, label=LABELS.get(method, method))

    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=7)
    ax.set_ylabel("Target accuracy change vs ERM (pp)")
    ax.set_title("Per-class change on Sketch", loc="left")
    ax.legend(frameon=False, ncol=3, fontsize=6.5)
    tidy(ax)
    fig.tight_layout()
    save(fig, out_dir, "task2_fig4_per_class")
    return rows


def figure_separability(rows, out_dir):
    """Domain separability against target performance, the RQ2 view."""
    points = [r for r in rows if r.get("domain_separability_pct") is not None]
    if not points:
        print("  (no separability values yet: skipping separability figure)")
        return None

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.62, 2.3))
    for row in points:
        colour, marker = STYLE.get(row["key"], ("#555555", "o"))
        ax.scatter(row["domain_separability_pct"], row["target_macro_f1_pct"],
                   color=colour, marker=marker, s=45, edgecolors="white", linewidths=0.6,
                   label=row["method"], zorder=3)
    ax.axvline(50, color="#999999", linestyle=":", linewidth=0.9)
    ax.text(50, ax.get_ylim()[0], " chance (50%)", fontsize=6, color="#777777", va="bottom")
    ax.set_xlabel("Domain separability (%)")
    ax.set_ylabel("Target macro-F1 (%)")
    ax.set_title("Separability vs target recognition", loc="left")
    ax.legend(frameon=False, fontsize=6.5, loc="best")
    tidy(ax)
    fig.tight_layout()
    save(fig, out_dir, "task2_fig5_separability")
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir")
    parser.add_argument("--out", default=None, help="default: <results_dir>/analysis")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else Path(args.results_dir) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.results_dir)
    diagnostics = load_diagnostics(args.results_dir)
    if not runs:
        raise SystemExit(f"No *_results.json found in {args.results_dir}")
    print(f"Loaded {len(runs)} runs: {', '.join(runs)}")
    if not diagnostics:
        print("  (no diagnostics.json yet: separability columns will be blank)")

    rows = comparison_rows(runs, diagnostics)
    save_rows(rows, out_dir / "task2_method_comparison.csv")
    save_rows(training_health_rows(runs), out_dir / "task2_training_health.csv")

    figure_erm_failures(args.results_dir, out_dir)
    figure_training_curves(runs, out_dir)
    figure_method_comparison(rows, out_dir)
    figure_per_class(args.results_dir, out_dir)
    figure_separability(rows, out_dir)
    study = figure_alignment_study(runs, diagnostics, out_dir)
    if study:
        save_rows(study, out_dir / "task2_alignment_study.csv")

    print(f"\nArtifacts in {out_dir}")


if __name__ == "__main__":
    main()
