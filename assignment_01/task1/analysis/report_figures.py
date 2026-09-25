"""Print-ready Task 1 figures, built from the exported analysis CSVs.

Sized for a NeurIPS page: 5.5 in text width, 8 pt labels. Saves PDF (vector,
for LaTeX) and PNG (for checking). Reads only the CSV files written by
report.py, so it needs no model, no GPU and no .pt files.

    python -m assignment_01.task1.analysis.report_figures <analysis_dir> [--out DIR]
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from assignment_01.figure_style import print_ready

TEXT_WIDTH = 5.5

METHODS = ["resnet50", "vit_b_16", "clip_vit_b_32", "clip_vit_b_32_zero_shot"]
LABELS = {
    "resnet50": "ResNet-50 head",
    "vit_b_16": "ViT-B/16 head",
    "clip_vit_b_32": "CLIP head",
    "clip_vit_b_32_zero_shot": "CLIP zero-shot",
}
BACKBONES = ["resnet50", "vit_b_16", "clip_vit_b_32"]
BACKBONE_LABELS = {"resnet50": "ResNet-50", "vit_b_16": "ViT-B/16", "clip_vit_b_32": "CLIP ViT-B/32"}
# Validated categorical slots, one marker each so colour is never the only cue.
STYLE = {
    "resnet50": ("#2a78d6", "o"),
    "vit_b_16": ("#eb6834", "s"),
    "clip_vit_b_32": ("#1baf7a", "^"),
    "clip_vit_b_32_zero_shot": ("#eda100", "D"),
}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.spines.top": False, "axes.spines.right": False,
})


def read(path):
    with open(path, newline="", encoding="utf-8") as file:
        return [{k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
                for row in csv.DictReader(file)]


def tidy(ax):
    ax.grid(True, color="#e8e8e8", linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#999999")
    ax.tick_params(colors="#444444", length=2.5)


def save(fig, out_dir, name):
    print_ready(fig, width_in=TEXT_WIDTH)
    for suffix in ("pdf", "png"):
        fig.savefig(Path(out_dir) / f"{name}.{suffix}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def figure_interventions(analysis_dir, out_dir):
    """(a) accuracy change per intervention with 95% ranges, (b) translation curves."""
    tests = [r for r in read(Path(analysis_dir) / "significance_tests.csv")
             if r["family"] == "color_and_patch"]
    summary = read(Path(analysis_dir) / "translation_summary.csv")

    fig, (left, middle, right) = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.1),
                                              gridspec_kw={"width_ratios": [1.15, 1, 1]})

    conditions = ["grey_scale", "hue", "patch_shuffle"]
    names = {"grey_scale": "Grayscale", "hue": "Hue 30°", "patch_shuffle": "Patch shuffle"}
    width = 0.2
    for i, method in enumerate(METHODS):
        colour, marker = STYLE[method]
        x, y, lo, hi = [], [], [], []
        for j, condition in enumerate(conditions):
            row = next(r for r in tests if r["method"] == method and r["condition"] == condition)
            centre = j + (i - 1.5) * width
            x.append(centre)
            y.append(float(row["accuracy_change_pp"]))
            lo.append(float(row["accuracy_change_pp"]) - float(row["change_ci_low_pp"]))
            hi.append(float(row["change_ci_high_pp"]) - float(row["accuracy_change_pp"]))
        left.errorbar(x, y, yerr=[lo, hi], fmt=marker, color=colour, markersize=3.6,
                      elinewidth=0.9, capsize=1.8, linestyle="none",
                      markeredgecolor="white", markeredgewidth=0.5, label=LABELS[method])
    left.axhline(0, color="#777777", linewidth=0.8)
    left.set_xticks(range(len(conditions)))
    left.set_xticklabels(["Gray-\nscale", "Hue\n30°", "Patch\nshuffle"])
    left.set_xlim(-0.55, len(conditions) - 0.45)
    left.set_ylabel("Accuracy change vs clean (pp)")
    left.set_title("(a) Colour and patch structure", loc="left")
    tidy(left)

    for axis, key, title, ylabel in (
        (middle, "mean_accuracy_pct", "(b) Translation: accuracy", "Top-1 accuracy (%)"),
        (right, "mean_consistency_pct", "(c) Translation: consistency", "Consistency (%)"),
    ):
        for method in METHODS:
            colour, marker = STYLE[method]
            rows = sorted((r for r in summary if r["model_name"] == method),
                          key=lambda r: int(r["displacement"]))
            axis.plot([int(r["displacement"]) for r in rows], [float(r[key]) for r in rows],
                      color=colour, marker=marker, markersize=3.2, linewidth=1.1,
                      markeredgecolor="white", markeredgewidth=0.5)
        axis.set_xticks([0, 8, 16, 32])
        axis.set_xlabel("Displacement (px)")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left")
        tidy(axis)

    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.09), handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, out_dir, "fig1_interventions")


def figure_cue_and_representation(analysis_dir, out_dir):
    """(a) shape bias and coverage with ranges, (b) cosine stability per backbone."""
    estimates = read(Path(analysis_dir) / "estimates_with_ci.csv")
    stability = read(Path(analysis_dir) / "representation_stability.csv")

    fig, (left, right) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.4),
                                      gridspec_kw={"width_ratios": [1, 1.05]})

    y = np.arange(len(METHODS))
    for i, method in enumerate(METHODS):
        colour, marker = STYLE[method]
        bias = next(r for r in estimates if r["method"] == method and r["quantity"].startswith("shape bias"))
        cover = next(r for r in estimates if r["method"] == method and r["quantity"].startswith("coverage"))
        value, low, high = float(bias["estimate"]), float(bias["ci_low"]), float(bias["ci_high"])
        left.plot([low, high], [i, i], color=colour, linewidth=1.4, solid_capstyle="round")
        left.plot(value, i, marker=marker, color=colour, markersize=4.5,
                  markeredgecolor="white", markeredgewidth=0.6)
        left.plot(float(cover["estimate"]), i, marker="|", color="#444444", markersize=7,
                  markeredgewidth=1.2)
    left.set_yticks(y)
    left.set_yticklabels([LABELS[m] for m in METHODS])
    left.invert_yaxis()
    left.set_xlim(70, 101)
    left.set_xlabel("Percent")
    left.set_title("(a) Shape bias, 95% range, coverage $|$", loc="left", fontsize=8)
    tidy(left)

    conditions = ["hue", "grey_scale", "translation_32px_mean", "patch_shuffle", "cue_conflict"]
    names = {"hue": "Hue 30°", "grey_scale": "Grayscale", "translation_32px_mean": "Translation 32px",
             "patch_shuffle": "Patch shuffle", "cue_conflict": "Cue conflict"}
    offsets = dict(zip(BACKBONES, (-0.2, 0.0, 0.2)))
    for backbone in BACKBONES:
        colour, marker = STYLE[backbone]
        xs, ys = [], []
        for j, condition in enumerate(conditions):
            row = next((r for r in stability
                        if r["backbone"] == backbone and r["condition"] == condition), None)
            if row:
                xs.append(float(row["cosine_stability_mean"]))
                ys.append(j + offsets[backbone])
        right.scatter(xs, ys, color=colour, marker=marker, s=22, edgecolors="white",
                      linewidths=0.5, label=BACKBONE_LABELS[backbone], zorder=3)
    right.set_yticks(range(len(conditions)))
    right.set_yticklabels([names[c] for c in conditions])
    right.invert_yaxis()
    right.set_xlim(0.35, 1.02)
    right.set_xlabel("Cosine stability $I_T$")
    right.set_title("(b) Representation stability", loc="left", fontsize=8)
    right.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3,
                 handletextpad=0.2, columnspacing=1.0)
    tidy(right)

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out_dir, "fig2_cue_and_representation")


CUE_CASES = [
    ("All four follow shape", lambda d: set(d) == {"shape"}),
    ("Methods split: shape vs texture", lambda d: {"shape", "texture"} <= set(d)),
    ("All four follow texture", lambda d: set(d) == {"texture"}),
    ("A method answers a third class", lambda d: "other" in d),
]


def load_conflict_records(conflict_dir):
    import json

    conflict_dir = Path(conflict_dir)
    records = {}
    for entry in json.loads((conflict_dir / "manifest.json").read_text(encoding="utf-8")):
        record = json.loads((conflict_dir / entry["metadata_path"]).read_text(encoding="utf-8"))
        records[record["conflict_id"]] = record
    return records


def case_pools(decisions):
    """Candidate conflicts for each decision pattern, sorted by id."""
    pools = {}
    for title, predicate in CUE_CASES:
        pools[title] = sorted((r for r in decisions
                               if predicate([r[f"{m}_decision"] for m in METHODS])),
                              key=lambda r: r["conflict_id"])
    return pools


def figure_cue_candidates(analysis_dir, conflict_dir, out_dir, per_case=10):
    """Contact sheet: conflict images per decision pattern, labelled with their id."""
    from PIL import Image

    conflict_dir = Path(conflict_dir)
    records = load_conflict_records(conflict_dir)
    pools = case_pools(read(Path(analysis_dir) / "cue_conflict_decisions.csv"))

    for title, pool in pools.items():
        if not pool:
            continue
        stride = max(1, len(pool) // per_case)
        shown = pool[::stride][:per_case]
        cols = 5
        rows = int(np.ceil(len(shown) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(TEXT_WIDTH * 1.6, 2.1 * rows), squeeze=False)
        for axis in axes.flat:
            axis.axis("off")
        for axis, row in zip(axes.flat, shown):
            with Image.open(conflict_dir / records[row["conflict_id"]]["image_path"]) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_title(f"{row['conflict_id']}\n{row['content_class']} / {row['style_class']}",
                           fontsize=6)
        fig.suptitle(f"{title}  ({len(pool)} available, showing {len(shown)})", fontsize=9)
        fig.tight_layout()
        name = "candidates_" + title.lower().replace(" ", "_").replace(":", "")
        save(fig, out_dir, name)
        print(f"    ids: {', '.join(r['conflict_id'] for r in shown)}")


def figure_cue_examples(analysis_dir, conflict_dir, out_dir, per_case=1, seed=6304,
                        conflict_ids=None):
    """One compact row per decision pattern: content, style, conflict, predictions.

    conflict_ids picks specific conflicts instead of sampling; their decision
    pattern is looked up so the row keeps the right caption.
    """
    from PIL import Image

    conflict_dir = Path(conflict_dir)
    decisions = read(Path(analysis_dir) / "cue_conflict_decisions.csv")
    records = load_conflict_records(conflict_dir)

    chosen = []
    if conflict_ids:
        by_id = {r["conflict_id"]: r for r in decisions}
        for conflict_id in conflict_ids:
            if conflict_id not in by_id:
                raise SystemExit(f"Unknown conflict id: {conflict_id}")
            row = by_id[conflict_id]
            pattern = [row[f"{m}_decision"] for m in METHODS]
            title = next((t for t, predicate in CUE_CASES if predicate(pattern)), "Cue conflict")
            chosen.append((title, row))
    else:
        rng = np.random.default_rng(seed)
        for title, pool in case_pools(decisions).items():
            if not pool:
                continue
            for index in rng.choice(len(pool), size=min(per_case, len(pool)), replace=False):
                chosen.append((title, pool[int(index)]))

    rows = len(chosen)
    fig, axes = plt.subplots(rows, 4, figsize=(TEXT_WIDTH, 1.28 * rows),
                             gridspec_kw={"width_ratios": [1, 1, 1, 2.35]}, squeeze=False)

    for (title, row), axis_row in zip(chosen, axes):
        record = records[row["conflict_id"]]
        panels = [(record["content_image_path"], f"content: {row['content_class']}"),
                  (record["style_image_path"], f"style: {row['style_class']}"),
                  (record["image_path"], "cue conflict")]
        for axis, (relative, caption) in zip(axis_row[:3], panels):
            with Image.open(conflict_dir / relative) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_title(caption, fontsize=6.5, pad=2)
            axis.axis("off")

        lines = [title] + [f"{LABELS[m]}: {row[f'{m}_prediction']} ({row[f'{m}_decision']})"
                           for m in METHODS]
        axis_row[3].text(0.0, 0.5, "\n".join(lines), fontsize=6.3, va="center", ha="left",
                         linespacing=1.45)
        axis_row[3].axis("off")

    fig.tight_layout(h_pad=0.4)
    save(fig, out_dir, "fig3_cue_examples")


CLASS_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                 "#008300", "#4a3aa7", "#e34948", "#7a7a7a", "#8c5a2b"]
STL10_CLASSES = ["airplane", "bird", "car", "cat", "deer",
                 "dog", "horse", "monkey", "ship", "truck"]


def figure_tsne(analysis_dir, out_dir, conditions=("cue_conflict", "patch_shuffle")):
    """One joint projection per backbone; rows are interventions, columns backbones."""
    points = read(Path(analysis_dir) / "tsne_points.csv")
    titles = {"cue_conflict": "Cue conflict", "patch_shuffle": "Patch shuffle"}

    fig, axes = plt.subplots(len(conditions), len(BACKBONES),
                             figsize=(TEXT_WIDTH, 1.9 * len(conditions)), squeeze=False)

    for row, condition in enumerate(conditions):
        for col, backbone in enumerate(BACKBONES):
            axis = axes[row][col]
            clean = [p for p in points if p["backbone"] == backbone and p["condition"] == "baseline"]
            moved = [p for p in points if p["backbone"] == backbone and p["condition"] == condition]

            for group, size, alpha, marker, width in ((clean, 2.2, 0.28, "o", 0), (moved, 4.5, 0.9, "x", 0.5)):
                xs = np.array([float(p["x"]) for p in group])
                ys = np.array([float(p["y"]) for p in group])
                labels = np.array([int(p["label"]) for p in group])
                for class_idx in range(10):
                    mask = labels == class_idx
                    if mask.any():
                        axis.scatter(xs[mask], ys[mask], s=size, alpha=alpha, marker=marker,
                                     c=CLASS_COLOURS[class_idx], linewidths=width)
            if row == 0:
                axis.set_title(BACKBONE_LABELS[backbone], fontsize=8)
            if col == 0:
                axis.set_ylabel(titles.get(condition, condition), fontsize=7.5)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#cccccc")
                spine.set_visible(True)

    handles = [plt.Line2D([], [], marker="o", linestyle="", color=CLASS_COLOURS[i], markersize=3,
                          label=name) for i, name in enumerate(STL10_CLASSES)]
    handles += [plt.Line2D([], [], marker="x", linestyle="", color="#555555", markersize=3.5,
                           label="transformed"),
                plt.Line2D([], [], marker="o", linestyle="", color="#555555", alpha=0.35,
                           markersize=3, label="clean")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.10), handletextpad=0.2, columnspacing=0.9)
    fig.tight_layout(h_pad=0.5, w_pad=0.5)
    save(fig, out_dir, "fig4_tsne")


def figure_cue_examples_compact(analysis_dir, conflict_dir, out_dir, per_case=1, seed=6304):
    """Space-saving fig3: one column per decision pattern, conflict image only.

    Same seeded selection as figure_cue_examples, so the examples are identical;
    the content and style classes move into the panel title.
    """
    from PIL import Image

    conflict_dir = Path(conflict_dir)
    decisions = read(Path(analysis_dir) / "cue_conflict_decisions.csv")
    records = load_conflict_records(conflict_dir)

    chosen = []
    rng = np.random.default_rng(seed)
    for title, pool in case_pools(decisions).items():
        if not pool:
            continue
        for index in rng.choice(len(pool), size=min(per_case, len(pool)), replace=False):
            chosen.append((title, pool[int(index)]))

    short = {"resnet50": "R50", "vit_b_16": "ViT", "clip_vit_b_32": "CLIP",
             "clip_vit_b_32_zero_shot": "CLIP-ZS"}
    fig, axes = plt.subplots(1, len(chosen), figsize=(TEXT_WIDTH, 2.25), squeeze=False)
    for axis, (title, row) in zip(axes[0], chosen):
        record = records[row["conflict_id"]]
        with Image.open(conflict_dir / record["image_path"]) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(f"{title}\nshape: {row['content_class']}, texture: {row['style_class']}",
                       fontsize=6.2, pad=3)
        lines = [f"{short[m]}: {row[f'{m}_prediction']} ({row[f'{m}_decision']})" for m in METHODS]
        axis.text(0.5, -0.05, "\n".join(lines), transform=axis.transAxes, fontsize=6.0,
                  va="top", ha="center", linespacing=1.35)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    fig.tight_layout(w_pad=0.6)
    save(fig, out_dir, "fig3_cue_examples_compact")


def figure_tsne_strip(analysis_dir, out_dir, conditions=("cue_conflict", "patch_shuffle")):
    """Space-saving fig4: the same projections as figure_tsne in a single row."""
    points = read(Path(analysis_dir) / "tsne_points.csv")
    titles = {"cue_conflict": "cue conflict", "patch_shuffle": "patch shuffle"}
    panels = [(c, b) for c in conditions for b in BACKBONES]

    fig, axes = plt.subplots(1, len(panels), figsize=(TEXT_WIDTH, 1.25), squeeze=False)
    for axis, (condition, backbone) in zip(axes[0], panels):
        clean = [p for p in points if p["backbone"] == backbone and p["condition"] == "baseline"]
        moved = [p for p in points if p["backbone"] == backbone and p["condition"] == condition]
        for group, size, alpha, marker, width in ((clean, 1.2, 0.28, "o", 0), (moved, 2.6, 0.9, "x", 0.4)):
            xs = np.array([float(p["x"]) for p in group])
            ys = np.array([float(p["y"]) for p in group])
            labels = np.array([int(p["label"]) for p in group])
            for class_idx in range(10):
                mask = labels == class_idx
                if mask.any():
                    axis.scatter(xs[mask], ys[mask], s=size, alpha=alpha, marker=marker,
                                 c=CLASS_COLOURS[class_idx], linewidths=width)
        axis.set_title(f"{BACKBONE_LABELS[backbone]}\n{titles.get(condition, condition)}",
                       fontsize=6.3, pad=2)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#cccccc")
            spine.set_visible(True)

    handles = [plt.Line2D([], [], marker="o", linestyle="", color=CLASS_COLOURS[i], markersize=3,
                          label=name) for i, name in enumerate(STL10_CLASSES)]
    handles += [plt.Line2D([], [], marker="x", linestyle="", color="#555555", markersize=3.5,
                           label="transformed"),
                plt.Line2D([], [], marker="o", linestyle="", color="#555555", alpha=0.35,
                           markersize=3, label="clean")]
    fig.legend(handles=handles, loc="lower center", ncol=12, frameon=False, fontsize=6,
               bbox_to_anchor=(0.5, -0.16), handletextpad=0.1, columnspacing=0.6)
    fig.tight_layout(w_pad=0.3)
    save(fig, out_dir, "fig4_tsne_strip")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_dir")
    parser.add_argument("--conflict-dir", default=None,
                        help="conflict_dataset folder, for the qualitative examples figure")
    parser.add_argument("--out", default=None, help="default: <analysis_dir>/figures")
    parser.add_argument("--candidates", action="store_true",
                        help="write contact sheets of cue-conflict candidates and exit")
    parser.add_argument("--cue-ids", nargs="*", default=None,
                        help="specific conflict ids to show in fig3, in order")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else Path(args.analysis_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    conflict_dir = args.conflict_dir
    if conflict_dir is None:
        from assignment_01.task1.config import task_config
        conflict_dir = task_config.TASK_CONFLICT_DATASET_DIR

    if args.candidates:
        figure_cue_candidates(args.analysis_dir, conflict_dir, out_dir)
        return

    figure_interventions(args.analysis_dir, out_dir)
    figure_cue_and_representation(args.analysis_dir, out_dir)

    conflict_dir = args.conflict_dir
    if conflict_dir is None:
        from assignment_01.task1.config import task_config
        conflict_dir = task_config.TASK_CONFLICT_DATASET_DIR
    if (Path(conflict_dir) / "manifest.json").is_file():
        figure_cue_examples(args.analysis_dir, conflict_dir, out_dir, conflict_ids=args.cue_ids)
        if not args.cue_ids:
            figure_cue_examples_compact(args.analysis_dir, conflict_dir, out_dir)
    else:
        print(f"  skipped fig3: no manifest.json under {conflict_dir}")

    figure_tsne(args.analysis_dir, out_dir)
    figure_tsne_strip(args.analysis_dir, out_dir)

    print(f"\nFigures in {out_dir}")


if __name__ == "__main__":
    main()
