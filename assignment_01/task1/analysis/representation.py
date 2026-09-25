"""Joint t-SNE projections of clean and transformed backbone features."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from assignment_01.task1.config import task_config

DEFAULT_CONDITIONS = ("grey_scale", "cue_conflict", "translation_right_32", "patch_shuffle")

CONDITION_TITLES = {
    "grey_scale": "Grayscale",
    "hue": "Hue rotation",
    "cue_conflict": "Cue conflict",
    "patch_shuffle": "Patch shuffle (4x4)",
}

# Eight validated categorical slots plus two neutrals for the remaining classes.
# Class names are written at each clean-cluster centroid, so identity does not
# rely on color alone.
CLASS_COLORS = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
    "#008300", "#4a3aa7", "#e34948", "#7a7a7a", "#8c5a2b",
]

PCA_COMPONENTS = 50
TSNE_SETTINGS = {
    "n_components": 2,
    "perplexity": 30.0,
    "init": "pca",
    "learning_rate": "auto",
    "metric": "euclidean",
}


def condition_title(condition):
    if condition.startswith("translation_"):
        _, direction, displacement = condition.split("_")
        return f"Translation {direction} {displacement}px"
    return CONDITION_TITLES.get(condition, condition)


def _labels(result):
    if "content_labels" in result:
        return torch.as_tensor(result["content_labels"], dtype=torch.long).numpy()
    return torch.as_tensor(result["y_true"], dtype=torch.long).numpy()


def _sample_ids(result):
    if "conflict_ids" in result:
        return list(result["conflict_ids"])
    return [int(i) for i in torch.as_tensor(result["image_ids"]).tolist()]


def fit_joint_projection(baseline_result, transformed_results, seed=task_config.SEED):
    """Fit one joint t-SNE for a backbone's clean and transformed features."""
    blocks = [("baseline", baseline_result)] + list(transformed_results.items())

    features, records = [], []
    for condition, result in blocks:
        block = torch.as_tensor(result["features"], dtype=torch.float32)
        if not torch.isfinite(block).all():
            raise ValueError(f"{condition}: non-finite features.")
        features.append(F.normalize(block, dim=1))

        for sample_id, label in zip(_sample_ids(result), _labels(result)):
            records.append({"condition": condition, "sample_id": sample_id, "label": int(label)})

    stacked = torch.cat(features, dim=0).numpy().astype(np.float64)

    dims = feature_dims = stacked.shape[1]
    n_pca = min(PCA_COMPONENTS, feature_dims, stacked.shape[0])
    if n_pca < feature_dims:
        stacked = PCA(n_components=n_pca, random_state=seed).fit_transform(stacked)
        dims = n_pca

    perplexity = min(TSNE_SETTINGS["perplexity"], (stacked.shape[0] - 1) / 3)
    settings = {**TSNE_SETTINGS, "perplexity": perplexity, "random_state": seed}
    coords = TSNE(**settings).fit_transform(stacked)

    for record, (x, y) in zip(records, coords):
        record["x"], record["y"] = float(x), float(y)

    counts = {}
    for record in records:
        counts[record["condition"]] = counts.get(record["condition"], 0) + 1

    settings = {
        "method": "t-SNE (scikit-learn)",
        **settings,
        "preprocessing": f"L2 normalization, PCA to {dims} dimensions" if dims < feature_dims
                         else "L2 normalization",
        "feature_dim": int(feature_dims),
        "points_per_condition": counts,
        "note": "Coordinates are only comparable within one backbone's fit.",
    }
    return records, settings


def plot_backbone_projection(axes, records, backbone_label, classes, conditions):
    """Draw one row: each panel shows clean points plus one intervention."""
    clean = [r for r in records if r["condition"] == "baseline"]
    clean_xy = np.array([[r["x"], r["y"]] for r in clean])
    clean_labels = np.array([r["label"] for r in clean])

    for ax, condition in zip(axes, conditions):
        moved = [r for r in records if r["condition"] == condition]
        moved_xy = np.array([[r["x"], r["y"]] for r in moved])
        moved_labels = np.array([r["label"] for r in moved])

        for class_idx in range(len(classes)):
            color = CLASS_COLORS[class_idx % len(CLASS_COLORS)]
            mask = clean_labels == class_idx
            ax.scatter(clean_xy[mask, 0], clean_xy[mask, 1], s=9, c=color, alpha=0.30,
                       marker="o", linewidths=0)
            mask = moved_labels == class_idx
            ax.scatter(moved_xy[mask, 0], moved_xy[mask, 1], s=16, c=color, alpha=0.9,
                       marker="x", linewidths=0.9)

        # Direct class labels at clean-cluster medians.
        for class_idx, name in enumerate(classes):
            mask = clean_labels == class_idx
            if mask.any():
                cx, cy = np.median(clean_xy[mask], axis=0)
                ax.text(cx, cy, name, fontsize=6.5, ha="center", va="center", color="#222222",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

        ax.set_title(f"{backbone_label}: {condition_title(condition)}", fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#cccccc")


def add_marker_legend(fig):
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="#555555", alpha=0.4, markersize=5,
                   label="clean"),
        plt.Line2D([], [], marker="x", linestyle="", color="#555555", markersize=5,
                   label="transformed"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8)


def build_projections(results_by_backbone, output_dir, backbone_labels, classes,
                      conditions=DEFAULT_CONDITIONS, seed=task_config.SEED):
    """results_by_backbone: backbone -> {"baseline": result, condition: result, ...}."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_settings, rows_out = {}, []
    fig, grid = plt.subplots(len(results_by_backbone), len(conditions),
                             figsize=(3.1 * len(conditions), 3.1 * len(results_by_backbone)),
                             squeeze=False)

    for row_axes, (backbone, results) in zip(grid, results_by_backbone.items()):
        transformed = {condition: results[condition] for condition in conditions}
        records, settings = fit_joint_projection(results["baseline"], transformed, seed=seed)
        all_settings[backbone] = settings

        for record in records:
            rows_out.append({"backbone": backbone, **record})

        plot_backbone_projection(row_axes, records, backbone_labels[backbone], classes, conditions)

        single, single_axes = plt.subplots(1, len(conditions), figsize=(3.1 * len(conditions), 3.4),
                                           squeeze=False)
        plot_backbone_projection(single_axes[0], records, backbone_labels[backbone], classes, conditions)
        add_marker_legend(single)
        single.tight_layout(rect=(0, 0.05, 1, 1))
        single.savefig(output_dir / f"tsne_{backbone}.png", dpi=200)
        plt.close(single)

    add_marker_legend(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output_dir / "tsne_all_backbones.png", dpi=200)
    plt.close(fig)

    with (output_dir / "tsne_settings.json").open("w", encoding="utf-8") as file:
        json.dump({"conditions": list(conditions), "backbones": all_settings}, file, indent=2)

    return rows_out, all_settings
