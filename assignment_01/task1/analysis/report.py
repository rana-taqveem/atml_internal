import csv
import torch
from pathlib import Path

from assignment_01.task1.config import task_config as TaskConfig
from assignment_01.task1.analysis.metrics import ( calculate_cue_metrics, summarize_metrics, 
                                                  compare_with_baseline,)

def load_inference_result(results_dir, model_name, transformation):
    """Load one saved inference result and check its identity."""
    results_dir = Path(results_dir)

    filename = f"{model_name}_{transformation}_results.pt"
    result_path = results_dir / filename

    if not result_path.is_file():
        raise FileNotFoundError(f"Missing inference result: {result_path}")

    result = torch.load(
        result_path,
        map_location="cpu",
        weights_only=True,
    )

    if result["model_name"] != model_name:
        raise ValueError(f"Model name mismatch in {result_path}")

    if result["transformation"] != transformation:
        raise ValueError(f"Transformation mismatch in {result_path}")

    return result

def build_baseline_report(results_dir, output_path=None):

    model_names = [
        TaskConfig.RESNET50,
        TaskConfig.VIT_B_16,
        TaskConfig.CLIP_VIT_B_32,
        f"{TaskConfig.CLIP_VIT_B_32}_zero_shot"
    ]
    
    results = []
    
    for model_name in model_names:
        result = load_inference_result(results_dir, model_name, transformation="baseline")
        results.append(result)
        
    validate_shared_test_set(results)
    output_dir = Path(results_dir) / "analysis" if output_path is None else Path(output_path)
    output_path = output_dir / "baseline_classification_report.csv"
    
    rows = save_classification_table(results, output_path)
    
    print(f"Saved baseline classification report to {output_path}")
    
    return rows

def validate_shared_test_set(results):
    """Check that model results describe the same labeled test images."""
    if not results:
        raise ValueError("No results were provided.")

    reference_classes = results[0]["classes"]
    reference_ids = None
    reference_labels = None

    for result in results:
        model_name = result["model_name"]
        ids = torch.as_tensor(result["image_ids"], device="cpu")
        labels = torch.as_tensor(result["y_true"], device="cpu")

        if ids.ndim != 1 or labels.ndim != 1:
            raise ValueError(f"{model_name}: IDs and labels must be 1D.")

        if ids.numel() == 0 or ids.numel() != labels.numel():
            raise ValueError(f"{model_name}: invalid ID/label counts.")

        if ids.unique().numel() != ids.numel():
            raise ValueError(f"{model_name}: duplicate image IDs.")

        if result["classes"] != reference_classes:
            raise ValueError(f"{model_name}: class ordering differs.")

        sorted_ids, order = ids.sort()
        sorted_labels = labels[order]

        if reference_ids is None:
            reference_ids = sorted_ids
            reference_labels = sorted_labels
        else:
            if not torch.equal(sorted_ids, reference_ids):
                raise ValueError(f"{model_name}: test images differ.")

            if not torch.equal(sorted_labels, reference_labels):
                raise ValueError(f"{model_name}: image labels differ.")

def summarize_intervention(baseline_result, transformation_result):
    if baseline_result["transformation"] != "baseline":
        raise ValueError("Baseline result is not a baseline transformation.")
    
    if baseline_result["model_name"] != transformation_result["model_name"]:
        raise ValueError("Model names do not match between baseline and transformation results.")
        
    if baseline_result["head_checkpoint"] != transformation_result["head_checkpoint"]:
        raise ValueError("Head checkpoints do not match between baseline and transformation results.")
        
    comparison = compare_with_baseline(baseline_result, transformation_result)
    summary = summarize_metrics(transformation_result)
    
    return {**summary, **comparison}

def build_color_comparison_rows(results_dir):
    
    model_names = [
        TaskConfig.RESNET50,
        TaskConfig.VIT_B_16,
        TaskConfig.CLIP_VIT_B_32,
        f"{TaskConfig.CLIP_VIT_B_32}_zero_shot"
    ]
    
    rows = []
    all_results = []
    
    for model_name in model_names:
        baseline_result = load_inference_result(results_dir, model_name, transformation="baseline")
        all_results.append(baseline_result)
        
        for transformation in ("grey_scale", "hue"):
            transformation_result = load_inference_result(results_dir, model_name, transformation=transformation)
            all_results.append(transformation_result)
        
            row = summarize_intervention(baseline_result, transformation_result)
            rows.append(row)
        
    validate_shared_test_set(all_results)
        
    return rows

def save_rows_to_csv(rows, output_path):
    if not rows:
        raise ValueError("No rows to save.")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved comparison report to {output_path}")
    
    return rows

def save_classification_table(results, output_path):
    """Summarize inference records and save one row per model/condition."""
    rows = [summarize_metrics(result) for result in results]

    if not rows:
        raise ValueError("No inference results were provided.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return save_rows_to_csv(rows, output_path)
  
def summarize_cue_conflicts(result):
    
    logits = torch.as_tensor(result["logits"], device="cpu")
    content_labels = torch.as_tensor(result["content_labels"], device="cpu")
    style_labels = torch.as_tensor(result["style_labels"], device="cpu")
    
    if logits.ndim != 2:
        raise ValueError("Logits must be a 2D tensor.")
    
    if logits.shape[1] == 0:
        raise ValueError("Logits must have shape (N, C) with C > 0.")

    if not torch.isfinite(logits).all():
        raise ValueError("Logits contain non-finite values.")
    
    predictions = torch.argmax(logits, dim=1)
    
    metrics = calculate_cue_metrics(
        predictions=predictions,
        content_labels=content_labels,
        style_labels=style_labels
    )
    
    num_classes = logits.shape[1]
    
    if (content_labels >= num_classes).any() or (style_labels >= num_classes).any():
        raise ValueError("Content or style labels exceed number of classes.")
    
    return {
        "model_name": result["model_name"],
        **metrics
    }
    
def build_cue_conflict_report(results_dir, output_dir=None):
    """Validate shared conflicts and export one summary per model."""
    model_names = [
        TaskConfig.RESNET50,
        TaskConfig.VIT_B_16,
        TaskConfig.CLIP_VIT_B_32,
        f"{TaskConfig.CLIP_VIT_B_32}_zero_shot",
    ]

    rows = []
    direction_rows = []
    reference_records = None
    reference_classes = None

    for model_name in model_names:
        result = load_inference_result(
            results_dir, model_name, "cue_conflict"
        )

        row = summarize_cue_conflicts(result)
        conflict_ids = result["conflict_ids"]

        fields = [
            result["content_ids"],
            result["style_ids"],
            result["content_labels"],
            result["style_labels"],
        ]

        if len(conflict_ids) != row["n_total"]:
            raise ValueError("Conflict ID count differs from predictions.")

        if len(set(conflict_ids)) != len(conflict_ids):
            raise ValueError("Duplicate conflict IDs.")

        if any(len(values) != len(conflict_ids) for values in fields):
            raise ValueError("Conflict metadata lengths differ.")

        records = {
            conflict_id: tuple(int(value) for value in values)
            for conflict_id, *values in zip(conflict_ids, *fields)
        }

        classes = list(result["classes"])

        if reference_records is None:
            reference_records = records
            reference_classes = classes
        elif records != reference_records or classes != reference_classes:
            raise ValueError("Models used different conflicts or class ordering.")

        rows.append(row)
        direction_rows.extend(summarize_cue_directions(result))

    output_dir = Path(results_dir) / "analysis" if output_dir is None else Path(output_dir)

    save_rows_to_csv(rows,
                     output_dir / "cue_conflict_report.csv",
    )

    save_rows_to_csv(direction_rows,
                     output_dir / "cue_conflict_directions.csv",
    )

    return rows

def summarize_cue_directions(result):
    
    summarize_cue_conflicts(result)  # Validate the full result first.

    logits = torch.as_tensor(result["logits"], device="cpu")
    content = torch.as_tensor(result["content_labels"], device="cpu")
    style = torch.as_tensor(result["style_labels"], device="cpu")

    directions = sorted(set(zip(content.tolist(), style.tolist())))
    rows = []

    for content_class, style_class in directions:
        mask = (content == content_class) & (style == style_class)

        subset = {
            "model_name": result["model_name"],
            "logits": logits[mask],
            "content_labels": content[mask],
            "style_labels": style[mask],
        }

        row = summarize_cue_conflicts(subset)
        row.update({
            "content_class": result["classes"][int(content_class)],
            "style_class": result["classes"][int(style_class)],
        })
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Full Task 1 evidence: shared definitions
# ---------------------------------------------------------------------------

import argparse
import json
import sys

import numpy as np
from matplotlib import pyplot as plt

from assignment_01.task1.data.translation_dataset import get_translation_conditions
from assignment_01.task1.data.transforms import apply_translation
from assignment_01.task1.analysis import feature_similarity as fs
from assignment_01.task1.analysis.representation import (
    DEFAULT_CONDITIONS as TSNE_CONDITIONS,
    build_projections,
)

ZERO_SHOT = f"{TaskConfig.CLIP_VIT_B_32}_zero_shot"
DECISION_METHODS = [TaskConfig.RESNET50, TaskConfig.VIT_B_16, TaskConfig.CLIP_VIT_B_32, ZERO_SHOT]
BACKBONES = [TaskConfig.RESNET50, TaskConfig.VIT_B_16, TaskConfig.CLIP_VIT_B_32]

METHOD_LABELS = {
    TaskConfig.RESNET50: "ResNet-50 head",
    TaskConfig.VIT_B_16: "ViT-B/16 head",
    TaskConfig.CLIP_VIT_B_32: "CLIP ViT-B/32 head",
    ZERO_SHOT: "CLIP ViT-B/32 zero-shot",
}
BACKBONE_LABELS = {
    TaskConfig.RESNET50: "ResNet-50",
    TaskConfig.VIT_B_16: "ViT-B/16",
    TaskConfig.CLIP_VIT_B_32: "CLIP ViT-B/32",
}
# Validated categorical slots 1-4, each paired with its own marker.
METHOD_STYLES = {
    TaskConfig.RESNET50: ("#2a78d6", "o"),
    TaskConfig.VIT_B_16: ("#eb6834", "s"),
    TaskConfig.CLIP_VIT_B_32: ("#1baf7a", "^"),
    ZERO_SHOT: ("#eda100", "D"),
}

INTERVENTIONS = ("grey_scale", "hue", "patch_shuffle")
TRANSLATION_IDS = [condition["condition_id"] for condition in get_translation_conditions()]
REPRESENTATION_CONDITIONS = ("grey_scale", "hue", "patch_shuffle", *TRANSLATION_IDS, "cue_conflict")

_RESULT_CACHE = {}


def load_result(results_dir, model_name, transformation):
    """Cached load_inference_result; each file is read once per report run."""
    key = (str(results_dir), model_name, transformation)
    if key not in _RESULT_CACHE:
        _RESULT_CACHE[key] = load_inference_result(results_dir, model_name, transformation)
    return _RESULT_CACHE[key]


def missing_files(results_dir, methods, conditions):
    names = [f"{m}_{c}_results.pt" for m in methods for c in conditions]
    return [name for name in names if not (Path(results_dir) / name).is_file()]


def fmt(value, digits=2):
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows, columns, headers=None):
    headers = headers or columns
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(columns)]
    names = {**METHOD_LABELS, **BACKBONE_LABELS}
    for row in rows:
        cells = [names.get(row.get(c), row.get(c)) if c in ("model_name", "backbone", "method") else row.get(c)
                 for c in columns]
        lines.append("| " + " | ".join(fmt(cell) for cell in cells) + " |")
    return "\n".join(lines)


def style_axis(ax):
    ax.grid(True, color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#999999")
    ax.tick_params(colors="#444444", labelsize=8)


# ---------------------------------------------------------------------------
# Steps 1, 2 and 5: clean, color and patch-shuffle comparison
# ---------------------------------------------------------------------------

def build_intervention_rows(results_dir, transformations=INTERVENTIONS):
    """Absolute metrics plus own-baseline change and consistency per method."""
    rows = []
    all_results = []

    for model_name in DECISION_METHODS:
        baseline_result = load_result(results_dir, model_name, "baseline")
        all_results.append(baseline_result)

        for transformation in transformations:
            transformation_result = load_result(results_dir, model_name, transformation)
            all_results.append(transformation_result)
            rows.append(summarize_intervention(baseline_result, transformation_result))

    validate_shared_test_set(all_results)
    return rows


def compact_comparison(baseline_rows, intervention_rows, transformations=INTERVENTIONS):
    """One row per method: clean metrics, then accuracy/change/consistency per intervention."""
    table = []
    for baseline in baseline_rows:
        model_name = baseline["model_name"]
        row = {
            "method": METHOD_LABELS.get(model_name, model_name),
            "baseline_top1_pct": baseline["top_1_accuracy_pct"],
            "baseline_macro_f1_pct": baseline["macro_f1_pct"],
            "baseline_mean_max_confidence": baseline["mean_max_confidence"],
        }
        for transformation in transformations:
            match = [r for r in intervention_rows
                     if r["model_name"] == model_name and r["transformation"] == transformation]
            if len(match) != 1:
                raise ValueError(f"Expected one {transformation} row for {model_name}.")
            match = match[0]
            row[f"{transformation}_top1_pct"] = match["transformed_accuracy_pct"]
            row[f"{transformation}_change_pp"] = match["accuracy_change_pp"]
            row[f"{transformation}_consistency_pct"] = match["prediction_consistency_pct"]
            row[f"{transformation}_macro_f1_pct"] = match["macro_f1_pct"]
            row[f"{transformation}_mean_max_confidence"] = match["mean_max_confidence"]
        table.append(row)
    return table


# ---------------------------------------------------------------------------
# Step 4: translation
# ---------------------------------------------------------------------------

def check_zero_translation_identity(seed=TaskConfig.SEED):
    """Zero displacement must return the image unchanged (baseline reuse)."""
    generator = torch.Generator().manual_seed(seed)
    image = torch.rand(3, 224, 224, generator=generator)
    return bool(torch.equal(apply_translation(image, 0, 0), image))


def build_translation_rows(results_dir):
    """Per-direction rows and direction-averaged rows for 0/8/16/32 px."""
    if not check_zero_translation_identity():
        raise RuntimeError("apply_translation(x, 0, 0) changed the image; baseline reuse is invalid.")

    direction_rows, summary_rows = [], []

    for model_name in DECISION_METHODS:
        baseline_result = load_result(results_dir, model_name, "baseline")
        identity = compare_with_baseline(baseline_result, baseline_result)
        direction_rows.append({
            "model_name": model_name, "condition": "baseline", "displacement": 0, "direction": "none",
            "translation_x": 0, "translation_y": 0,
            "accuracy_pct": identity["transformed_accuracy_pct"],
            "accuracy_change_pp": identity["accuracy_change_pp"],
            "prediction_consistency_pct": identity["prediction_consistency_pct"],
        })

        for condition in get_translation_conditions():
            result = load_result(results_dir, model_name, condition["condition_id"])
            metadata = result["metadata"]
            for key in ("displacement", "direction", "translation_x", "translation_y"):
                if metadata.get(key) != condition[key]:
                    raise ValueError(f"{model_name} {condition['condition_id']}: metadata {key} mismatch.")

            comparison = summarize_intervention(baseline_result, result)
            direction_rows.append({
                "model_name": model_name, "condition": condition["condition_id"],
                "displacement": condition["displacement"], "direction": condition["direction"],
                "translation_x": condition["translation_x"], "translation_y": condition["translation_y"],
                "accuracy_pct": comparison["transformed_accuracy_pct"],
                "accuracy_change_pp": comparison["accuracy_change_pp"],
                "prediction_consistency_pct": comparison["prediction_consistency_pct"],
            })

        for displacement in (0, *TaskConfig.TRANSLATION_DISPLACEMENTS):
            group = [r for r in direction_rows
                     if r["model_name"] == model_name and r["displacement"] == displacement]
            expected = 1 if displacement == 0 else 4
            if len(group) != expected:
                raise ValueError(f"{model_name}: expected {expected} direction(s) at {displacement}px.")

            accuracy = [r["accuracy_pct"] for r in group]
            consistency = [r["prediction_consistency_pct"] for r in group]
            summary_rows.append({
                "model_name": model_name, "displacement": displacement, "n_directions": len(group),
                "mean_accuracy_pct": float(np.mean(accuracy)),
                "mean_accuracy_change_pp": float(np.mean([r["accuracy_change_pp"] for r in group])),
                "mean_consistency_pct": float(np.mean(consistency)),
                "min_accuracy_pct": min(accuracy), "max_accuracy_pct": max(accuracy),
                "min_consistency_pct": min(consistency), "max_consistency_pct": max(consistency),
            })

    return direction_rows, summary_rows


def plot_translation_curves(summary_rows, output_path):
    """Accuracy and consistency vs displacement; bars span the four directions."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    panels = [
        ("mean_accuracy_pct", "min_accuracy_pct", "max_accuracy_pct", "Top-1 accuracy (%)"),
        ("mean_consistency_pct", "min_consistency_pct", "max_consistency_pct", "Prediction consistency (%)"),
    ]
    offsets = dict(zip(DECISION_METHODS, (-0.9, -0.3, 0.3, 0.9)))

    for ax, (mean_key, low_key, high_key, ylabel) in zip(axes, panels):
        for model_name in DECISION_METHODS:
            rows = sorted((r for r in summary_rows if r["model_name"] == model_name),
                          key=lambda r: r["displacement"])
            if not rows:
                continue
            color, marker = METHOD_STYLES[model_name]
            x = np.array([r["displacement"] for r in rows], dtype=float) + offsets[model_name]
            y = np.array([r[mean_key] for r in rows])
            low = y - np.array([r[low_key] for r in rows])
            high = np.array([r[high_key] for r in rows]) - y
            ax.errorbar(x, y, yerr=[low, high], color=color, marker=marker, markersize=5.5,
                        linewidth=1.6, elinewidth=0.9, capsize=2.5,
                        markeredgecolor="white", markeredgewidth=0.8,
                        label=METHOD_LABELS[model_name])
        ax.set_xticks([0, 8, 16, 32])
        ax.set_xlabel("Displacement (pixels, mean over 4 directions)", fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.5)
        style_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_translation_directions(direction_rows, output_path):
    """Small multiples: accuracy per direction for each decision method."""
    directions = ["right", "left", "down", "up"]
    colors = dict(zip(directions, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")))
    markers = dict(zip(directions, ("o", "s", "^", "D")))

    fig, axes = plt.subplots(1, len(DECISION_METHODS), figsize=(12, 3.2), sharey=True)
    for ax, model_name in zip(axes, DECISION_METHODS):
        rows = [r for r in direction_rows if r["model_name"] == model_name]
        base = [r for r in rows if r["displacement"] == 0]
        for direction in directions:
            points = base + sorted((r for r in rows if r["direction"] == direction),
                                   key=lambda r: r["displacement"])
            ax.plot([r["displacement"] for r in points], [r["accuracy_pct"] for r in points],
                    color=colors[direction], marker=markers[direction], markersize=5, linewidth=1.4,
                    markeredgecolor="white", markeredgewidth=0.7, label=direction)
        ax.set_title(METHOD_LABELS[model_name], fontsize=9)
        ax.set_xticks([0, 8, 16, 32])
        ax.set_xlabel("Displacement (pixels)", fontsize=8.5)
        style_axis(ax)
    axes[0].set_ylabel("Top-1 accuracy (%)", fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8,
               title="Direction of content shift", title_fontsize=8)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 3: cue conflicts
# ---------------------------------------------------------------------------

def decision_name(prediction, content_label, style_label):
    if prediction == content_label:
        return "shape"
    if prediction == style_label:
        return "texture"
    return "other"


def load_conflict_metadata(conflict_dir):
    """conflict_id -> per-image metadata record, or {} when the folder is absent."""
    if conflict_dir is None or not (Path(conflict_dir) / "manifest.json").is_file():
        return {}
    from assignment_01.task1.data.conflict_dataset import load_conflict_records
    return {record["conflict_id"]: record for record in load_conflict_records(conflict_dir)}


def build_cue_conflict_details(results_dir, conflict_metadata):
    """Per-conflict decisions of every method, plus pooled per-pair summaries."""
    results = {m: load_result(results_dir, m, "cue_conflict") for m in DECISION_METHODS}
    reference = results[DECISION_METHODS[0]]
    classes = list(reference["classes"])
    conflict_ids = list(reference["conflict_ids"])

    content = torch.as_tensor(reference["content_labels"], dtype=torch.long)
    style = torch.as_tensor(reference["style_labels"], dtype=torch.long)
    content_ids = torch.as_tensor(reference["content_ids"], dtype=torch.long)
    style_ids = torch.as_tensor(reference["style_ids"], dtype=torch.long)

    predictions = {}
    for model_name, result in results.items():
        position = {cid: i for i, cid in enumerate(result["conflict_ids"])}
        if set(position) != set(conflict_ids):
            raise ValueError(f"{model_name}: conflict IDs differ from {DECISION_METHODS[0]}.")
        order = torch.tensor([position[cid] for cid in conflict_ids])
        if not torch.equal(torch.as_tensor(result["content_labels"], dtype=torch.long)[order], content):
            raise ValueError(f"{model_name}: content labels differ.")
        predictions[model_name] = torch.as_tensor(result["logits"]).argmax(dim=1)[order]

    detail_rows = []
    for i, conflict_id in enumerate(conflict_ids):
        c, s = int(content[i]), int(style[i])
        record = conflict_metadata.get(conflict_id, {})
        row = {
            "conflict_id": conflict_id,
            "pair": "__".join(classes[k] for k in sorted((c, s))),
            "content_class": classes[c], "style_class": classes[s],
            "content_id": int(content_ids[i]), "style_id": int(style_ids[i]),
            "alpha": record.get("alpha"),
        }
        for model_name in DECISION_METHODS:
            p = int(predictions[model_name][i])
            row[f"{model_name}_prediction"] = classes[p]
            row[f"{model_name}_decision"] = decision_name(p, c, s)
        detail_rows.append(row)

    pair_rows = []
    pairs = sorted({row["pair"] for row in detail_rows})
    for model_name in DECISION_METHODS:
        for pair in pairs:
            mask = torch.tensor([row["pair"] == pair for row in detail_rows])
            metrics = calculate_cue_metrics(predictions[model_name][mask], content[mask], style[mask])
            pair_rows.append({"model_name": model_name, "pair": pair, **metrics})

    return detail_rows, pair_rows


def build_conflict_audit(conflict_dir, output_dir):
    """Generated / selected / removed counts per direction, from the dataset files.

    Rejection reasons for removed candidates are read from an optional
    rejection_reasons.csv (conflict_id,rejection_reason) in the conflict
    folder. A template listing the removed IDs is written when it is absent.
    """
    conflict_dir = Path(conflict_dir)
    records = load_conflict_metadata(conflict_dir)
    summary_path = conflict_dir / "selection_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    removed = [entry["conflict_id"] for entry in summary.get("removed_candidates", [])]

    reasons = {}
    reasons_path = conflict_dir / "rejection_reasons.csv"
    if not reasons_path.is_file():
        # Repo-tracked copy; survives the per-session re-extraction on Colab.
        reasons_path = Path(__file__).resolve().parents[1] / "data" / "conflict_rejection_reasons.csv"
    if reasons_path.is_file():
        with reasons_path.open(encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if (row.get("rejection_reason") or "").strip():
                    reasons[row["conflict_id"].strip()] = row["rejection_reason"].strip()
    elif removed:
        template = Path(output_dir) / "rejection_reasons_TEMPLATE.csv"
        save_rows_to_csv([{"conflict_id": cid, "rejection_reason": ""} for cid in removed], template)

    classes = TaskConfig.STL10_CLASSES

    def direction_of(conflict_id):
        content_part, style_part = conflict_id.split("_")[:2]
        return int(content_part[1:]), int(style_part[1:])

    directions = sorted({direction_of(cid) for cid in records} | {direction_of(cid) for cid in removed})
    rows = []
    for c, s in directions:
        selected = [cid for cid in records if direction_of(cid) == (c, s)]
        dropped = [cid for cid in removed if direction_of(cid) == (c, s)]
        documented = [cid for cid in dropped if cid in reasons]
        rows.append({
            "content_class": classes[c], "style_class": classes[s],
            "generated_candidates": len(selected) + len(dropped),
            "accepted_selected": len(selected),
            "removed_not_selected": len(dropped),
            "removed_with_documented_reason": len(documented),
            "reason_counts": "; ".join(f"{r}={sum(reasons[x] == r for x in documented)}"
                                       for r in sorted({reasons[x] for x in documented})),
            "alpha": ",".join(sorted({str(records[cid]["alpha"]) for cid in selected})),
        })
    all_documented = [cid for cid in removed if cid in reasons]
    rows.append({
        "content_class": "ALL", "style_class": "ALL",
        "generated_candidates": sum(r["generated_candidates"] for r in rows),
        "accepted_selected": sum(r["accepted_selected"] for r in rows),
        "removed_not_selected": sum(r["removed_not_selected"] for r in rows),
        "removed_with_documented_reason": len(all_documented),
        "reason_counts": "; ".join(f"{r}={sum(reasons[x] == r for x in all_documented)}"
                                   for r in sorted({reasons[x] for x in all_documented})),
        "alpha": ",".join(sorted({str(r["alpha"]) for r in records.values()})),
    })
    return rows


CUE_EXAMPLE_CATEGORIES = [
    ("All methods follow shape", lambda d: set(d) == {"shape"}),
    ("All methods follow texture", lambda d: set(d) == {"texture"}),
    ("Methods split between shape and texture", lambda d: {"shape", "texture"} <= set(d)),
    ("Some method predicts another class", lambda d: "other" in d),
]


def select_cue_examples(detail_rows, per_category=2, seed=TaskConfig.SEED):
    """Seeded random picks per decision pattern, so examples are not hand-picked."""
    rng = np.random.default_rng(seed)
    chosen, used = [], set()
    for title, predicate in CUE_EXAMPLE_CATEGORIES:
        pool = sorted(
            (row for row in detail_rows
             if row["conflict_id"] not in used
             and predicate([row[f"{m}_decision"] for m in DECISION_METHODS])),
            key=lambda row: row["conflict_id"],
        )
        if not pool:
            continue
        picks = rng.choice(len(pool), size=min(per_category, len(pool)), replace=False)
        for index in sorted(int(i) for i in picks):
            chosen.append((title, pool[index]))
            used.add(pool[index]["conflict_id"])
    return chosen


def plot_cue_examples(examples, conflict_metadata, conflict_dir, output_path):
    from PIL import Image

    fig, axes = plt.subplots(len(examples), 4, figsize=(10, 2.35 * len(examples)),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.55]}, squeeze=False)
    for (title, row), ax_row in zip(examples, axes):
        record = conflict_metadata[row["conflict_id"]]
        panels = [
            (record["content_image_path"], f"content: {row['content_class']}"),
            (record["style_image_path"], f"style: {row['style_class']}"),
            (record["image_path"], "cue conflict"),
        ]
        for ax, (relative, caption) in zip(ax_row[:3], panels):
            with Image.open(Path(conflict_dir) / relative) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(caption, fontsize=8)
            ax.axis("off")

        text = [title, row["conflict_id"], ""]
        for model_name in DECISION_METHODS:
            text.append(f"{METHOD_LABELS[model_name]}: {row[f'{model_name}_prediction']} "
                        f"({row[f'{model_name}_decision']})")
        ax_row[3].text(0.0, 0.5, "\n".join(text), fontsize=7.8, va="center", ha="left")
        ax_row[3].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cue conflicts below top-1: where does the texture class rank?
# ---------------------------------------------------------------------------

def _class_rank(logits, class_idx):
    """1 = highest score. Ties count in the class's favour."""
    return 1 + int((logits > logits[class_idx]).sum())


def build_texture_rank_rows(results_dir):
    """Rank and probability of the style (texture) class, conflict vs clean.

    For every conflict, the same style class is also scored on the clean
    content image. Clean pairs show how often that class ranks high from
    class similarity alone (e.g. airplane/bird); the change to the conflict
    image is the effect of the transferred texture. Probabilities are softmax
    over each method's own logits, so compare them within a method only;
    ranks are comparable across methods.
    """
    sample_rows, summary_rows = [], []

    for model_name in DECISION_METHODS:
        conflict = load_result(results_dir, model_name, "cue_conflict")
        baseline = load_result(results_dir, model_name, "baseline")

        row_of = {int(i): r for r, i in enumerate(torch.as_tensor(baseline["image_ids"]).tolist())}
        clean_logits = torch.as_tensor(baseline["logits"], dtype=torch.float32)
        clean_probs = torch.softmax(clean_logits, dim=1)
        new_logits = torch.as_tensor(conflict["logits"], dtype=torch.float32)
        new_probs = torch.softmax(new_logits, dim=1)
        content = torch.as_tensor(conflict["content_labels"], dtype=torch.long).tolist()
        style = torch.as_tensor(conflict["style_labels"], dtype=torch.long).tolist()
        content_ids = torch.as_tensor(conflict["content_ids"], dtype=torch.long).tolist()

        rows = []
        for i, conflict_id in enumerate(conflict["conflict_ids"]):
            if content_ids[i] not in row_of:
                raise ValueError(f"{model_name}: content image {content_ids[i]} missing from baseline.")
            j = row_of[content_ids[i]]
            c, s = content[i], style[i]
            prediction = int(new_logits[i].argmax())
            rows.append({
                "method": model_name,
                "conflict_id": conflict_id,
                "content_class": TaskConfig.STL10_CLASSES[c],
                "style_class": TaskConfig.STL10_CLASSES[s],
                "decision": decision_name(prediction, c, s),
                "clean_prediction_is_content": int(clean_logits[j].argmax()) == c,
                "texture_rank_clean": _class_rank(clean_logits[j], s),
                "texture_rank_conflict": _class_rank(new_logits[i], s),
                "texture_prob_clean": float(clean_probs[j, s]),
                "texture_prob_conflict": float(new_probs[i, s]),
                "shape_prob_clean": float(clean_probs[j, c]),
                "shape_prob_conflict": float(new_probs[i, c]),
            })
        sample_rows.extend(rows)

        def pct(values):
            return 100 * float(np.mean(values)) if values else None

        shape_rows = [r for r in rows if r["decision"] == "shape"]
        summary_rows.append({
            "method": model_name,
            "n_conflicts": len(rows),
            "mean_texture_rank_clean": float(np.mean([r["texture_rank_clean"] for r in rows])),
            "mean_texture_rank_conflict": float(np.mean([r["texture_rank_conflict"] for r in rows])),
            "texture_in_top2_clean_pct": pct([r["texture_rank_clean"] <= 2 for r in rows]),
            "texture_in_top2_conflict_pct": pct([r["texture_rank_conflict"] <= 2 for r in rows]),
            "mean_texture_prob_clean": float(np.mean([r["texture_prob_clean"] for r in rows])),
            "mean_texture_prob_conflict": float(np.mean([r["texture_prob_conflict"] for r in rows])),
            "mean_shape_prob_clean": float(np.mean([r["shape_prob_clean"] for r in rows])),
            "mean_shape_prob_conflict": float(np.mean([r["shape_prob_conflict"] for r in rows])),
            # Among conflicts decided by shape: is texture the runner-up?
            "n_shape_decisions": len(shape_rows),
            "texture_runner_up_given_shape_clean_pct": pct([r["texture_rank_clean"] == 2 for r in shape_rows]),
            "texture_runner_up_given_shape_conflict_pct": pct([r["texture_rank_conflict"] == 2 for r in shape_rows]),
            "runner_up_chance_pct": 100 / 9,
        })

    return sample_rows, summary_rows


def plot_texture_rank(summary_rows, output_path):
    """Texture class in the top 2, clean content image vs cue conflict."""
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    methods = [r["method"] for r in summary_rows]
    y = np.arange(len(methods))
    clean = [r["texture_in_top2_clean_pct"] for r in summary_rows]
    conflict = [r["texture_in_top2_conflict_pct"] for r in summary_rows]

    for yi, a, b in zip(y, clean, conflict):
        ax.plot([a, b], [yi, yi], color="#bbbbbb", linewidth=1.5, zorder=1)
    ax.scatter(clean, y, color="#2a78d6", marker="o", s=45, edgecolors="white", linewidths=0.8,
               label="clean content image", zorder=3)
    ax.scatter(conflict, y, color="#eb6834", marker="s", s=45, edgecolors="white", linewidths=0.8,
               label="cue conflict", zorder=3)
    for yi, b in zip(y, conflict):
        ax.text(b + 1.5, yi, f"{b:.1f}", va="center", fontsize=7.5, color="#444444")

    ax.axvline(100 * 2 / 10, color="#999999", linestyle=":", linewidth=1)
    ax.text(100 * 2 / 10, len(methods) - 0.45, " chance (2/10)", fontsize=7, color="#777777", va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABELS[m] for m in methods], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Style (texture) class ranked 1st or 2nd (% of conflicts)", fontsize=8.5)
    style_axis(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 6 and RQ3: CLIP head vs zero-shot, representation stability
# ---------------------------------------------------------------------------

def build_clip_comparison_rows(results_dir, conditions):
    rows = []
    for condition in conditions:
        head = load_result(results_dir, TaskConfig.CLIP_VIT_B_32, condition)
        zero_shot = load_result(results_dir, ZERO_SHOT, condition)
        hp = torch.as_tensor(head["logits"]).argmax(1)
        zp = torch.as_tensor(zero_shot["logits"]).argmax(1)

        if condition == "cue_conflict":
            if list(head["conflict_ids"]) != list(zero_shot["conflict_ids"]):
                raise ValueError("CLIP head and zero-shot conflict orders differ.")
            content = torch.as_tensor(head["content_labels"]).tolist()
            style = torch.as_tensor(head["style_labels"]).tolist()
            head_decisions = [decision_name(int(p), c, s) for p, c, s in zip(hp, content, style)]
            zs_decisions = [decision_name(int(p), c, s) for p, c, s in zip(zp, content, style)]
            rows.append({
                "condition": condition, "n": len(hp),
                "prediction_agreement_pct": 100 * (hp == zp).float().mean().item(),
                "decision_agreement_pct": 100 * float(np.mean([a == b for a, b in zip(head_decisions, zs_decisions)])),
            })
            continue

        if not torch.equal(torch.as_tensor(head["image_ids"]), torch.as_tensor(zero_shot["image_ids"])):
            raise ValueError(f"{condition}: CLIP head and zero-shot image orders differ.")
        labels = torch.as_tensor(head["y_true"], dtype=torch.long)
        head_ok, zs_ok = hp == labels, zp == labels
        rows.append({
            "condition": condition, "n": len(labels),
            "head_accuracy_pct": 100 * head_ok.float().mean().item(),
            "zero_shot_accuracy_pct": 100 * zs_ok.float().mean().item(),
            "prediction_agreement_pct": 100 * (hp == zp).float().mean().item(),
            "decision_agreement_pct": None,
            "both_correct": int((head_ok & zs_ok).sum()),
            "head_only_correct": int((head_ok & ~zs_ok).sum()),
            "zero_shot_only_correct": int((~head_ok & zs_ok).sum()),
            "both_wrong": int((~head_ok & ~zs_ok).sum()),
        })

    # Put a full-width row first so the CSV header covers every column.
    rows.sort(key=lambda r: "head_accuracy_pct" not in r)
    return rows


def build_representation_rows(results_dir, conditions=REPRESENTATION_CONDITIONS):
    """Cosine stability per backbone, sample-level rows and prediction/feature agreement."""
    summary_rows, sample_rows, agreement_rows = [], [], []

    for backbone in BACKBONES:
        methods = [backbone] + ([ZERO_SHOT] if backbone == TaskConfig.CLIP_VIT_B_32 else [])
        for condition in conditions:
            for model_name in methods:
                baseline = load_result(results_dir, model_name, "baseline")
                transformed = load_result(results_dir, model_name, condition)
                cosine, ids, base_rows, new_rows = fs.paired_cosine(baseline, transformed)

                if model_name == backbone:
                    summary_rows.append({"backbone": backbone, "condition": condition,
                                         **fs.summarize_cosine(cosine)})

                rows = fs.sample_rows(model_name, condition, cosine, ids, base_rows, new_rows,
                                      baseline, transformed)
                sample_rows.extend(rows)
                agreement_rows.append(fs.prediction_feature_agreement(rows))

        for displacement in TaskConfig.TRANSLATION_DISPLACEMENTS:
            group = [r for r in summary_rows if r["backbone"] == backbone
                     and r["condition"].startswith("translation_")
                     and r["condition"].endswith(f"_{displacement}")]
            if len(group) == 4:
                summary_rows.append({
                    "backbone": backbone, "condition": f"translation_{displacement}px_mean",
                    "n": sum(r["n"] for r in group),
                    "cosine_stability_mean": float(np.mean([r["cosine_stability_mean"] for r in group])),
                    "cosine_std": None, "cosine_median": None, "cosine_p10": None,
                    "cosine_min": min(r["cosine_min"] for r in group),
                })

    return summary_rows, sample_rows, agreement_rows


def plot_representation_stability(summary_rows, output_path):
    """Dot plot of I_T per intervention; one color/marker per backbone."""
    names = {"grey_scale": "Grayscale", "hue": "Hue rotation", "cue_conflict": "Cue conflict",
             "translation_8px_mean": "Translation 8px", "translation_16px_mean": "Translation 16px",
             "translation_32px_mean": "Translation 32px", "patch_shuffle": "Patch shuffle"}
    present = [c for c in names if any(r["condition"] == c for r in summary_rows)]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    offsets = dict(zip(BACKBONES, (-0.18, 0.0, 0.18)))
    for backbone in BACKBONES:
        color, marker = METHOD_STYLES[backbone]
        xs, ys = [], []
        for y, condition in enumerate(present):
            match = [r for r in summary_rows if r["backbone"] == backbone and r["condition"] == condition]
            if match:
                xs.append(match[0]["cosine_stability_mean"])
                ys.append(y + offsets[backbone])
        ax.scatter(xs, ys, color=color, marker=marker, s=40, edgecolors="white", linewidths=0.8,
                   label=BACKBONE_LABELS[backbone], zorder=3)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels([names[c] for c in present], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Cosine stability $I_T$ (mean over images)", fontsize=8.5)
    style_axis(ax)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all(results_dir, conflict_dir=None, output_dir=None, make_tsne=True):
    """Build every Task 1 table and figure that the saved results allow.

    Sections whose inference files are missing are skipped and listed in
    completeness.json; a validation error inside a section stops the run.
    """
    results_dir = Path(results_dir)
    output_dir = results_dir / "analysis" if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if conflict_dir is not None:
        from assignment_01.task1.data.download import prepare_conflict_dataset
        conflict_dir = prepare_conflict_dataset(conflict_dir)
    conflict_metadata = load_conflict_metadata(conflict_dir)

    all_conditions = ["baseline", *INTERVENTIONS, *TRANSLATION_IDS, "cue_conflict"]
    completeness = {
        "results_dir": str(results_dir),
        "missing_result_files": missing_files(results_dir, DECISION_METHODS, all_conditions),
        "zero_translation_is_identity": check_zero_translation_identity(),
        "skipped_sections": {},
        "written": [],
    }
    summary_md = ["# Task 1 evidence summary", "",
                  "Values come from the CSV files in this folder (shown with 2 decimals; "
                  "the CSVs keep full precision).", ""]

    def section(name, methods, conditions):
        missing = missing_files(results_dir, methods, conditions)
        if missing:
            completeness["skipped_sections"][name] = missing
            print(f"[skip] {name}: {len(missing)} result file(s) missing, e.g. {missing[0]}")
            return False
        print(f"[run ] {name}")
        return True

    def write(rows, filename):
        save_rows_to_csv(rows, output_dir / filename)
        completeness["written"].append(filename)

    # Step 1
    baseline_rows = None
    if section("baseline", DECISION_METHODS, ["baseline"]):
        baseline_rows = build_baseline_report(results_dir, output_path=output_dir)
        completeness["written"].append("baseline_classification_report.csv")
        summary_md += ["## Clean baseline", "",
                       markdown_table(baseline_rows, ["model_name", "n_images", "top_1_accuracy_pct",
                                                      "macro_f1_pct", "mean_max_confidence"],
                                      ["method", "N", "top-1 (%)", "macro-F1 (%)", "mean max confidence"]), ""]

    # Steps 2 and 5
    if baseline_rows and section("interventions", DECISION_METHODS, INTERVENTIONS):
        rows = build_intervention_rows(results_dir)
        write(rows, "intervention_comparison_long.csv")
        compact = compact_comparison(baseline_rows, rows)
        write(compact, "intervention_comparison_compact.csv")
        columns, headers = ["method", "baseline_top1_pct"], ["method", "clean top-1"]
        for t, label in (("grey_scale", "gray"), ("hue", "hue"), ("patch_shuffle", "patch")):
            columns += [f"{t}_top1_pct", f"{t}_change_pp", f"{t}_consistency_pct"]
            headers += [f"{label} top-1", f"{label} change (pp)", f"{label} consistency"]
        summary_md += ["## Clean, color and patch-shuffle comparison (%)", "",
                       markdown_table(compact, columns, headers), ""]

    # Step 4
    if section("translation", DECISION_METHODS, ["baseline", *TRANSLATION_IDS]):
        direction_rows, translation_rows = build_translation_rows(results_dir)
        write(direction_rows, "translation_directions.csv")
        write(translation_rows, "translation_summary.csv")
        plot_translation_curves(translation_rows, output_dir / "translation_curves.png")
        plot_translation_directions(direction_rows, output_dir / "translation_directions.png")
        completeness["written"] += ["translation_curves.png", "translation_directions.png"]
        summary_md += ["## Translation (mean over four directions)", "",
                       markdown_table(translation_rows, ["model_name", "displacement", "mean_accuracy_pct",
                                                         "mean_consistency_pct", "min_accuracy_pct",
                                                         "max_accuracy_pct"],
                                      ["method", "px", "accuracy (%)", "consistency (%)",
                                       "min acc", "max acc"]), ""]

    # Step 3
    if section("cue_conflict", DECISION_METHODS, ["cue_conflict"]):
        cue_rows = build_cue_conflict_report(results_dir, output_dir=output_dir)
        completeness["written"] += ["cue_conflict_report.csv", "cue_conflict_directions.csv"]
        detail_rows, pair_rows = build_cue_conflict_details(results_dir, conflict_metadata)
        write(detail_rows, "cue_conflict_decisions.csv")
        write(pair_rows, "cue_conflict_pairs.csv")
        summary_md += ["## Cue conflicts (pooled over all selected conflicts)", "",
                       markdown_table(cue_rows, ["model_name", "n_total", "n_shape", "n_texture", "n_other",
                                                 "shape_bias_pct", "coverage_pct"],
                                      ["method", "N", "shape", "texture", "other", "shape bias (%)",
                                       "coverage (%)"]), ""]

        if conflict_metadata:
            examples = select_cue_examples(detail_rows)
            plot_cue_examples(examples, conflict_metadata, conflict_dir, output_dir / "cue_conflict_examples.png")
            write([{"category": title, **row} for title, row in examples], "cue_conflict_examples.csv")
            completeness["written"].append("cue_conflict_examples.png")
        else:
            completeness["skipped_sections"]["cue_conflict_examples"] = ["conflict dataset folder not found"]

    if conflict_metadata:
        audit_rows = build_conflict_audit(conflict_dir, output_dir)
        write(audit_rows, "cue_conflict_dataset_audit.csv")
        summary_md += ["## Cue-conflict dataset audit", "",
                       markdown_table(audit_rows, ["content_class", "style_class", "generated_candidates",
                                                   "accepted_selected", "removed_not_selected",
                                                   "removed_with_documented_reason", "alpha"],
                                      ["content", "style", "generated", "selected", "removed",
                                       "removed w/ reason", "alpha"]), ""]

    # Cue conflicts below top-1: texture-class rank, conflict vs clean
    if section("texture_rank", DECISION_METHODS, ["baseline", "cue_conflict"]):
        rank_samples, rank_summary = build_texture_rank_rows(results_dir)
        write(rank_samples, "cue_conflict_texture_rank_samples.csv")
        write(rank_summary, "cue_conflict_texture_rank.csv")
        plot_texture_rank(rank_summary, output_dir / "cue_conflict_texture_rank.png")
        completeness["written"].append("cue_conflict_texture_rank.png")
        summary_md += ["## Texture class below top-1 (clean content image vs cue conflict)", "",
                       markdown_table(rank_summary,
                                      ["method", "mean_texture_rank_clean", "mean_texture_rank_conflict",
                                       "texture_in_top2_clean_pct", "texture_in_top2_conflict_pct",
                                       "texture_runner_up_given_shape_clean_pct",
                                       "texture_runner_up_given_shape_conflict_pct"],
                                      ["method", "mean rank clean", "mean rank conflict", "top-2 clean (%)",
                                       "top-2 conflict (%)", "runner-up | shape, clean (%)",
                                       "runner-up | shape, conflict (%)"]), ""]

    # RQ3: CLIP head vs zero-shot on shared features
    clip_conditions = [c for c in all_conditions
                       if not missing_files(results_dir, [TaskConfig.CLIP_VIT_B_32, ZERO_SHOT], [c])]
    if clip_conditions:
        print("[run ] clip_head_vs_zero_shot")
        write(build_clip_comparison_rows(results_dir, clip_conditions), "clip_head_vs_zero_shot.csv")

    # Step 6
    if section("representation", DECISION_METHODS, ["baseline", *REPRESENTATION_CONDITIONS]):
        summary_rows, sample_rows, agreement_rows = build_representation_rows(results_dir)
        write(summary_rows, "representation_stability.csv")
        write(sample_rows, "representation_samples.csv")
        write(agreement_rows, "prediction_vs_feature.csv")
        plot_representation_stability(summary_rows, output_dir / "representation_stability.png")
        completeness["written"].append("representation_stability.png")
        summary_md += ["## Representation stability I_T (cosine)", "",
                       markdown_table([r for r in summary_rows if not r["condition"].startswith("translation_")
                                       or r["condition"].endswith("_mean")],
                                      ["backbone", "condition", "n", "cosine_stability_mean"],
                                      ["backbone", "condition", "N", "I_T"]), ""]

    if make_tsne and section("tsne", BACKBONES, ["baseline", *TSNE_CONDITIONS]):
        results_by_backbone = {
            backbone: {c: load_result(results_dir, backbone, c) for c in ("baseline", *TSNE_CONDITIONS)}
            for backbone in BACKBONES
        }
        point_rows, _ = build_projections(results_by_backbone, output_dir, BACKBONE_LABELS,
                                          TaskConfig.STL10_CLASSES)
        write(point_rows, "tsne_points.csv")
        completeness["written"] += ["tsne_all_backbones.png", "tsne_settings.json"] + \
                                   [f"tsne_{b}.png" for b in BACKBONES]

    run_manifest = results_dir / "run_manifest.json"
    if run_manifest.is_file():
        completeness["run_manifest"] = json.loads(run_manifest.read_text(encoding="utf-8"))

    with (output_dir / "completeness.json").open("w", encoding="utf-8") as file:
        json.dump(completeness, file, indent=2)

    if completeness["skipped_sections"]:
        summary_md += ["## Missing evidence", ""] + \
                      [f"- {name}: {len(items)} missing item(s)"
                       for name, items in completeness["skipped_sections"].items()]
    (output_dir / "evidence_summary.md").write_text("\n".join(summary_md) + "\n", encoding="utf-8")

    print(f"\nWrote {len(completeness['written'])} artifacts to {output_dir}")
    if completeness["skipped_sections"]:
        print("Skipped sections:", ", ".join(completeness["skipped_sections"]))
    return completeness


def main():
    parser = argparse.ArgumentParser(description="Build Task 1 tables and figures from saved inference results.")
    parser.add_argument("--results-dir", default=TaskConfig.TASK_RESULTS_DIR)
    parser.add_argument("--conflict-dir", default=TaskConfig.TASK_CONFLICT_DATASET_DIR,
                        help="Finalized cue-conflict folder (images, alpha and the dataset audit).")
    parser.add_argument("--output-dir", default=None, help="Default: <results-dir>/analysis")
    parser.add_argument("--skip-tsne", action="store_true")
    args = parser.parse_args()

    completeness = run_all(args.results_dir, args.conflict_dir, args.output_dir, make_tsne=not args.skip_tsne)
    sys.exit(1 if completeness["skipped_sections"] else 0)


if __name__ == "__main__":
    main()
