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
    output_dir = Path(results_dir if output_path is None else output_path)
    output_path = output_dir / "analysis" / "baseline_classification_report.csv"
    
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
    
def build_cue_conflict_report(results_dir):
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

    output_dir = Path(results_dir) / "analysis"

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

if __name__ == "__main__":
    
    results_dir = TaskConfig.TASK_RESULTS_DIR
    
    build_baseline_report(results_dir)
    
    color_rows = build_color_comparison_rows(results_dir)
    color_output_path = Path(results_dir) / "analysis" / "color_comparison_report.csv"
    save_rows_to_csv(color_rows, color_output_path)
    
    print(f"Saved color comparison: {color_output_path}")
    
    build_cue_conflict_report(results_dir)