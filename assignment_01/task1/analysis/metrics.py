import torch
from sklearn.metrics import precision_recall_fscore_support
from assignment_01.task1.config import task_config

def calcultate_metrics(result, top_k=(1, )):
    """Calculate accuracy, macro/per-class metrics, and top-k accuracy."""

    labels = torch.as_tensor(result["y_true"], device="cpu")
    logits = torch.as_tensor(result["logits"], dtype=torch.float32, device="cpu")

    if logits.ndim != 2 or labels.ndim != 1:
        raise ValueError("Logits must be a 2D tensor and labels must be a 1D tensor.")

    if len(labels) == 0 or logits.shape[0] != len(labels):
        raise ValueError("Labels must not be empty and must match the number of logits.")

    if not torch.isfinite(labels).all():
        raise ValueError("labels must not contain NaN or Inf values.")

    if labels.is_floating_point() and not torch.equal(labels, labels.round()):
        raise ValueError("Labels must contain integer class IDs.")

    labels = labels.to(torch.long)

    num_classes = logits.shape[1]

    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError(f"Labels must be in the range [0, {num_classes - 1}].")

    if not torch.isfinite(logits).all():
        raise ValueError("Logits must not contain NaN or infinity.")
    
    probabilities = torch.softmax(logits, dim=1)

    confidence, predictions = torch.max(probabilities, dim=1)
    accuracy = (predictions == labels).float().mean().item()

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels.numpy(),
        predictions.numpy(),
        labels=list(range(num_classes)),
        average=None,
        zero_division=0
    )

    macro_f1 = f1.mean() if len(f1) > 0 else 0.0
    macro_precesion = precision.mean() if len(precision) > 0 else 0.0
    macro_recall = recall.mean() if len(recall) > 0 else 0.0

    per_class = []

    for class_idx in range(num_classes):
        per_class.append({
            "class_idx": class_idx,
            "precision": float(precision[class_idx]),
            "recall": float(recall[class_idx]),
            "f1": float(f1[class_idx]),
            "support": int((labels == class_idx).sum().item())
        })

    ranked_classes = logits.argsort(dim=1, descending=True, stable=True)

    top_k_accuracies = {}

    for k in top_k:
 
        if isinstance(k, bool) or not isinstance(k, int):
            raise ValueError("Each top_k value must be an integer.")

        if not 1 <= k <= num_classes:
            raise ValueError(f"Each top_k value must be between 1 and {num_classes}.")
        
        top_classes = ranked_classes[:, :k]

        matches = top_classes == labels[:, None]

        correct = matches.any(dim=1).float()

        top_k_accuracy = correct.mean().item() * 100
        top_k_accuracies[f"top_{k}_accuracy"] = top_k_accuracy


    return{
        "n_images": len(labels),
        "accuracy_pct": 100 * accuracy,
        "macro_precision": 100 * macro_precesion,
        "macro_recall": 100 * macro_recall,
        "macro_f1": 100 * macro_f1,
        "per_class": per_class,
        "top_k_accuracies": top_k_accuracies,
        "mean_max_confidence": confidence.mean().item()
    }

def compare_with_baseline(baseline_result, transformed_result):
    """Compare paired baseline and transformed predictions by image ID."""


    baseline_ids =  torch.as_tensor(baseline_result["image_ids"], dtype=torch.long, device="cpu")
    transformed_ids = torch.as_tensor(transformed_result["image_ids"], dtype=torch.long, device="cpu")


    if baseline_ids.ndim != 1 or transformed_ids.ndim != 1:
        raise ValueError("Image IDs must be 1D tensors.")

    if len(baseline_ids) == 0 or len(transformed_ids) == 0:
        raise ValueError("Image IDs must not be empty.")

    if baseline_ids.unique().numel() != len(baseline_ids) or transformed_ids.unique().numel() != len(transformed_ids):
        raise ValueError("Image IDs must be unique within each result.")
    
    if list(baseline_result["classes"]) != list(transformed_result["classes"]):
        raise ValueError("Class names and their order must match.")

    baseline_sorted = baseline_ids.sort()
    transformed_sorted = transformed_ids.sort()

    if not torch.equal(baseline_ids.sort().values, transformed_ids.sort().values):
        raise ValueError("Image IDs must match between baseline and transformed results.")

    baseline_logits = torch.as_tensor(baseline_result["logits"], dtype=torch.float32, device="cpu")
    transformed_logits = torch.as_tensor(transformed_result["logits"], dtype=torch.float32, device="cpu")

    baseline_labels = torch.as_tensor(baseline_result["y_true"], dtype=torch.long, device="cpu")
    transformed_labels = torch.as_tensor(transformed_result["y_true"], dtype=torch.long, device="cpu")

    baseline_metrics = calcultate_metrics(baseline_result)
    transformed_metrics = calcultate_metrics(transformed_result)


    if baseline_logits.shape[0] != baseline_ids.numel() or transformed_logits.shape[0] != transformed_ids.numel():
        raise ValueError("Logits must have the same number of rows as image IDs.")

    if baseline_logits.shape[1] != transformed_logits.shape[1]:
        raise ValueError("Logits must have the same number of columns as transformed logits.")

    if list(baseline_labels.shape) != list(transformed_labels.shape):
        raise ValueError("Labels must have the same shape as transformed labels.")

    baseline_logits = baseline_logits[baseline_sorted.indices]
    transformed_logits = transformed_logits[transformed_sorted.indices]

    baseline_labels = baseline_labels[baseline_sorted.indices]
    transformed_labels = transformed_labels[transformed_sorted.indices]


    if not torch.equal(baseline_labels, transformed_labels):
        raise ValueError("Labels must match between baseline and transformed results.")

    baseline_predictions = torch.argmax(baseline_logits, dim=1)
    transformed_predictions = torch.argmax(transformed_logits, dim=1)

    unchanged = baseline_predictions == transformed_predictions

    consistency_pct = unchanged.float().mean().item() * 100
    baseline_accuracy = baseline_metrics["accuracy_pct"]
    transformed_accuracy = transformed_metrics["accuracy_pct"]


    return {
        "n_images": len(baseline_ids),
        "baseline_accuracy_pct": baseline_accuracy,
        "transformed_accuracy_pct": transformed_accuracy,
        "accuracy_change_pp": transformed_accuracy - baseline_accuracy,
        "accuracy_drop_pp": baseline_accuracy - transformed_accuracy,
        "prediction_consistency_pct": consistency_pct
    }

def summarize_metrics(result):
    
    metrics = calcultate_metrics(result)
    
    return {
        "model_name": result["model_name"],
        "transformation": result["transformation"],
        "n_images": metrics["n_images"],
        "top_1_accuracy_pct": metrics["accuracy_pct"],
        "macro_f1_pct": metrics["macro_f1"],
        "mean_max_confidence": metrics["mean_max_confidence"]
    }        
    
def count_cue_decisions(predictions, content_labels, style_labels):

    predictions = torch.as_tensor(predictions, device="cpu")
    content_labels = torch.as_tensor(content_labels, device="cpu")
    style_labels = torch.as_tensor(style_labels, device="cpu")
    
    arrays = (predictions, content_labels, style_labels)
    
    for values in arrays:
        if values.ndim != 1 or values.numel() == 0:
            raise ValueError("Predictions and labels must be non-empty 1D tensors.")
        
        if not torch.isfinite(values).all():
            raise ValueError("Predictions and labels must not contain NaN or Inf values.")
        
        if values.is_floating_point() and not torch.equal(values, values.round()):
            raise ValueError("Predictions and labels must contain integer class IDs.")
        
        if values.min() < 0:
            raise ValueError("Predictions and labels must contain non-negative class IDs.")
        
    if not (predictions.shape == content_labels.shape == style_labels.shape):
        raise ValueError("All three arrays must have the same shape.")

    if (content_labels == style_labels).any():
        raise ValueError("Content and style classes must differ.")

    shape_mask = predictions == content_labels
    texture_mask = predictions == style_labels
    other_mask = ~(shape_mask | texture_mask)

    return {
        "n_total": predictions.numel(),
        "n_shape": int(shape_mask.sum().item()),
        "n_texture": int(texture_mask.sum().item()),
        "n_other": int(other_mask.sum().item()),
    }
    
def calculate_cue_metrics(predictions, content_labels, style_labels):
    counts = count_cue_decisions(predictions, content_labels, style_labels)
    
    n_total = counts["n_total"]
    n_shape = counts["n_shape"]
    n_texture = counts["n_texture"]
  
    
    n_shape_or_texture = n_shape + n_texture
    
    if n_shape_or_texture == 0:
        shape_bias_pct = None
    else:
        shape_bias_pct = 100 * n_shape / n_shape_or_texture
        
    coverage_pct = 100 * n_shape_or_texture / n_total
    
    return {
        **counts,
        "shape_bias_pct": shape_bias_pct,
        "coverage_pct": coverage_pct
    }
    
