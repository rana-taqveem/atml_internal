"""Cosine similarity between clean and transformed feature representations."""

import torch
import torch.nn.functional as F


def _features(result, name):
    features = torch.as_tensor(result["features"], dtype=torch.float32, device="cpu")

    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"{name}: features must have shape [N, D] with N > 0.")

    if not torch.isfinite(features).all():
        raise ValueError(f"{name}: features contain NaN or Inf values.")

    if (features.norm(dim=1) == 0).any():
        raise ValueError(f"{name}: zero-norm feature vectors make cosine similarity undefined.")

    return features


def pair_rows(baseline_result, transformed_result):
    """Pair ordinary samples by ID and cue conflicts by content ID."""
    baseline_ids = torch.as_tensor(baseline_result["image_ids"], dtype=torch.long, device="cpu")

    if baseline_ids.unique().numel() != baseline_ids.numel():
        raise ValueError("Baseline image IDs must be unique.")

    row_of = {int(image_id): row for row, image_id in enumerate(baseline_ids.tolist())}

    if "conflict_ids" in transformed_result:
        reference_ids = torch.as_tensor(transformed_result["content_ids"], dtype=torch.long).tolist()
        sample_ids = list(transformed_result["conflict_ids"])
    else:
        reference_ids = torch.as_tensor(transformed_result["image_ids"], dtype=torch.long).tolist()
        sample_ids = list(reference_ids)

        if len(set(reference_ids)) != len(reference_ids):
            raise ValueError("Transformed image IDs must be unique.")

        if set(reference_ids) != set(row_of):
            raise ValueError("Baseline and transformed image sets differ.")

    missing = [image_id for image_id in reference_ids if image_id not in row_of]
    if missing:
        raise ValueError(f"{len(missing)} transformed samples have no clean counterpart, e.g. {missing[:5]}.")

    baseline_rows = torch.tensor([row_of[image_id] for image_id in reference_ids], dtype=torch.long)
    transformed_rows = torch.arange(len(reference_ids), dtype=torch.long)

    return baseline_rows, transformed_rows, sample_ids


def paired_cosine(baseline_result, transformed_result):
    """Per-sample cosine similarity [N] and the sample IDs it refers to."""
    if baseline_result.get("backbone_name", baseline_result["model_name"]) != \
            transformed_result.get("backbone_name", transformed_result["model_name"]):
        raise ValueError("Cosine stability requires features from the same backbone.")

    clean = _features(baseline_result, "baseline")
    transformed = _features(transformed_result, "transformed")

    if clean.shape[1] != transformed.shape[1]:
        raise ValueError("Feature dimensions differ between baseline and transformed results.")

    baseline_rows, transformed_rows, sample_ids = pair_rows(baseline_result, transformed_result)

    if transformed_rows.numel() != transformed.shape[0]:
        raise ValueError("Sample identifiers do not match the number of transformed feature rows.")

    cosine = F.cosine_similarity(clean[baseline_rows], transformed[transformed_rows], dim=1)
    return cosine, sample_ids, baseline_rows, transformed_rows


def summarize_cosine(cosine):
    values = cosine.double()
    return {
        "n": int(values.numel()),
        "cosine_stability_mean": float(values.mean()),
        "cosine_std": float(values.std(unbiased=False)),
        "cosine_median": float(values.median()),
        "cosine_p10": float(torch.quantile(values, 0.10)),
        "cosine_min": float(values.min()),
    }


def sample_rows(method_name, condition, cosine, sample_ids, baseline_rows, transformed_rows,
                baseline_result, transformed_result):
    """One row per sample: similarity, clean/transformed predictions and labels."""
    clean_logits = torch.as_tensor(baseline_result["logits"], dtype=torch.float32)
    new_logits = torch.as_tensor(transformed_result["logits"], dtype=torch.float32)

    clean_pred = clean_logits.argmax(dim=1)[baseline_rows]
    new_pred = new_logits.argmax(dim=1)[transformed_rows]
    labels = torch.as_tensor(baseline_result["y_true"], dtype=torch.long)[baseline_rows]

    rows = []
    for i, sample_id in enumerate(sample_ids):
        rows.append({
            "method": method_name,
            "condition": condition,
            "sample_id": sample_id,
            "label": int(labels[i]),
            "cosine": float(cosine[i]),
            "clean_prediction": int(clean_pred[i]),
            "transformed_prediction": int(new_pred[i]),
            "prediction_unchanged": bool(clean_pred[i] == new_pred[i]),
            "clean_correct": bool(clean_pred[i] == labels[i]),
            "transformed_correct": bool(new_pred[i] == labels[i]),
        })
    return rows


def prediction_feature_agreement(rows):
    """Compare feature similarity for changed and unchanged predictions."""
    unchanged = torch.tensor([row["cosine"] for row in rows if row["prediction_unchanged"]], dtype=torch.float64)
    changed = torch.tensor([row["cosine"] for row in rows if not row["prediction_unchanged"]], dtype=torch.float64)

    def stat(values, fn):
        return float(fn(values)) if values.numel() else None

    return {
        "method": rows[0]["method"],
        "condition": rows[0]["condition"],
        "n": len(rows),
        "n_prediction_unchanged": int(unchanged.numel()),
        "n_prediction_changed": int(changed.numel()),
        "prediction_consistency_pct": 100 * unchanged.numel() / len(rows),
        "mean_cosine_unchanged": stat(unchanged, torch.mean),
        "mean_cosine_changed": stat(changed, torch.mean),
        "median_cosine_unchanged": stat(unchanged, torch.median),
        "median_cosine_changed": stat(changed, torch.median),
    }
