"""Per-class accuracy, baseline changes, and dominant confusions."""

import numpy as np

from assignment_01.task2.config import task_config


def per_class_accuracy(labels, predictions, class_names=None):
    """Accuracy within each true class, plus its support."""
    class_names = class_names or task_config.PACS_CLASSES
    labels, predictions = np.asarray(labels), np.asarray(predictions)

    rows = {}
    for index, name in enumerate(class_names):
        mask = labels == index
        support = int(mask.sum())
        rows[name] = {
            "accuracy_pct": float(100 * (predictions[mask] == index).mean()) if support else None,
            "support": support,
        }
    return rows


def dominant_confusion(labels, predictions, class_index, class_names=None):
    """The wrong class this class is most often mistaken for."""
    class_names = class_names or task_config.PACS_CLASSES
    labels, predictions = np.asarray(labels), np.asarray(predictions)

    wrong = predictions[(labels == class_index) & (predictions != class_index)]
    if wrong.size == 0:
        return None, 0

    values, counts = np.unique(wrong, return_counts=True)
    worst = values[counts.argmax()]
    return class_names[worst], int(counts.max())


def compare_to_baseline(baseline, method, class_names=None):
    """Compare class accuracy and confusions for predictions on the same set."""
    class_names = class_names or task_config.PACS_CLASSES
    if not np.array_equal(baseline["labels"], method["labels"]):
        raise ValueError("Baseline and method must be evaluated on the same images in order.")

    base_rows = per_class_accuracy(baseline["labels"], baseline["predictions"], class_names)
    method_rows = per_class_accuracy(method["labels"], method["predictions"], class_names)

    rows = []
    for index, name in enumerate(class_names):
        confused_with, count = dominant_confusion(
            method["labels"], method["predictions"], index, class_names)
        base = base_rows[name]["accuracy_pct"]
        new = method_rows[name]["accuracy_pct"]
        rows.append({
            "class": name,
            "support": method_rows[name]["support"],
            "baseline_accuracy_pct": base,
            "method_accuracy_pct": new,
            "change_pp": None if base is None or new is None else new - base,
            "confused_with": confused_with,
            "confused_count": count,
        })
    return rows


def summarize_transfer(rows, top=2):
    """The classes that improved and degraded most: the negative-transfer check."""
    scored = [r for r in rows if r["change_pp"] is not None]
    ordered = sorted(scored, key=lambda r: r["change_pp"])
    return {
        "largest_degradations": ordered[:top],
        "largest_improvements": list(reversed(ordered[-top:])),
        "n_classes_improved": sum(1 for r in scored if r["change_pp"] > 0),
        "n_classes_degraded": sum(1 for r in scored if r["change_pp"] < 0),
        "mean_change_pp": float(np.mean([r["change_pp"] for r in scored])) if scored else None,
    }


def confusion_matrix(labels, predictions, class_names=None):
    """Counts of true class (rows) against predicted class (columns)."""
    class_names = class_names or task_config.PACS_CLASSES
    size = len(class_names)
    matrix = np.zeros((size, size), dtype=int)
    for true, predicted in zip(np.asarray(labels), np.asarray(predictions)):
        matrix[true, predicted] += 1
    return matrix
