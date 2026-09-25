"""AUROC and validation-calibrated open-set recognition metrics."""

import numpy as np


def auroc(known_scores, unknown_scores):
    """Compute AUROC with the Mann-Whitney rank identity."""
    known_scores = np.asarray(known_scores, dtype=np.float64)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64)
    n_known, n_unknown = len(known_scores), len(unknown_scores)
    if n_known == 0 or n_unknown == 0:
        return float("nan")

    combined = np.concatenate([known_scores, unknown_scores])
    order = combined.argsort()
    ranks = np.empty(len(combined), dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1)

    # Average the ranks inside each group of equal values.
    unique, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(unique))
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]

    unknown_rank_sum = ranks[n_known:].sum()
    u_statistic = unknown_rank_sum - n_unknown * (n_unknown + 1) / 2.0
    return float(u_statistic / (n_known * n_unknown))


def calibrate_threshold(validation_scores, percentile=95.0):
    """Calibrate a known-only validation-score percentile threshold."""
    return float(np.percentile(np.asarray(validation_scores, dtype=np.float64), percentile))


def acceptance_rate(scores, threshold):
    """Fraction accepted, i.e. u(x) <= tau."""
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0:
        return float("nan")
    return float((scores <= threshold).mean())


def rejection_rate(scores, threshold):
    """Fraction rejected, i.e. u(x) > tau. Equals 1 - FPR@95TPR for unknowns."""
    accepted = acceptance_rate(scores, threshold)
    return float("nan") if np.isnan(accepted) else 1.0 - accepted


def closed_set_accuracy(logits, labels):
    """Compute accuracy on known examples before rejection."""
    predictions = np.asarray(logits).argmax(axis=1)
    return float((predictions == np.asarray(labels)).mean() * 100.0)


def evaluate_score(known_test_scores, near_scores, far_scores, validation_scores,
                   percentile=95.0):
    """Evaluate one model-score pair at all required comparisons."""
    all_unknown = np.concatenate([np.asarray(near_scores), np.asarray(far_scores)])
    threshold = calibrate_threshold(validation_scores, percentile)

    return {
        "auroc_near": auroc(known_test_scores, near_scores),
        "auroc_far": auroc(known_test_scores, far_scores),
        "auroc_all": auroc(known_test_scores, all_unknown),
        "threshold": threshold,
        "threshold_percentile": percentile,
        "known_test_acceptance_rate": acceptance_rate(known_test_scores, threshold),
        "near_rejection_rate": rejection_rate(near_scores, threshold),
        "far_rejection_rate": rejection_rate(far_scores, threshold),
        "all_rejection_rate": rejection_rate(all_unknown, threshold),
        "fpr_at_95tpr_near": acceptance_rate(near_scores, threshold),
        "fpr_at_95tpr_far": acceptance_rate(far_scores, threshold),
        "fpr_at_95tpr_all": acceptance_rate(all_unknown, threshold),
    }
