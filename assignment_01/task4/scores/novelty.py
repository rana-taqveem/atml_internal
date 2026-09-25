"""Post-hoc novelty scores; larger values indicate greater novelty."""

import numpy as np

from assignment_01.task4.config import task_config


def msp_score(logits):
    """Return 1 - maximum softmax probability for each sample."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    return 1.0 - probabilities.max(axis=1)


def mls_score(logits):
    """Return negative maximum logit for each sample."""
    return -logits.max(axis=1)


def energy_score(logits):
    """Return numerically stable negative log-sum-exp energy."""
    maxima = logits.max(axis=1)
    shifted = logits - maxima[:, None]
    return -(maxima + np.log(np.exp(shifted).sum(axis=1)))


def fit_mahalanobis(train_features, train_labels, num_classes=None, epsilon=None):
    """Fit class means and shared diagonal within-class covariance."""
    num_classes = num_classes or task_config.NUM_CLASSES
    epsilon = task_config.MAHALANOBIS_EPSILON if epsilon is None else epsilon

    means = np.zeros((num_classes, train_features.shape[1]), dtype=np.float64)
    for class_index in range(num_classes):
        rows = train_features[train_labels == class_index]
        if len(rows) == 0:
            raise ValueError(f"No training features for class {class_index}.")
        means[class_index] = rows.mean(axis=0)

    residuals = train_features - means[train_labels]        # [N, 512]
    variances = residuals.var(axis=0) + epsilon             # [512]
    return {"means": means, "inverse_variance": 1.0 / variances, "epsilon": epsilon}


def mahalanobis_score(features, fitted):
    """Return minimum class-conditional diagonal Mahalanobis distance."""
    means = fitted["means"]                                  # [C, 512]
    inverse_variance = fitted["inverse_variance"]            # [512]

    distances = np.empty((features.shape[0], means.shape[0]), dtype=np.float64)
    for class_index, mean in enumerate(means):
        difference = features - mean                         # [N, 512]
        distances[:, class_index] = (difference ** 2 * inverse_variance).sum(axis=1)
    return distances.min(axis=1)


# Name -> (score function, requires Mahalanobis fit).
POST_HOC_SCORES = {
    "msp": (msp_score, False),
    "mls": (mls_score, False),
    "energy": (energy_score, False),
    "mahalanobis": (mahalanobis_score, True),
}


def compute_scores(logits, features, fitted=None, names=None):
    """Apply requested scores to cached logits and features."""
    names = names or list(POST_HOC_SCORES)
    results = {}
    for name in names:
        function, needs_fit = POST_HOC_SCORES[name]
        if needs_fit:
            if fitted is None:
                raise ValueError(f"{name} needs the training-set fit; call fit_mahalanobis.")
            results[name] = function(features, fitted)
        else:
            results[name] = function(logits)
    return results
