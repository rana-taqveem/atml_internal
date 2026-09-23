"""The four post-hoc unknownness scores, computed on frozen model outputs.

Every score returns u(x) with the same orientation: larger means more novel,
so a rejection rule is always "reject when u(x) exceeds the threshold". The
raw quantities point in different directions - a confident prediction has a
large max logit but should give a small unknownness - so the signs below are
part of the definition, not a convention choice.

    u_MSP     = 1 - max_k p_k              normalized confidence
    u_MLS     = - max_k z_k                absolute logit magnitude
    u_Energy  = - log sum_k exp(z_k)       all logits, aggregated
    u_Mah     = min_c (f - mu_c)^T S^-1 (f - mu_c)    feature-space distance

The first three read the same logits, so they differ only in how they collapse
them: MSP normalizes first and therefore discards scale, MLS keeps the scale
of a single logit, and Energy keeps the scale of all of them. Mahalanobis
ignores the classifier entirely and works in the penultimate feature space.

All four must be computed from exactly the same logits and features, which is
why extract_outputs.py caches them once per model and every score reads that
cache.
"""

import numpy as np

from assignment_01.task4.config import task_config


def msp_score(logits):
    """u = 1 - max_k softmax(z)_k.

    logits   [N, 10]
    returns  [N] in [0, 1)

    Subtracting the row max before exponentiating is the standard numerically
    stable softmax: it cannot change the result, since the constant cancels.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    return 1.0 - probabilities.max(axis=1)


def mls_score(logits):
    """u = - max_k z_k (maximum logit score).

    logits   [N, 10]
    returns  [N]
    """
    return -logits.max(axis=1)


def energy_score(logits):
    """u = - log sum_k exp(z_k), the negative free energy.

    logits   [N, 10]
    returns  [N]

    Computed with the log-sum-exp shift for stability. Unlike MLS this uses
    every logit, so a sample with several moderately high logits scores
    differently from one with a single high logit.
    """
    maxima = logits.max(axis=1)
    shifted = logits - maxima[:, None]
    return -(maxima + np.log(np.exp(shifted).sum(axis=1)))


def fit_mahalanobis(train_features, train_labels, num_classes=None, epsilon=None):
    """Class means and one shared diagonal covariance from training features.

    train_features  [N, 512]   unaugmented CIFAR-10 training features
    train_labels    [N]
    returns         dict with means [C, 512] and inverse_variance [512]

    The assignment specifies a single shared *diagonal* covariance, so only
    per-dimension variances are needed. They are computed on the residuals
    after subtracting each example's own class mean, which is what makes the
    covariance shared ("within-class") rather than per-class. A small epsilon
    is added to every diagonal entry so no dimension can divide by zero.
    """
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
    """u = min_c (f - mu_c)^T S^-1 (f - mu_c), S diagonal.

    features  [N, 512]
    returns   [N]

    With a diagonal S the quadratic form is just a weighted sum of squared
    differences, so this is computed class by class rather than by building an
    [N, C, 512] tensor.
    """
    means = fitted["means"]                                  # [C, 512]
    inverse_variance = fitted["inverse_variance"]            # [512]

    distances = np.empty((features.shape[0], means.shape[0]), dtype=np.float64)
    for class_index, mean in enumerate(means):
        difference = features - mean                         # [N, 512]
        distances[:, class_index] = (difference ** 2 * inverse_variance).sum(axis=1)
    return distances.min(axis=1)


# Name -> (function, whether it needs the Mahalanobis fit). Used by
# evaluate_osr.py so the four scores are always applied in a fixed order.
POST_HOC_SCORES = {
    "msp": (msp_score, False),
    "mls": (mls_score, False),
    "energy": (energy_score, False),
    "mahalanobis": (mahalanobis_score, True),
}


def compute_scores(logits, features, fitted=None, names=None):
    """Apply the requested scores to one set of cached outputs.

    Returns {score_name: u [N]}. `fitted` is required for Mahalanobis and
    ignored by the rest.
    """
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
