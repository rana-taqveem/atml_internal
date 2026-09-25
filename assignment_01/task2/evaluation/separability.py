"""Measure domain separability with a probe on balanced frozen features."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from assignment_01.task2.config import task_config
from assignment_01.task2.evaluation.features import collect_features


def domain_separability(model, loaders, seed=None, test_size=0.30, regularization=1.0,
                        max_per_group=None):
    """Return held-out domain-classifier accuracy on frozen features."""
    seed = task_config.SEED if seed is None else seed

    # Balance the groups: the smallest decides, so the probe cannot win by
    # predicting the majority group.
    counts = {name: len(loader.dataset) for name, loader in loaders.items()}
    per_group = min(counts.values()) if max_per_group is None else max_per_group

    collected = collect_features(model, loaders, max_per_group=per_group, seed=seed)
    X, y = collected["features"], collected["groups"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    probe = LogisticRegression(C=regularization, class_weight="balanced",
                               max_iter=2000, random_state=seed)
    probe.fit(X_train, y_train)
    accuracy = 100 * probe.score(X_test, y_test)

    n_groups = len(collected["group_names"])
    return {
        "domain_separability_pct": float(accuracy),
        "chance_pct": 100.0 / n_groups,
        "groups": collected["group_names"],
        "n_per_group": int(per_group),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "available_per_group": counts,
        "settings": {"seed": seed, "test_size": test_size, "C": regularization,
                     "class_weight": "balanced"},
    }
