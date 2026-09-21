"""Domain separability: how much domain information survives in the features.

Freeze the backbone, collect balanced features, then train a logistic
regression to predict which domain each feature came from. Its held-out
accuracy is the separability score.

    Task 2  source-validation vs target        2 classes, chance = 50%
    Task 3  photo vs art_painting vs cartoon   3 classes, chance = 33.3%

Both use the same recipe from the assignment: equal numbers per group, seed
6304, a 70/30 split, and a balanced logistic-regression classifier with C = 1.

Reading the score: a lower value means domain information is harder to
recover. It does NOT mean class information was preserved, and it does not by
itself mean target recognition improved. Always report it next to accuracy.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from assignment_01.task2.config import task_config
from assignment_01.task2.evaluation.features import collect_features


def domain_separability(model, loaders, seed=None, test_size=0.30, regularization=1.0,
                        max_per_group=None):
    """Held-out accuracy of a domain classifier trained on frozen features.

    loaders   dict of domain name -> DataLoader. Two entries gives the Task 2
              source-vs-target probe; three gives the Task 3 source probe.
    """
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
