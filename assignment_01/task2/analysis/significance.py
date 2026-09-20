"""Uncertainty for the Task 1 evidence: paired tests and confidence intervals.

Every condition is evaluated on the same images, so comparisons are paired:
an image that is correct in both conditions carries no information about the
change. McNemar's test uses only the disagreements. The same logic applies to
two methods scored on one condition, and to each cue conflict compared with
its own clean content image.

Intervals accompany the tests. Proportions use Wilson intervals; ratios such
as shape bias and means such as cosine stability use a seeded bootstrap.

These functions take plain arrays so they can be checked on known inputs.
The report layer supplies the loaded results.
"""

import numpy as np
from scipy import stats

Z_95 = 1.959963984540054


def mcnemar(b, c):
    """Paired test on discordant counts.

    b = correct before and wrong after, c = wrong before and right after.
    Exact binomial while the discordant total is small, otherwise the
    chi-square approximation with continuity correction.
    """
    b, c = int(b), int(c)
    n_discordant = b + c

    if n_discordant == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0, "test": "none (no disagreements)"}

    if n_discordant < 25:
        p = float(stats.binomtest(b, n_discordant, 0.5).pvalue)
        test = "McNemar exact"
    else:
        statistic = (abs(b - c) - 1) ** 2 / n_discordant
        p = float(stats.chi2.sf(statistic, df=1))
        test = "McNemar chi-square (continuity corrected)"

    return {"b": b, "c": c, "n_discordant": n_discordant, "p_value": p, "test": test}


def paired_proportion_diff_ci(b, c, n, z=Z_95):
    """Approximate CI for the paired change in a proportion, in points.

    The change is (c - b) / n: positive means the second condition is better.
    """
    b, c, n = int(b), int(c), int(n)
    if n == 0:
        return None, None, None

    difference = (c - b) / n
    variance = (b + c) - (c - b) ** 2 / n
    error = z * np.sqrt(max(variance, 0.0)) / n
    return 100 * difference, 100 * (difference - error), 100 * (difference + error)


def wilson_interval(successes, total, z=Z_95):
    """Wilson score interval for a proportion, returned as percentages."""
    successes, total = int(successes), int(total)
    if total == 0:
        return None, None

    proportion = successes / total
    denominator = 1 + z ** 2 / total
    centre = (proportion + z ** 2 / (2 * total)) / denominator
    margin = z * np.sqrt(proportion * (1 - proportion) / total + z ** 2 / (4 * total ** 2)) / denominator
    return 100 * (centre - margin), 100 * (centre + margin)


def bootstrap_ci(values, statistic=np.mean, n_boot=10000, seed=6304, percentiles=(2.5, 97.5)):
    """Percentile bootstrap over samples (resamples rows, not conditions)."""
    values = np.asarray(values)
    if values.size == 0:
        return None, None

    rng = np.random.default_rng(seed)
    index = rng.integers(0, values.shape[0], size=(n_boot, values.shape[0]))
    draws = statistic(values[index], axis=1) if statistic is np.mean else \
        np.array([statistic(values[i]) for i in index])
    low, high = np.percentile(draws, percentiles)
    return float(low), float(high)


def bootstrap_shape_bias_ci(shape_flags, texture_flags, n_boot=10000, seed=6304):
    """CI for 100 * shape / (shape + texture), resampling conflicts.

    Draws where no conflict is decided by shape or texture leave the ratio
    undefined and are skipped, which is reported back as n_valid_draws.
    """
    shape_flags = np.asarray(shape_flags, dtype=float)
    texture_flags = np.asarray(texture_flags, dtype=float)

    rng = np.random.default_rng(seed)
    index = rng.integers(0, shape_flags.size, size=(n_boot, shape_flags.size))
    shape = shape_flags[index].sum(axis=1)
    covered = shape + texture_flags[index].sum(axis=1)

    valid = covered > 0
    if not valid.any():
        return None, None, 0

    ratios = 100 * shape[valid] / covered[valid]
    low, high = np.percentile(ratios, (2.5, 97.5))
    return float(low), float(high), int(valid.sum())


def bootstrap_paired_diff_ci(values_a, values_b, n_boot=10000, seed=6304):
    """CI for the mean paired difference a - b (same samples in both)."""
    difference = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
    return bootstrap_ci(difference)


def wilcoxon_signed_rank(differences):
    """Paired signed-rank test; ignores exact zeros."""
    differences = np.asarray(differences, dtype=float)
    nonzero = differences[differences != 0]

    if nonzero.size == 0:
        return {"n_nonzero": 0, "p_value": 1.0, "test": "none (no changes)"}

    result = stats.wilcoxon(nonzero)
    return {"n_nonzero": int(nonzero.size), "p_value": float(result.pvalue), "test": "Wilcoxon signed-rank"}


def holm_adjust(p_values):
    """Holm-Bonferroni adjusted p-values, in the order the inputs were given.

    Controls the chance of any false positive within one family of tests.
    """
    p_values = list(p_values)
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])

    adjusted = [None] * n
    running = 0.0
    for rank, i in enumerate(order):
        value = (n - rank) * p_values[i]
        running = max(running, min(value, 1.0))
        adjusted[i] = running
    return adjusted


def apply_holm(rows, family_key="family", p_key="p_value"):
    """Add Holm-adjusted p-values and a 0.05 flag within each family."""
    families = {}
    for row in rows:
        families.setdefault(row[family_key], []).append(row)

    for family_rows in families.values():
        adjusted = holm_adjust([row[p_key] for row in family_rows])
        for row, value in zip(family_rows, adjusted):
            row["p_holm"] = value
            row["significant_holm_0.05"] = bool(value < 0.05)
    return rows
