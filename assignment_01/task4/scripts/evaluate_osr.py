"""Task 4 evaluation: both required tables, the figure data and the failures.

Reads the cached outputs written by extract_outputs.py, so every score is
computed from identical model outputs.

Produces:

  Table 1  MSP / MLS / Energy / Mahalanobis on the frozen Vanilla model,
           with near, far and all-unknown AUROC and the validation-calibrated
           rejection numbers.
  Table 2  Vanilla / GCSC / PROSER compared with MLS as the common score,
           plus a PROSER row using its placeholder-based detection score, and
           CSA for each.
  Failures at least three near and three far unknowns that were wrongly
           accepted, with unknown class, predicted CIFAR-10 class, score and
           threshold.
  Figure   per-score arrays for the score-distribution / ROC panel.

    python -m assignment_01.task4.scripts.evaluate_osr
"""

import argparse
import csv
import json
import os

import numpy as np

from assignment_01.task4.config import task_config
from assignment_01.task4.evaluation.metrics import closed_set_accuracy, evaluate_score
from assignment_01.task4.scores.novelty import compute_scores, fit_mahalanobis
from assignment_01.task4.scripts.extract_outputs import cache_path

TABLE1_SCORES = ["msp", "mls", "energy", "mahalanobis"]
FIGURE_SCORES = ["msp", "mls", "mahalanobis"]     # the three the assignment names


def load_cache(method):
    path = cache_path(method)
    if not os.path.isfile(path):
        raise SystemExit(f"No cached outputs for '{method}' at {path}. "
                         "Run extract_outputs.py first.")
    return np.load(path, allow_pickle=True)


def known_columns(logits, method):
    """PROSER's logits carry dummy units; every comparable score uses the ten."""
    return logits[:, :task_config.NUM_CLASSES] if method == "proser" else logits


def scores_for(cache, method, names):
    """Compute the requested scores on all four splits of one model."""
    fitted = fit_mahalanobis(cache["train_features"], cache["train_labels"])

    result = {}
    for split in ("val", "test", "near", "far"):
        logits = known_columns(cache[f"{split}_logits"], method)
        features = cache[f"{split}_features"]
        result[split] = compute_scores(logits, features, fitted=fitted, names=names)
    return result


def placeholder_scores(cache):
    """PROSER's own detection score: strongest dummy minus strongest known."""
    from assignment_01.task4.methods.proser import placeholder_detection_score

    num_known = task_config.NUM_CLASSES
    return {split: placeholder_detection_score(cache[f"{split}_logits"], num_known)
            for split in ("val", "test", "near", "far")}


def build_table1(cache):
    """Four post-hoc scores on the frozen Vanilla model."""
    by_score = scores_for(cache, "vanilla", TABLE1_SCORES)
    rows = []
    for name in TABLE1_SCORES:
        metrics = evaluate_score(
            by_score["test"][name], by_score["near"][name], by_score["far"][name],
            by_score["val"][name], percentile=task_config.ACCEPT_PERCENTILE)
        rows.append({"model": "vanilla", "score": name, **metrics})
    return rows, by_score


def build_table2(caches):
    """Vanilla / GCSC / PROSER on MLS, plus PROSER's placeholder score."""
    rows = []
    for method, cache in caches.items():
        csa = closed_set_accuracy(
            known_columns(cache["test_logits"], method), cache["test_labels"])

        by_score = scores_for(cache, method, ["mls"])
        metrics = evaluate_score(
            by_score["test"]["mls"], by_score["near"]["mls"], by_score["far"]["mls"],
            by_score["val"]["mls"], percentile=task_config.ACCEPT_PERCENTILE)
        rows.append({"model": method, "score": "mls", "csa_pct": csa, **metrics})

        if method == "proser":
            placeholder = placeholder_scores(cache)
            metrics = evaluate_score(
                placeholder["test"], placeholder["near"], placeholder["far"],
                placeholder["val"], percentile=task_config.ACCEPT_PERCENTILE)
            rows.append({"model": "proser", "score": "placeholder",
                         "csa_pct": csa, **metrics})
    return rows


def find_failures(cache, method, score_name="mls", per_group=5):
    """Unknowns wrongly accepted under the calibrated threshold.

    The assignment asks for at least three near and three far examples, with
    the unknown class, the predicted CIFAR-10 class, the score and the
    threshold, so that plausible confusions can be told apart from surprising
    ones. The most confidently accepted are reported first - those are the
    informative failures.
    """
    by_score = scores_for(cache, method, [score_name])
    from assignment_01.task4.evaluation.metrics import calibrate_threshold
    threshold = calibrate_threshold(by_score["val"][score_name],
                                    task_config.ACCEPT_PERCENTILE)

    rows = []
    for group in ("near", "far"):
        unknown_scores = by_score[group][score_name]
        logits = known_columns(cache[f"{group}_logits"], method)
        predictions = logits.argmax(axis=1)
        names = cache[f"{group}_names"]

        accepted = np.where(unknown_scores <= threshold)[0]
        # Lowest unknownness first: the most confidently mistaken for known.
        for index in accepted[np.argsort(unknown_scores[accepted])][:per_group]:
            rows.append({
                "group": group,
                "unknown_class": str(names[index]),
                "predicted_cifar10_class": task_config.CIFAR10_CLASSES[predictions[index]],
                "score_name": score_name,
                "score": float(unknown_scores[index]),
                "threshold": float(threshold),
            })
    return rows


def run(methods=None, results_dir=None):
    task_config.init_env()
    results_dir = results_dir or task_config.TASK_RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    methods = methods or ["vanilla", "gcsc", "proser"]
    caches = {}
    for method in methods:
        try:
            caches[method] = load_cache(method)
        except SystemExit as error:
            print(f"  {error}")

    if "vanilla" not in caches:
        raise SystemExit("Vanilla outputs are required for Table 1.")

    print("\n=== Table 1: post-hoc scores on the frozen Vanilla model")
    table1, vanilla_scores = build_table1(caches["vanilla"])
    _print_table(table1, ("score",))

    print("\n=== Table 2: Vanilla / GCSC / PROSER (MLS common score)")
    table2 = build_table2(caches)
    _print_table(table2, ("model", "score"), extra=("csa_pct",))

    failures = []
    for method, cache in caches.items():
        for row in find_failures(cache, method):
            failures.append({"model": method, **row})

    # Arrays behind the score-distribution / ROC figure.
    figure = {name: {split: vanilla_scores[split][name].tolist()
                     for split in ("test", "near", "far")}
              for name in FIGURE_SCORES if name in vanilla_scores["test"]}

    _write_csv(os.path.join(results_dir, "table1_posthoc_scores.csv"), table1)
    _write_csv(os.path.join(results_dir, "table2_model_comparison.csv"), table2)
    _write_csv(os.path.join(results_dir, "failure_analysis.csv"), failures)
    with open(os.path.join(results_dir, "figure_scores.json"), "w", encoding="utf-8") as file:
        json.dump(figure, file)
    with open(os.path.join(results_dir, "osr_results.json"), "w", encoding="utf-8") as file:
        json.dump({"table1": table1, "table2": table2, "failures": failures,
                   "accept_percentile": task_config.ACCEPT_PERCENTILE}, file, indent=2)

    print(f"\nSaved tables, failure analysis and figure data to {results_dir}")
    return {"table1": table1, "table2": table2, "failures": failures}


def _print_table(rows, key_fields, extra=()):
    if not rows:
        return
    header = "  ".join(f"{field:<12}" for field in key_fields)
    header += "".join(f"{field:>10}" for field in extra)
    header += f"{'AUROC near':>12}{'AUROC far':>11}{'AUROC all':>11}"
    header += f"{'known acc':>11}{'rej near':>10}{'rej far':>9}"
    print(header)
    for row in rows:
        line = "  ".join(f"{str(row[field]):<12}" for field in key_fields)
        line += "".join(f"{row[field]:>10.2f}" for field in extra)
        line += (f"{row['auroc_near']:>12.4f}{row['auroc_far']:>11.4f}"
                 f"{row['auroc_all']:>11.4f}")
        line += (f"{row['known_test_acceptance_rate'] * 100:>10.2f}%"
                 f"{row['near_rejection_rate'] * 100:>9.2f}%"
                 f"{row['far_rejection_rate'] * 100:>8.2f}%")
        print(line)


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()
    run(methods=args.methods, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
