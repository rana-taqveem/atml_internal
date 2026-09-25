"""Compute Task 2 post-training separability and per-class diagnostics."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from assignment_01.task2.config import task_config
from assignment_01.task2.data.download import prepare_pacs
from assignment_01.task2.data.pacs import get_source_loaders, get_target_loaders
from assignment_01.task2.evaluation import class_analysis as CA
from assignment_01.task2.evaluation.features import collect_features
from assignment_01.task2.evaluation.separability import domain_separability
from assignment_01.task2.models.backbones import build_model
from assignment_01.task2.scripts.main import DEVICE, evaluate, set_seed

DEFAULT_METHODS = ["erm", "dan", "dann", "cdan", "dann_alpha0.25", "dann_alpha0.5"]


def available_checkpoints(methods=None):
    """Map method name -> checkpoint path, skipping anything not trained yet."""
    found = {}
    for method in methods or DEFAULT_METHODS:
        path = os.path.join(task_config.MODEL_WEIGHTS_DIR, f"{method}_best.pth")
        if os.path.isfile(path):
            found[method] = path
        else:
            print(f"  skipping {method}: no checkpoint at {path}")
    return found


def run(methods=None, num_workers=2, data_root=None, results_dir=None):
    task_config.init_env()
    set_seed(task_config.SEED)

    results_dir = Path(results_dir or task_config.TASK_RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    domain_root = prepare_pacs(data_root or task_config.TASK_DATASET_DIR)
    _, validation_loaders = get_source_loaders(domain_root=domain_root, num_workers=num_workers)
    _, target_loader = get_target_loaders(domain_root=domain_root, num_workers=num_workers)

    checkpoints = available_checkpoints(methods)
    if not checkpoints:
        raise SystemExit("No checkpoints found; train the methods first.")

    criterion = nn.CrossEntropyLoss()
    diagnostics = {}
    target_predictions = {}

    for method, path in checkpoints.items():
        print(f"\n=== {method}")
        model = build_model()
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()

        # Domain separability: source validation pooled against the target.
        # One entry per group, so the probe is a 2-class problem (chance 50%).
        pooled_source = {"source_validation": _PooledLoader(validation_loaders)}
        probe = domain_separability(
            model, {**pooled_source, "target": target_loader},
            seed=task_config.SEED, test_size=task_config.SEPARABILITY_TEST_SIZE
            if hasattr(task_config, "SEPARABILITY_TEST_SIZE") else 0.30)
        print(f"   domain separability {probe['domain_separability_pct']:.2f}% "
              f"(chance {probe['chance_pct']:.1f}%, {probe['n_per_group']} per group)")

        # Target predictions, for per-class analysis against ERM.
        collected = collect_features(model, {"target": target_loader})
        target_predictions[method] = {
            "labels": collected["labels"], "predictions": collected["predictions"]}

        loss, accuracy, macro_f1 = evaluate(model, target_loader, criterion)
        print(f"   target accuracy {accuracy:.2f}%, macro-F1 {macro_f1:.2f}%")

        diagnostics[method] = {
            "checkpoint": path,
            "domain_separability_pct": probe["domain_separability_pct"],
            "separability_chance_pct": probe["chance_pct"],
            "separability_settings": probe["settings"],
            "separability_n_per_group": probe["n_per_group"],
            "target_accuracy_pct": accuracy,
            "target_macro_f1_pct": macro_f1,
        }

    # Per-class target accuracy and the change against ERM.
    per_class_rows = []
    if "erm" in target_predictions:
        for method, predictions in target_predictions.items():
            rows = CA.compare_to_baseline(target_predictions["erm"], predictions)
            summary = CA.summarize_transfer(rows)
            diagnostics[method]["per_class_summary"] = {
                key: value for key, value in summary.items()
                if not isinstance(value, list)
            }
            diagnostics[method]["largest_degradations"] = summary["largest_degradations"]
            diagnostics[method]["largest_improvements"] = summary["largest_improvements"]
            for row in rows:
                per_class_rows.append({"method": method, **row})
    else:
        print("\n  no ERM checkpoint: per-class comparison skipped")

    with (results_dir / "diagnostics.json").open("w", encoding="utf-8") as file:
        json.dump(diagnostics, file, indent=2)
    print(f"\nSaved {results_dir / 'diagnostics.json'}")

    if per_class_rows:
        path = results_dir / "per_class_target.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(per_class_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_class_rows)
        print(f"Saved {path}")

    return diagnostics


class _PooledLoader:
    """Iterate several loaders as one, so the probe sees pooled source validation."""

    def __init__(self, loaders):
        self.loaders = list(loaders.values())
        self.dataset = _PooledDataset([loader.dataset for loader in self.loaders])

    def __iter__(self):
        for loader in self.loaders:
            yield from loader


class _PooledDataset:
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return sum(len(d) for d in self.datasets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()
    run(methods=args.methods, num_workers=args.num_workers,
        data_root=args.data_root, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
