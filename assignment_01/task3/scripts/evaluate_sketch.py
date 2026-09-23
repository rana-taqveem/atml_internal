"""Task 3 final evaluation: the only script that loads Sketch.

Run this after every Task 3 configuration is frozen. Training, checkpoint
selection and the source-side diagnostics happen in train.py, which never
touches the target domain; keeping the Sketch load in a separate file is how
that separation is enforced rather than merely intended.

For ERM, DAN-DG, SAM and the configured DAN-DG strength-study checkpoints,
this reports:

  * accuracy and macro-F1 on each source validation domain, with mean and
    worst-domain values (recomputed here so one table holds everything)
  * accuracy and macro-F1 on Sketch, and the change relative to ERM
  * source-domain separability: a three-class probe over Photo, Art Painting
    and Cartoon features, so chance is 33.3%, not 50%
  * the shared local sharpness proxy, measured on one fixed validation batch
  * per-class Sketch accuracy, the change against ERM, and dominant confusions

It writes three report-ready files:

  * task3_final_results.json - complete nested diagnostics
  * task3_method_comparison.csv - one compact aggregate row per method
  * task3_per_class_sketch.csv - class-level changes against ERM

    python -m assignment_01.task3.scripts.evaluate_sketch
"""

import argparse
import csv
import json
import os
from pathlib import Path

import torch
import torch.nn as nn

from assignment_01.task2.data.download import prepare_pacs
from assignment_01.task2.data.pacs import get_source_loaders, get_target_loaders
from assignment_01.task2.evaluation import class_analysis as CA
from assignment_01.task2.evaluation.features import collect_features
from assignment_01.task2.evaluation.separability import domain_separability
from assignment_01.task2.evaluation.sharpness import fixed_validation_batch, sharpness_proxy
from assignment_01.task2.models.backbones import build_model
from assignment_01.task2.scripts.main import DEVICE, evaluate, set_seed
from assignment_01.task3.config import task_config

DEFAULT_METHODS = [
    "erm",
    "dan_dg",
    "sam",
    "dan_dg_lambda0.1",
    "dan_dg_lambda10",
]


def method_settings(method):
    """Return the controlled setting represented by a checkpoint name."""
    if method == "dan_dg":
        return {"lambda_dg": task_config.LAMBDA_DG}
    if method.startswith("dan_dg_lambda"):
        return {"lambda_dg": float(method.removeprefix("dan_dg_lambda"))}
    if method == "sam":
        return {"rho": task_config.SAM_RHO}
    return {}


def checkpoint_for(method):
    """ERM reuses the Task 2 checkpoint; the others come from Task 3 training."""
    if method == "erm":
        return task_config.ERM_CHECKPOINT
    return os.path.join(task_config.MODEL_WEIGHTS_DIR, f"{method}_best.pth")


def run(methods=None, num_workers=2, data_root=None, results_dir=None, strict=False,
        download_hf=False):
    task_config.init_env()
    set_seed(task_config.SEED)

    results_dir = Path(results_dir or task_config.TASK_RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    # download_hf lets a machine without the prepared archive (Kaggle, a fresh
    # VM) fetch PACS from the hub instead of failing.
    domain_root = prepare_pacs(data_root or task_config.TASK_DATASET_DIR,
                               allow_huggingface=download_hf)
    _, validation_loaders = get_source_loaders(domain_root=domain_root, num_workers=num_workers)
    _, target_loader = get_target_loaders(domain_root=domain_root, num_workers=num_workers)

    sharpness_images, sharpness_labels = fixed_validation_batch(
        validation_loaders, per_domain=task_config.SHARPNESS_BATCH_PER_DOMAIN,
        seed=task_config.SEED, num_workers=0)
    print(f"Sharpness batch: {len(sharpness_labels)} examples "
          f"({task_config.SHARPNESS_BATCH_PER_DOMAIN} per source domain, seed {task_config.SEED})")

    criterion = nn.CrossEntropyLoss()
    report = {}
    target_predictions = {}

    selected_methods = methods or DEFAULT_METHODS
    if strict:
        missing = [
            (method, checkpoint_for(method))
            for method in selected_methods
            if not os.path.isfile(checkpoint_for(method))
        ]
        if missing:
            details = "\n".join(f"  {method}: {path}" for method, path in missing)
            raise FileNotFoundError(f"Required Task 3 checkpoints are missing:\n{details}")

    for method in selected_methods:
        path = checkpoint_for(method)
        if not os.path.isfile(path):
            print(f"  missing {method}: no checkpoint at {path}")
            continue

        print(f"\n=== {method}")
        model = build_model()
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()

        source = {}
        for domain, loader in validation_loaders.items():
            loss, accuracy, macro_f1 = evaluate(model, loader, criterion)
            source[domain] = {"accuracy_pct": accuracy, "macro_f1_pct": macro_f1}
        accuracies = [d["accuracy_pct"] for d in source.values()]
        macro_f1s = [d["macro_f1_pct"] for d in source.values()]

        # Three-class probe over the observed source domains: chance is 33.3%.
        probe = domain_separability(model, validation_loaders, seed=task_config.SEED,
                                    test_size=task_config.SEPARABILITY_TEST_SIZE,
                                    regularization=task_config.SEPARABILITY_C)

        sharp = sharpness_proxy(model, sharpness_images, sharpness_labels,
                                radius=task_config.SHARPNESS_RADIUS, criterion=criterion)

        _, target_accuracy, target_macro_f1 = evaluate(model, target_loader, criterion)
        collected = collect_features(model, {"sketch": target_loader})
        target_predictions[method] = {
            "labels": collected["labels"], "predictions": collected["predictions"]}

        report[method] = {
            "checkpoint": path,
            "settings": method_settings(method),
            "source_validation": source,
            "mean_source_accuracy_pct": sum(accuracies) / len(accuracies),
            "mean_source_macro_f1_pct": sum(macro_f1s) / len(macro_f1s),
            "worst_source_macro_f1_pct": min(macro_f1s),
            "worst_source_domain": min(source, key=lambda d: source[d]["macro_f1_pct"]),
            "target_accuracy_pct": target_accuracy,
            "target_macro_f1_pct": target_macro_f1,
            "source_domain_separability_pct": probe["domain_separability_pct"],
            "separability_chance_pct": probe["chance_pct"],
            "sharpness": sharp,
        }
        print(f"   mean source macro-F1 {report[method]['mean_source_macro_f1_pct']:.2f}%, "
              f"worst {report[method]['worst_source_macro_f1_pct']:.2f}% "
              f"({report[method]['worst_source_domain']})")
        print(f"   Sketch accuracy {target_accuracy:.2f}%, macro-F1 {target_macro_f1:.2f}%")
        print(f"   source separability {probe['domain_separability_pct']:.2f}% "
              f"(chance {probe['chance_pct']:.1f}%)")
        print(f"   sharpness proxy {sharp['delta_sharp']:+.4f} "
              f"(loss {sharp['loss']:.4f} -> {sharp['perturbed_loss']:.4f})")

    if "erm" not in report:
        raise RuntimeError(
            "The ERM checkpoint is required to calculate aggregate and per-class changes."
        )

    # Changes against ERM, including the per-class view.
    per_class_rows = []
    if "erm" in report:
        baseline_accuracy = report["erm"]["target_accuracy_pct"]
        baseline_f1 = report["erm"]["target_macro_f1_pct"]
        for method, entry in report.items():
            entry["target_accuracy_change_pp"] = entry["target_accuracy_pct"] - baseline_accuracy
            entry["target_macro_f1_change_pp"] = entry["target_macro_f1_pct"] - baseline_f1
            rows = CA.compare_to_baseline(target_predictions["erm"], target_predictions[method])
            summary = CA.summarize_transfer(rows)
            entry["per_class_summary"] = {k: v for k, v in summary.items()
                                          if not isinstance(v, list)}
            entry["largest_degradations"] = summary["largest_degradations"]
            entry["largest_improvements"] = summary["largest_improvements"]
            for row in rows:
                per_class_rows.append({"method": method, **row})

    with (results_dir / "task3_final_results.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"\nSaved {results_dir / 'task3_final_results.json'}")

    comparison_rows = []
    for method, entry in report.items():
        source = entry["source_validation"]
        settings = entry["settings"]
        comparison_rows.append({
            "method": method,
            "lambda_dg": settings.get("lambda_dg", ""),
            "rho": settings.get("rho", ""),
            "photo_accuracy_pct": source["photo"]["accuracy_pct"],
            "photo_macro_f1_pct": source["photo"]["macro_f1_pct"],
            "art_painting_accuracy_pct": source["art_painting"]["accuracy_pct"],
            "art_painting_macro_f1_pct": source["art_painting"]["macro_f1_pct"],
            "cartoon_accuracy_pct": source["cartoon"]["accuracy_pct"],
            "cartoon_macro_f1_pct": source["cartoon"]["macro_f1_pct"],
            "mean_source_accuracy_pct": entry["mean_source_accuracy_pct"],
            "mean_source_macro_f1_pct": entry["mean_source_macro_f1_pct"],
            "worst_source_macro_f1_pct": entry["worst_source_macro_f1_pct"],
            "worst_source_domain": entry["worst_source_domain"],
            "target_accuracy_pct": entry["target_accuracy_pct"],
            "target_macro_f1_pct": entry["target_macro_f1_pct"],
            "target_accuracy_change_pp": entry["target_accuracy_change_pp"],
            "target_macro_f1_change_pp": entry["target_macro_f1_change_pp"],
            "source_domain_separability_pct": entry["source_domain_separability_pct"],
            "separability_chance_pct": entry["separability_chance_pct"],
            "sharpness_loss": entry["sharpness"]["loss"],
            "sharpness_perturbed_loss": entry["sharpness"]["perturbed_loss"],
            "sharpness_delta": entry["sharpness"]["delta_sharp"],
            "sharpness_radius": entry["sharpness"]["radius"],
        })

    comparison_path = results_dir / "task3_method_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"Saved {comparison_path}")

    if per_class_rows:
        path = results_dir / "task3_per_class_sketch.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(per_class_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_class_rows)
        print(f"Saved {path}")

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail instead of skipping a requested checkpoint that is missing",
    )
    parser.add_argument(
        "--download-hf",
        action="store_true",
        help="fetch PACS from the Hugging Face hub when no prepared archive is present",
    )
    args = parser.parse_args()
    run(methods=args.methods, num_workers=args.num_workers,
        data_root=args.data_root, results_dir=args.results_dir, strict=args.strict,
        download_hf=args.download_hf)


if __name__ == "__main__":
    main()
