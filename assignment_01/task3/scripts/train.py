"""Train Task 3 ERM, DAN-DG, or SAM without loading Sketch."""

import argparse
import json
import os

import torch
import torch.nn as nn

from assignment_01.task2.data.pacs import DomainBalancedBatches, get_source_loaders
from assignment_01.task2.data.download import prepare_pacs
from assignment_01.task2.models.backbones import build_model
from assignment_01.task2.scripts.main import (
    DEVICE,
    evaluate,
    set_seed,
    train_model,
)
from assignment_01.task3.config import task_config


def source_validation_report(model, validation_loaders, criterion):
    """Return per-domain, mean, and worst-domain validation performance."""
    per_domain = {}
    for domain, loader in validation_loaders.items():
        loss, accuracy, macro_f1 = evaluate(model, loader, criterion)
        per_domain[domain] = {"loss": loss, "accuracy_pct": accuracy, "macro_f1_pct": macro_f1}

    accuracies = [d["accuracy_pct"] for d in per_domain.values()]
    macro_f1s = [d["macro_f1_pct"] for d in per_domain.values()]
    worst_domain = min(per_domain, key=lambda d: per_domain[d]["macro_f1_pct"])

    return {
        "per_domain": per_domain,
        "mean_accuracy_pct": sum(accuracies) / len(accuracies),
        "mean_macro_f1_pct": sum(macro_f1s) / len(macro_f1s),
        "worst_accuracy_pct": min(accuracies),
        "worst_macro_f1_pct": min(macro_f1s),
        "worst_domain": worst_domain,
    }


def run_method(method="dan_dg", num_workers=2, num_epochs=None, data_root=None,
               lambda_dg=None, rho=None):
    """Train (or load, for ERM) one Task 3 method and report source-side results."""
    task_config.init_env()
    set_seed(task_config.SEED)

    domain_root = prepare_pacs(data_root or task_config.TASK_DATASET_DIR)
    source_loaders, validation_loaders = get_source_loaders(
        domain_root=domain_root, num_workers=num_workers)

    # target_loader=None: Task 3 never draws Sketch batches.
    train_batches = DomainBalancedBatches(source_loaders, target_loader=None)

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    run_name = method
    settings = {}
    history = None

    if method == "erm":
        # Reuse the Task 2 source-only checkpoint unchanged, as required.
        if not os.path.isfile(task_config.ERM_CHECKPOINT):
            raise SystemExit(
                f"Task 2 ERM checkpoint not found: {task_config.ERM_CHECKPOINT}. "
                "Train Task 2's source-only model first."
            )
        model.load_state_dict(torch.load(task_config.ERM_CHECKPOINT, map_location=DEVICE))
        settings = {"reused_checkpoint": task_config.ERM_CHECKPOINT}
        print(f"ERM: reusing Task 2 checkpoint {task_config.ERM_CHECKPOINT} (no retraining)")

    elif method == "dan_dg":
        from assignment_01.task3.methods.dan_dg import make_dan_dg_loss

        extra_loss_fn = make_dan_dg_loss(lambda_dg=lambda_dg)
        settings = {"lambda_dg": extra_loss_fn.lambda_value,
                    "bandwidth_multipliers": list(task_config.MMD_BANDWIDTH_MULTIPLIERS),
                    "pairs": "photo-art, photo-cartoon, art-cartoon"}
        if extra_loss_fn.lambda_value != task_config.LAMBDA_DG:
            run_name = f"dan_dg_lambda{extra_loss_fn.lambda_value:g}"
        print(f"DAN-DG: lambda_DG = {extra_loss_fn.lambda_value}, aligning the three source pairs")

        optimizer = torch.optim.AdamW(model.parameters(), lr=task_config.LEARNING_RATE,
                                      weight_decay=task_config.WEIGHT_DECAY)
        history = train_model(
            model, run_name, train_batches, validation_loaders, criterion, optimizer,
            num_epochs=num_epochs or task_config.NUM_EPOCHS,
            early_stopping_patience=task_config.EARLY_STOPPING_PATIENCE,
            extra_loss_fn=extra_loss_fn, grad_clip=task_config.GRAD_CLIP_NORM,
            weights_dir=task_config.MODEL_WEIGHTS_DIR,
            results_dir=task_config.TASK_RESULTS_DIR,
        )

    elif method == "sam":
        from assignment_01.task3.methods.sam import SAM, train_one_epoch_sam

        rho = task_config.SAM_RHO if rho is None else rho
        settings = {"rho": rho, "base_optimizer": "AdamW"}
        if rho != task_config.SAM_RHO:
            run_name = f"sam_rho{rho:g}"
        print(f"SAM: rho = {rho}, two forward/backward passes per step")

        optimizer = SAM(model.parameters(), torch.optim.AdamW, rho=rho,
                        lr=task_config.LEARNING_RATE, weight_decay=task_config.WEIGHT_DECAY)

        def sam_epoch(model, loader, criterion, optimizer, extra_loss_fn=None, scaler=None,
                      epoch=0, num_epochs=1, grad_clip=None):
            return train_one_epoch_sam(model, loader, criterion, optimizer,
                                       epoch=epoch, num_epochs=num_epochs, grad_clip=grad_clip)

        history = train_model(
            model, run_name, train_batches, validation_loaders, criterion, optimizer,
            num_epochs=num_epochs or task_config.NUM_EPOCHS,
            early_stopping_patience=task_config.EARLY_STOPPING_PATIENCE,
            use_amp=False, grad_clip=task_config.GRAD_CLIP_NORM, epoch_fn=sam_epoch,
            weights_dir=task_config.MODEL_WEIGHTS_DIR,
            results_dir=task_config.TASK_RESULTS_DIR,
        )
    else:
        raise SystemExit(f"Unknown method '{method}'.")

    report = source_validation_report(model, validation_loaders, criterion)
    print(f"\nSource validation (no Sketch involved):")
    for domain, scores in report["per_domain"].items():
        print(f"   {domain:<14} accuracy {scores['accuracy_pct']:.2f}%, "
              f"macro-F1 {scores['macro_f1_pct']:.2f}%")
    print(f"   mean macro-F1  {report['mean_macro_f1_pct']:.2f}%")
    print(f"   worst domain   {report['worst_domain']} at {report['worst_macro_f1_pct']:.2f}%")

    checkpoint = (history["checkpoint"] if history
                  else task_config.ERM_CHECKPOINT)
    os.makedirs(task_config.TASK_RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(task_config.TASK_RESULTS_DIR, f"{run_name}_source_results.json")
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump({"method": run_name, "settings": settings, "checkpoint": checkpoint,
                   "history": history, "source_validation": report}, file, indent=2)
    print(f"Saved: {results_path}")

    return history, report


def main():
    parser = argparse.ArgumentParser(
        description="Task 3 training (source domains only; Sketch is never loaded)")
    parser.add_argument("--method", choices=["erm", "dan_dg", "sam"], required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--lambda-dg", type=float, default=None,
                        help="DAN-DG only; study uses 0.1, 1, 10")
    parser.add_argument("--rho", type=float, default=None,
                        help="SAM only; study uses 0.01, 0.05, 0.1")
    args = parser.parse_args()

    run_method(args.method, num_workers=args.num_workers, num_epochs=args.epochs,
               data_root=args.data_root, lambda_dg=args.lambda_dg, rho=args.rho)


if __name__ == "__main__":
    main()
