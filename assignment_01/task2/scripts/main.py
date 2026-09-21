"""Task 2 training and evaluation scaffolding (PACS unsupervised domain adaptation).

Reused from the Task 1 pipeline: the epoch loop, the evaluation loop, early
stopping, seeding, checkpoint hashing and the run manifest. Adapted for Task 2:

  * the whole network is fine-tuned, not just a linear head on frozen features;
  * BatchNorm running statistics stay frozen at their pretrained ImageNet values
    while the affine parameters remain trainable (assignment requirement);
  * checkpoints are selected on mean macro-F1 across the three source-domain
    validation splits, not on accuracy.

Method-specific objectives (DAN's MMD penalty, DANN's gradient reversal) plug in
through `extra_loss_fn`; the domain-balanced loaders they need are not built here.
"""

import argparse
import hashlib
import json
import os
import platform
import time
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from assignment_01.task2.config import task_config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {DEVICE}")


def set_seed(seed=task_config.SEED):
    """Seed Python, NumPy and torch so a run can be reproduced."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_batchnorm_statistics(model):
    """Keep BatchNorm running mean/variance fixed while training.

    Source and target images come from different visual distributions, so
    updating the running statistics on adaptation batches would make them
    depend on the source-target mixture and act as an implicit extra form of
    adaptation. Call this after model.train() on every epoch: it places only
    the BatchNorm modules in evaluation mode. The scale and bias parameters
    stay trainable.
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def train_one_epoch(model, loader, criterion, optimizer, extra_loss_fn=None, scaler=None,
                    epoch=0, num_epochs=1):
    """One pass over a labelled loader.

    extra_loss_fn(model, batch, source_features, progress) may return an extra
    scalar loss (an MMD penalty, a domain-adversarial loss) added to the
    classification loss before the backward pass. It receives the source
    features already computed here, so the backbone runs once per batch, and
    `progress`, the fraction of training completed, which the DANN schedule
    needs.

    scaler enables mixed precision on CUDA, which roughly halves the time per
    epoch for this model with no change to the objective.

    Returns a dict with the mean total, classification and alignment losses
    and the training accuracy, so the two loss curves can be reported apart.
    """
    model.train()
    freeze_batchnorm_statistics(model)

    totals = {"loss": 0.0, "classification_loss": 0.0, "alignment_loss": 0.0}
    accurate_predictions = 0
    sample_count = 0
    steps = max(len(loader), 1)

    for step, batch in enumerate(loader):
        images, labels = batch[0].to(DEVICE).float(), batch[1].to(DEVICE)
        progress = min(max((epoch + step / steps) / max(num_epochs, 1), 0.0), 1.0)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=DEVICE.type, enabled=scaler is not None):
            if extra_loss_fn is None:
                outputs = model(images)
                classification_loss = criterion(outputs, labels)
                extra = None
            else:
                outputs, features = model(images, return_features=True)
                classification_loss = criterion(outputs, labels)
                extra = extra_loss_fn(model=model, batch=batch, source_features=features,
                                      progress=progress)
            loss = classification_loss if extra is None else classification_loss + extra

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        _, predicted_classes = torch.max(outputs.data, 1)
        accurate_predictions += (predicted_classes == labels).sum().item()
        batch_size = images.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["classification_loss"] += classification_loss.item() * batch_size
        totals["alignment_loss"] += (float(extra) if extra is not None else 0.0) * batch_size
        sample_count += batch_size

    result = {key: value / sample_count for key, value in totals.items()}
    result["accuracy"] = 100 * accurate_predictions / sample_count
    return result


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Return mean loss, accuracy (%) and macro-F1 (%) on a labelled loader."""
    model.eval()

    total_loss = 0.0
    sample_count = 0
    all_labels, all_predictions = [], []

    for batch in loader:
        images, labels = batch[0].to(DEVICE).float(), batch[1].to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        _, predicted_classes = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predicted_classes.cpu().tolist())

        total_loss += loss.item() * images.size(0)
        sample_count += images.size(0)

    accuracy = 100 * np.mean(np.array(all_labels) == np.array(all_predictions))
    macro_f1 = 100 * f1_score(all_labels, all_predictions, average="macro", zero_division=0)
    return total_loss / sample_count, float(accuracy), float(macro_f1)


def mean_source_validation_f1(model, validation_loaders, criterion):
    """Mean macro-F1 over the source-domain validation splits.

    This is the checkpoint-selection metric: no target labels are involved.
    """
    scores = {}
    for domain, loader in validation_loaders.items():
        _, _, macro_f1 = evaluate(model, loader, criterion)
        scores[domain] = macro_f1
    return float(np.mean(list(scores.values()))), scores


def train_model(model, method_name, train_loader, validation_loaders, criterion, optimizer,
                num_epochs=None, early_stopping_patience=None, extra_loss_fn=None, use_amp=None):
    """Fine-tune with early stopping on mean source-validation macro-F1.

    The best checkpoint and the history are written as soon as they improve,
    so a disconnected Colab session still leaves a usable model on Drive.
    Returns the per-epoch history; the model is left holding the best weights.
    """
    num_epochs = num_epochs or task_config.NUM_EPOCHS
    early_stopping_patience = early_stopping_patience or task_config.EARLY_STOPPING_PATIENCE
    use_amp = DEVICE.type == "cuda" if use_amp is None else use_amp
    scaler = torch.amp.GradScaler(DEVICE.type) if use_amp else None

    os.makedirs(task_config.MODEL_WEIGHTS_DIR, exist_ok=True)
    os.makedirs(task_config.TASK_RESULTS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(task_config.MODEL_WEIGHTS_DIR, f"{method_name}_best.pth")
    history_path = os.path.join(task_config.TASK_RESULTS_DIR, f"{method_name}_history.json")

    history = {"method": method_name, "train_loss": [], "train_acc": [],
               "classification_loss": [], "alignment_loss": [],
               "val_macro_f1": [], "val_per_domain": [], "epoch_seconds": [],
               "checkpoint": checkpoint_path, "mixed_precision": bool(scaler)}
    best_score = -1.0
    epochs_since_improvement = 0

    print(f"\nTraining {method_name}: up to {num_epochs} epochs, {len(train_loader)} steps each, "
          f"mixed precision {'on' if scaler else 'off'}")

    for epoch in range(num_epochs):
        started = time.time()
        stats = train_one_epoch(
            model, train_loader, criterion, optimizer, extra_loss_fn=extra_loss_fn, scaler=scaler,
            epoch=epoch, num_epochs=num_epochs,
        )
        val_macro_f1, per_domain = mean_source_validation_f1(model, validation_loaders, criterion)
        elapsed = time.time() - started

        history["train_loss"].append(stats["loss"])
        history["train_acc"].append(stats["accuracy"])
        history["classification_loss"].append(stats["classification_loss"])
        history["alignment_loss"].append(stats["alignment_loss"])
        history["val_macro_f1"].append(val_macro_f1)
        history["val_per_domain"].append(per_domain)
        history["epoch_seconds"].append(elapsed)

        detail = ", ".join(f"{d}: {s:.2f}" for d, s in per_domain.items())
        remaining = (num_epochs - epoch - 1) * elapsed / 60
        alignment = (f", align {stats['alignment_loss']:.4f}"
                     if extra_loss_fn is not None else "")
        print(f"Epoch {epoch + 1}/{num_epochs} | cls loss {stats['classification_loss']:.4f}"
              f"{alignment}, train acc {stats['accuracy']:.2f}% | "
              f"mean source val macro-F1 {val_macro_f1:.2f}% ({detail}) | "
              f"{elapsed:.0f}s, ~{remaining:.0f} min left")

        if val_macro_f1 > best_score:
            best_score = val_macro_f1
            epochs_since_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"   new best, saved {checkpoint_path}")
        else:
            epochs_since_improvement += 1

        history["best_val_macro_f1"] = best_score
        history["best_epoch"] = int(np.argmax(history["val_macro_f1"])) + 1
        with open(history_path, "w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

        if epochs_since_improvement >= early_stopping_patience:
            print(f"Early stopping after {epoch + 1} epochs.")
            break

    # Restore the best checkpoint, which may be from an earlier epoch.
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    print(f"Loaded best weights from epoch {history['best_epoch']} "
          f"(mean source val macro-F1 {best_score:.2f}%)")
    return history


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_run_manifest(methods, extra=None):
    """Record settings, checkpoint hashes and software versions for the run."""
    import sklearn
    import torchvision

    manifest = {
        "task": task_config.task_name,
        "seed": task_config.SEED,
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "source_domains": list(task_config.SOURCE_DOMAINS),
        "target_domain": task_config.TARGET_DOMAIN,
        "classes": list(task_config.PACS_CLASSES),
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": task_config.LEARNING_RATE,
            "weight_decay": task_config.WEIGHT_DECAY,
            "max_epochs": task_config.NUM_EPOCHS,
            "early_stopping_patience": task_config.EARLY_STOPPING_PATIENCE,
            "source_batch_per_domain": task_config.SOURCE_BATCH_PER_DOMAIN,
            "target_batch_size": task_config.TARGET_BATCH_SIZE,
            "checkpoint_selection": "mean macro-F1 over the three source validation splits",
        },
        "batchnorm_policy": "running statistics frozen at pretrained values; affine parameters trainable",
        "methods": {
            name: {
                "checkpoint": path,
                "sha256": file_sha256(path) if path and os.path.isfile(path) else None,
            }
            for name, path in (methods or {}).items()
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "sklearn": sklearn.__version__,
        },
        **(extra or {}),
    }

    os.makedirs(task_config.TASK_RESULTS_DIR, exist_ok=True)
    path = os.path.join(task_config.TASK_RESULTS_DIR, "run_manifest.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(f"Saved run manifest: {path}")
    return path


def evaluate_final(model, validation_loaders, target_loader, criterion):
    """Source-validation and target performance for the finished model.

    Target labels are read here only, after training and checkpoint selection
    are complete, which keeps the protocol honest: no target label ever
    influences training, early stopping or model choice.
    """
    results = {"source_validation": {}}

    for domain, loader in validation_loaders.items():
        loss, accuracy, macro_f1 = evaluate(model, loader, criterion)
        results["source_validation"][domain] = {
            "loss": loss, "accuracy_pct": accuracy, "macro_f1_pct": macro_f1,
        }

    scores = [d["macro_f1_pct"] for d in results["source_validation"].values()]
    results["mean_source_validation_macro_f1_pct"] = float(np.mean(scores))

    loss, accuracy, macro_f1 = evaluate(model, target_loader, criterion)
    results["target"] = {
        "domain": task_config.TARGET_DOMAIN,
        "loss": loss, "accuracy_pct": accuracy, "macro_f1_pct": macro_f1,
    }
    results["domain_gap_pp"] = results["mean_source_validation_macro_f1_pct"] - macro_f1
    return results


def run_method(method="erm", num_workers=2, num_epochs=None, use_amp=None,
               data_root=None, download_hf=False, lambda_mmd=None):
    """Train one Task 2 method and evaluate it once on the target.

    All methods share the loaders, the model, the optimizer settings and the
    checkpoint-selection rule; they differ only in the extra loss term and in
    whether target images are drawn during training:

      erm  source-only, no target data in the objective (the control)
      dan  + lambda_MMD * MMD^2 between source and target features
    """
    from assignment_01.task2.data.pacs import (
        DomainBalancedBatches, get_source_loaders, get_target_loaders,
    )
    from assignment_01.task2.models.backbones import build_model
    from assignment_01.task2.data.download import prepare_pacs

    domain_root = prepare_pacs(data_root, allow_huggingface=download_hf)

    source_loaders, validation_loaders = get_source_loaders(
        domain_root=domain_root, num_workers=num_workers)
    target_train_loader, target_eval_loader = get_target_loaders(
        domain_root=domain_root, num_workers=num_workers)

    extra_loss_fn = None
    settings = {}
    if method == "erm":
        # ERM never sees the target domain during training.
        target_train_loader = None
    elif method == "dan":
        from assignment_01.task2.methods.dan import make_dan_loss
        extra_loss_fn = make_dan_loss(lambda_mmd=lambda_mmd)
        settings = {"lambda_mmd": extra_loss_fn.lambda_value,
                    "bandwidth_multipliers": list(task_config.MMD_BANDWIDTH_MULTIPLIERS)}
        print(f"DAN: lambda_MMD = {extra_loss_fn.lambda_value}, "
              f"bandwidths = {settings['bandwidth_multipliers']} x median squared distance")
    else:
        raise SystemExit(f"Unknown method '{method}'.")

    train_batches = DomainBalancedBatches(source_loaders, target_loader=target_train_loader)

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=task_config.LEARNING_RATE,
        weight_decay=task_config.WEIGHT_DECAY,
    )

    run_name = method
    if method == "dan" and extra_loss_fn.lambda_value != task_config.DAN_LAMBDA_MMD:
        run_name = f"dan_lambda{extra_loss_fn.lambda_value:g}"   # keeps study runs separate

    history = train_model(
        model, run_name, train_batches, validation_loaders, criterion, optimizer,
        num_epochs=num_epochs, use_amp=use_amp, extra_loss_fn=extra_loss_fn,
    )
    history["settings"] = settings

    results = evaluate_final(model, validation_loaders, target_eval_loader, criterion)

    print(f"\nMean source validation macro-F1: "
          f"{results['mean_source_validation_macro_f1_pct']:.2f}%")
    print(f"Target ({results['target']['domain']}) macro-F1: "
          f"{results['target']['macro_f1_pct']:.2f}%, "
          f"accuracy: {results['target']['accuracy_pct']:.2f}%")
    print(f"Domain gap: {results['domain_gap_pp']:.2f} percentage points")

    os.makedirs(task_config.TASK_RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(task_config.TASK_RESULTS_DIR, f"{run_name}_results.json")
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump({"method": run_name, "settings": settings,
                   "history": history, "results": results}, file, indent=2)
    print(f"Saved results: {results_path}")

    save_run_manifest({run_name: history["checkpoint"]}, extra={"settings": settings})
    return history, results


def run_erm(**kwargs):
    """Step 1: source-only ERM (kept as a named entry point)."""
    return run_method("erm", **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Task 2: unsupervised domain adaptation on PACS")
    parser.add_argument("--mode", choices=["train", "infer", "dry-run"], required=True)
    parser.add_argument("--method", default="erm", choices=["erm", "dan"],
                        help="erm (source-only baseline) or dan (MMD alignment)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=None,
                        help="override config.NUM_EPOCHS; use --epochs 1 for a quick check")
    parser.add_argument("--data-root", default=None,
                        help="folder holding the PACS domain folders, or containing pacs.zip/.tar "
                             "(default: task_config.TASK_DATASET_DIR)")
    parser.add_argument("--download-hf", action="store_true",
                        help="if no local copy is found, download PACS from the Hugging Face hub "
                             "(flwrlabs/pacs) and cache it back to the dataset folder as pacs.zip")
    parser.add_argument("--lambda-mmd", type=float, default=None,
                        help="DAN only: override lambda_MMD (study uses 0.1, 1, 10)")
    parser.add_argument("--no-amp", action="store_true",
                        help="disable mixed precision (on by default on CUDA)")
    args = parser.parse_args()

    task_config.init_env()
    set_seed()

    if args.mode == "dry-run":
        print("Setup ready. Run --mode train --method erm to train the source-only baseline.")
        return

    if args.mode == "train":
        run_method(args.method, num_workers=args.num_workers, num_epochs=args.epochs,
                   use_amp=False if args.no_amp else None, data_root=args.data_root,
                   download_hf=args.download_hf, lambda_mmd=args.lambda_mmd)
        return

    raise SystemExit(f"TODO: --mode {args.mode} --method {args.method} is not implemented yet.")


if __name__ == "__main__":
    main()
