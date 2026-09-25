"""Train Task 4 Vanilla, GCSC, or PROSER models using CIFAR-10 only."""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from assignment_01.task4.config import task_config
from assignment_01.task4.data.cifar import get_cifar10_loaders
from assignment_01.task4.models.resnet_cifar import build_model

DEVICE = task_config.DEVICE


def set_seed(seed=None):
    seed = task_config.SEED if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, criterion=None, num_known=None):
    """Return mean loss and known-class accuracy."""
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()
    total_loss, correct, seen = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        if num_known is not None:
            logits = logits[:, :num_known]
        total_loss += criterion(logits, labels).item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += images.size(0)

    return total_loss / seen, 100.0 * correct / seen


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, seen = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += images.size(0)

    return {"loss": total_loss / seen, "accuracy": 100.0 * correct / seen}


def run_standard(method, num_workers=2, num_epochs=None, data_root=None):
    """Vanilla (no RandAugment) or GCSC (with RandAugment)."""
    randaugment = method == "gcsc"
    num_epochs = num_epochs or task_config.NUM_EPOCHS

    train_loader, val_loader, test_loader = get_cifar10_loaders(
        root=data_root, num_workers=num_workers, randaugment=randaugment)

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=task_config.LEARNING_RATE,
        momentum=task_config.MOMENTUM, weight_decay=task_config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"\nTraining {method}: {num_epochs} epochs, {len(train_loader)} steps each, "
          f"SGD lr {task_config.LEARNING_RATE} cosine"
          f"{', RandAugment(2, 9)' if randaugment else ''}")

    history = {"method": method, "train_loss": [], "train_acc": [], "val_acc": [],
               "learning_rate": [], "epoch_seconds": []}
    checkpoint_path = task_config.checkpoint_path(method)
    best_accuracy = -1.0

    for epoch in range(num_epochs):
        started = time.time()
        stats = train_one_epoch(model, train_loader, criterion, optimizer)
        _, val_accuracy = evaluate(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - started

        history["train_loss"].append(stats["loss"])
        history["train_acc"].append(stats["accuracy"])
        history["val_acc"].append(val_accuracy)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        history["epoch_seconds"].append(elapsed)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            marker = "  new best"
        else:
            marker = ""

        remaining = (num_epochs - epoch - 1) * elapsed / 60
        print(f"Epoch {epoch + 1}/{num_epochs} | loss {stats['loss']:.4f}, "
              f"train acc {stats['accuracy']:.2f}% | val acc {val_accuracy:.2f}%{marker} | "
              f"{elapsed:.0f}s, ~{remaining:.0f} min left")

        history["best_val_acc"] = best_accuracy
        history["best_epoch"] = int(np.argmax(history["val_acc"])) + 1
        _save_history(history, method)

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    _, test_accuracy = evaluate(model, test_loader, criterion)
    history["cifar10_test_accuracy"] = test_accuracy
    print(f"\nBest epoch {history['best_epoch']} (val {best_accuracy:.2f}%)")
    print(f"CIFAR-10 test accuracy (CSA): {test_accuracy:.2f}%")
    _save_history(history, method)
    return history


def run_proser(num_workers=2, num_epochs=None, data_root=None):
    """Fine-tune the selected Vanilla checkpoint with placeholder objectives."""
    from assignment_01.task4.methods.proser import ProserNet, train_one_epoch as proser_epoch

    num_epochs = num_epochs or task_config.PROSER_EPOCHS
    vanilla_path = task_config.checkpoint_path("vanilla")
    if not os.path.isfile(vanilla_path):
        raise SystemExit(f"PROSER needs the Vanilla checkpoint: {vanilla_path}. "
                         "Train --method vanilla first.")

    # PROSER uses the Vanilla augmentation recipe, not GCSC's.
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        root=data_root, num_workers=num_workers, randaugment=False)

    backbone = build_model()
    backbone.load_state_dict(torch.load(vanilla_path, map_location=DEVICE))
    model = ProserNet(backbone).to(DEVICE)
    print(f"PROSER: initialized from {vanilla_path}, "
          f"{model.num_dummy} dummy classifiers appended")

    optimizer = torch.optim.SGD(
        model.parameters(), lr=task_config.PROSER_LR,
        momentum=task_config.MOMENTUM, weight_decay=task_config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"\nTraining proser: {num_epochs} epochs, beta {task_config.PROSER_BETA}, "
          f"gamma {task_config.PROSER_GAMMA}, mixup Beta(2,2) after layer2")

    history = {"method": "proser", "train_loss": [], "classifier_loss": [],
               "data_loss": [], "train_acc": [], "val_acc": [], "epoch_seconds": []}
    checkpoint_path = task_config.checkpoint_path("proser")
    best_accuracy = -1.0

    for epoch in range(num_epochs):
        started = time.time()
        stats = proser_epoch(model, train_loader, optimizer)
        # Selection uses known-class validation accuracy only.
        _, val_accuracy = evaluate(model, val_loader, num_known=model.num_known)
        scheduler.step()
        elapsed = time.time() - started

        history["train_loss"].append(stats["loss"])
        history["classifier_loss"].append(stats["classifier"])
        history["data_loss"].append(stats["data"])
        history["train_acc"].append(stats["accuracy"])
        history["val_acc"].append(val_accuracy)
        history["epoch_seconds"].append(elapsed)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            marker = "  new best"
        else:
            marker = ""

        remaining = (num_epochs - epoch - 1) * elapsed / 60
        print(f"Epoch {epoch + 1}/{num_epochs} | loss {stats['loss']:.4f} "
              f"(cls {stats['classifier']:.4f}, data {stats['data']:.4f}), "
              f"train acc {stats['accuracy']:.2f}% | val acc {val_accuracy:.2f}%{marker} | "
              f"{elapsed:.0f}s, ~{remaining:.0f} min left")

        history["best_val_acc"] = best_accuracy
        history["best_epoch"] = int(np.argmax(history["val_acc"])) + 1
        _save_history(history, "proser")

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    _, test_accuracy = evaluate(model, test_loader, num_known=model.num_known)
    history["cifar10_test_accuracy"] = test_accuracy
    print(f"\nBest epoch {history['best_epoch']} (val {best_accuracy:.2f}%)")
    print(f"CIFAR-10 test accuracy (CSA, known logits only): {test_accuracy:.2f}%")
    _save_history(history, "proser")
    return history


def _save_history(history, method):
    os.makedirs(task_config.TASK_RESULTS_DIR, exist_ok=True)
    path = os.path.join(task_config.TASK_RESULTS_DIR, f"{method}_history.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["vanilla", "gcsc", "proser"], required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    task_config.init_env()
    set_seed(task_config.SEED)

    if args.method == "proser":
        run_proser(num_workers=args.num_workers, num_epochs=args.epochs,
                   data_root=args.data_root)
    else:
        run_standard(args.method, num_workers=args.num_workers,
                     num_epochs=args.epochs, data_root=args.data_root)


if __name__ == "__main__":
    main()
