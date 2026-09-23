"""Cache penultimate features and logits for every split, once per model.

Every score must see exactly the same examples and the same model outputs, so
the outputs are extracted once and written to a .npz cache that the scoring
step reads. That also makes the comparison between MSP, MLS, Energy and
Mahalanobis a pure difference in score definition rather than in data handling.

This is the first script that loads CIFAR-100, which is legitimate: the model
is already trained and its checkpoint already selected. Nothing here writes to
a checkpoint or influences a training decision.

    python -m assignment_01.task4.scripts.extract_outputs --method vanilla

Cached per model:
    train_features/train_labels   unaugmented, for the Mahalanobis fit
    val_features/val_logits       for threshold calibration
    test_features/test_logits     known-class evaluation (CSA + acceptance)
    near_*/far_*                  the two unknown groups, plus class names
"""

import argparse
import os

import numpy as np
import torch

from assignment_01.task4.config import task_config
from assignment_01.task4.data.cifar import get_cifar10_loaders, get_unknown_loaders
from assignment_01.task4.models.resnet_cifar import build_model
from assignment_01.task4.scripts.train import DEVICE, set_seed


def cache_path(method):
    return os.path.join(task_config.CACHE_DIR, f"{method}_outputs.npz")


def load_trained_model(method):
    """Rebuild the model and restore its selected checkpoint.

    PROSER is wrapped so its dummy units are restored too; its logits are
    therefore [N, 10 + C_dummy] while Vanilla and GCSC give [N, 10]. The
    scoring step slices the known columns where it needs them.
    """
    path = task_config.checkpoint_path(method)
    if not os.path.isfile(path):
        raise SystemExit(f"No checkpoint for '{method}' at {path}. Train it first.")

    backbone = build_model()
    if method == "proser":
        from assignment_01.task4.methods.proser import ProserNet
        model = ProserNet(backbone).to(DEVICE)
    else:
        model = backbone
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


@torch.no_grad()
def collect(model, loader, with_names=False):
    """Run a loader through the model and gather features, logits and labels."""
    features, logits, labels, names = [], [], [], []

    for batch in loader:
        images = batch[0].to(DEVICE)
        batch_logits, batch_features = model(images, return_features=True)
        features.append(batch_features.cpu().numpy())
        logits.append(batch_logits.cpu().numpy())
        labels.append(np.asarray(batch[1]))
        if with_names:
            names.extend(batch[2])

    collected = {
        "features": np.concatenate(features).astype(np.float64),
        "logits": np.concatenate(logits).astype(np.float64),
        "labels": np.concatenate(labels),
    }
    if with_names:
        collected["names"] = np.array(names)
    return collected


def run(method, num_workers=2, data_root=None):
    task_config.init_env()
    set_seed(task_config.SEED)

    model = load_trained_model(method)

    # Training split with *no* augmentation: the Mahalanobis fit requires it.
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        root=data_root, num_workers=num_workers, train_eval_transform=True)
    unknown_loaders = get_unknown_loaders(root=data_root, num_workers=num_workers)

    print(f"\nExtracting outputs for {method} ...")
    train = collect(model, train_loader)
    validation = collect(model, val_loader)
    test = collect(model, test_loader)
    near = collect(model, unknown_loaders["near"], with_names=True)
    far = collect(model, unknown_loaders["far"], with_names=True)

    payload = {
        "method": method,
        "train_features": train["features"], "train_labels": train["labels"],
        "val_features": validation["features"], "val_logits": validation["logits"],
        "val_labels": validation["labels"],
        "test_features": test["features"], "test_logits": test["logits"],
        "test_labels": test["labels"],
        "near_features": near["features"], "near_logits": near["logits"],
        "near_names": near["names"],
        "far_features": far["features"], "far_logits": far["logits"],
        "far_names": far["names"],
    }

    os.makedirs(task_config.CACHE_DIR, exist_ok=True)
    path = cache_path(method)
    np.savez_compressed(path, **payload)

    print(f"   train {train['features'].shape}, val {validation['features'].shape}, "
          f"test {test['features'].shape}")
    print(f"   near {near['features'].shape}, far {far['features'].shape}")
    print(f"   logits {test['logits'].shape[1]} columns")
    print(f"Saved {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["vanilla", "gcsc", "proser"], required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    run(args.method, num_workers=args.num_workers, data_root=args.data_root)


if __name__ == "__main__":
    main()
