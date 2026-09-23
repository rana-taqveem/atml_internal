"""CIFAR-10 knowns and the fixed CIFAR-100 unknown groups.

Three things live here:

  * the stratified 90/10 split of the CIFAR-10 training partition (seed 6304),
    written to disk as indices so every model and every score sees exactly the
    same examples;
  * the three CIFAR-10 loaders (train / validation / test);
  * the two unknown loaders, built from the fixed CIFAR-100 *test* classes.

Two rules from the assignment shape this file. CIFAR-100 training images may
never be used, so only the test partition is read. And unknowns are evaluation
data, so nothing here is called by train.py - only extract_outputs.py and
evaluate_osr.py touch the unknown loaders.

Augmentation differs between Vanilla and GCSC by exactly one transform
(RandAugment), inserted after the crop and flip and before tensor conversion
and normalization, which is where the assignment specifies it.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from assignment_01.task4.config import task_config


def _normalize():
    return transforms.Normalize(task_config.NORMALIZE_MEAN, task_config.NORMALIZE_STD)


def train_transform(randaugment=False):
    """Crop and flip, optionally RandAugment, then tensor + normalize.

    randaugment=False is the Vanilla recipe; True is GCSC. That single
    difference is the whole controlled comparison, so the two share this
    function rather than duplicating the pipeline.
    """
    steps = [
        transforms.RandomCrop(task_config.CROP_SIZE, padding=task_config.CROP_PADDING),
        transforms.RandomHorizontalFlip(),
    ]
    if randaugment:
        steps.append(transforms.RandAugment(
            num_ops=task_config.RANDAUGMENT_NUM_OPS,
            magnitude=task_config.RANDAUGMENT_MAGNITUDE,
        ))
    steps += [transforms.ToTensor(), _normalize()]
    return transforms.Compose(steps)


def eval_transform():
    """No augmentation: validation, test, unknowns, and Mahalanobis fitting."""
    return transforms.Compose([transforms.ToTensor(), _normalize()])


def _split_path():
    return os.path.join(task_config.TASK_DATASET_DIR, "cifar10_split_seed6304.json")


def make_splits(root=None, seed=None, val_fraction=None):
    """Stratified 90/10 split of the CIFAR-10 train partition, cached to disk.

    Stratified means each of the ten classes keeps the same 90/10 proportion,
    so the validation split is class-balanced. The indices are saved and
    reused, because the rejection threshold is calibrated on this exact
    validation set and every model must share it.
    """
    root = root or task_config.TASK_DATASET_DIR
    seed = task_config.SEED if seed is None else seed
    val_fraction = task_config.TRAIN_VAL_SPLIT if val_fraction is None else val_fraction

    path = _split_path()
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as file:
            saved = json.load(file)
        return saved["train_indices"], saved["val_indices"]

    dataset = CIFAR10(root=root, train=True, download=True)
    labels = np.array(dataset.targets)

    generator = np.random.RandomState(seed)
    train_indices, val_indices = [], []
    for class_index in range(task_config.NUM_CLASSES):
        positions = np.where(labels == class_index)[0]
        generator.shuffle(positions)
        cut = int(round(len(positions) * val_fraction))
        val_indices.extend(int(i) for i in positions[:cut])
        train_indices.extend(int(i) for i in positions[cut:])

    train_indices.sort()
    val_indices.sort()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump({"seed": seed, "val_fraction": val_fraction,
                   "train_indices": train_indices, "val_indices": val_indices}, file)
    print(f"Wrote CIFAR-10 split: {len(train_indices)} train / {len(val_indices)} val "
          f"-> {path}")
    return train_indices, val_indices


def get_cifar10_loaders(root=None, batch_size=None, num_workers=2, randaugment=False,
                        train_eval_transform=False):
    """Train / validation / test loaders for the ten known classes.

    train_eval_transform=True returns the training split with no augmentation,
    which is what the Mahalanobis score needs: the assignment requires class
    means and the shared covariance to be estimated from unaugmented training
    features.
    """
    root = root or task_config.TASK_DATASET_DIR
    batch_size = batch_size or task_config.BATCH_SIZE
    train_indices, val_indices = make_splits(root=root)

    augmentation = (eval_transform() if train_eval_transform
                    else train_transform(randaugment=randaugment))
    train_base = CIFAR10(root=root, train=True, download=True, transform=augmentation)
    eval_base = CIFAR10(root=root, train=True, download=True, transform=eval_transform())
    test_set = CIFAR10(root=root, train=False, download=True, transform=eval_transform())

    train_set = Subset(train_base, train_indices)
    val_set = Subset(eval_base, val_indices)

    shuffle = not train_eval_transform
    common = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                              drop_last=False, **common)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **common)

    print(f"CIFAR-10: {len(train_set)} train / {len(val_set)} val / {len(test_set)} test"
          f"{' (RandAugment)' if randaugment and not train_eval_transform else ''}")
    return train_loader, val_loader, test_loader


class _UnknownSubset(torch.utils.data.Dataset):
    """CIFAR-100 test images from selected fine classes, relabelled.

    Every unknown carries label -1: they have no known class, and a negative
    label makes it impossible to score them as if they did by accident. The
    original fine-class name is returned alongside so the failure analysis can
    report which unknown class was accepted.
    """

    def __init__(self, dataset, indices, class_names):
        self.dataset = dataset
        self.indices = indices
        self.class_names = class_names

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        image, fine_label = self.dataset[self.indices[position]]
        return image, -1, self.class_names[fine_label]


def get_unknown_loaders(root=None, batch_size=None, num_workers=2):
    """Near and far unknown loaders, 800 images each, evaluation only.

    Called by extract_outputs.py and evaluate_osr.py and by nothing else: no
    CIFAR-100 image may reach training, checkpoint selection, score design or
    threshold selection.
    """
    root = root or task_config.TASK_DATASET_DIR
    batch_size = batch_size or task_config.BATCH_SIZE

    dataset = CIFAR100(root=root, train=False, download=True, transform=eval_transform())
    names = dataset.classes
    targets = np.array(dataset.targets)

    loaders = {}
    for group, wanted in (("near", task_config.NEAR_UNKNOWN_CLASSES),
                          ("far", task_config.FAR_UNKNOWN_CLASSES)):
        missing = [name for name in wanted if name not in names]
        if missing:
            raise ValueError(f"CIFAR-100 classes not found: {missing}")
        wanted_ids = [names.index(name) for name in wanted]
        indices = [int(i) for i in np.where(np.isin(targets, wanted_ids))[0]]
        subset = _UnknownSubset(dataset, indices, names)
        loaders[group] = DataLoader(subset, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True)
        print(f"{group} unknowns: {len(indices)} images across "
              f"{len(wanted)} CIFAR-100 classes")
    return loaders
