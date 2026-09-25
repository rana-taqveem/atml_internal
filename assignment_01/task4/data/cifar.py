"""CIFAR-10 splits and evaluation-only CIFAR-100 unknown loaders."""

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
    """Build the Vanilla or RandAugment training transform."""
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
    """Create or load the stratified CIFAR-10 train/validation split."""
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
    """Return CIFAR-10 train, validation, and test loaders."""
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
    """Selected CIFAR-100 test images with label -1 and original class names."""

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
    """Return evaluation-only near and far unknown loaders."""
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
