"""PACS datasets, stratified splits, and domain-balanced loaders."""

from pathlib import Path

import torch
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from assignment_01.task2.config import task_config
from assignment_01.task2.data.download import prepare_pacs, resolve_domain_dir

IMAGENET_NORMALIZATION = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

TRAIN_TRANSFORMS = T.Compose([
    T.Resize((task_config.RESIZE, task_config.RESIZE)),
    T.RandomCrop(task_config.CROP),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    IMAGENET_NORMALIZATION,
])

EVAL_TRANSFORMS = T.Compose([
    T.Resize((task_config.RESIZE, task_config.RESIZE)),
    T.CenterCrop(task_config.CROP),
    T.ToTensor(),
    IMAGENET_NORMALIZATION,
])


def get_domain_dataset(domain, transform=EVAL_TRANSFORMS, domain_root=None):
    """ImageFolder for one PACS domain, with a fixed class order."""
    
    domain_root = domain_root or prepare_pacs()
    dataset = ImageFolder(str(resolve_domain_dir(domain_root, domain)), 
                          transform=transform)

    expected = list(task_config.PACS_CLASSES)
    
    if [c.lower() for c in dataset.classes] != [c.lower() for c in expected]:
        raise ValueError(
            f"Class order for '{domain}' is {dataset.classes}, expected {expected}. "
            "Update task_config.PACS_CLASSES so label indices stay consistent."
        )
    return dataset


def split_domain(domain, domain_root=None, seed=task_config.SEED):
    """Create a stratified train/validation split with separate transforms."""
    domain_root = domain_root or prepare_pacs()
    
    train_view = get_domain_dataset(domain, TRAIN_TRANSFORMS, domain_root)
    eval_view = get_domain_dataset(domain, EVAL_TRANSFORMS, domain_root)
    
    labels = [label for _, label in train_view.samples]

    train_indices, val_indices = train_test_split(
        list(range(len(train_view))),
        test_size=task_config.TRAIN_VAL_SPLIT,
        stratify=labels,
        random_state=seed,
    )
    
    return Subset(train_view, train_indices), Subset(eval_view, val_indices)


def get_train_val_dataloaders(domain, domain_root=None, batch_per_domain=None, num_workers=2):
    """Return train and validation loaders for one source domain."""
    batch_per_domain = batch_per_domain or task_config.SOURCE_BATCH_PER_DOMAIN
    train_subset, val_subset = split_domain(domain, domain_root)

    train_loader = DataLoader(
        train_subset, batch_size=batch_per_domain, shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=task_config.TARGET_BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"{domain} Data Setup Complete:")
    print(f" -> Total Train Samples: {len(train_subset)} ({len(train_loader)} batches)")
    print(f" -> Total Val Samples:   {len(val_subset)} ({len(val_loader)} batches)")

    return train_loader, val_loader


def get_source_loaders(domains=None, domain_root=None, batch_per_domain=None, num_workers=2):
    """Return per-domain train and validation loaders for source domains."""
    domains = domains or task_config.SOURCE_DOMAINS
    domain_root = domain_root or prepare_pacs()

    train_loaders, validation_loaders = {}, {}
    for domain in domains:
        train_loaders[domain], validation_loaders[domain] = get_train_val_dataloaders(
            domain, domain_root, batch_per_domain, num_workers
        )

    return train_loaders, validation_loaders


def get_target_loaders(domain=None, domain_root=None, batch_size=None, num_workers=2):
    """Return adaptation and evaluation loaders for the target domain."""
    domain = domain or task_config.TARGET_DOMAIN
    batch_size = batch_size or task_config.TARGET_BATCH_SIZE
    domain_root = domain_root or prepare_pacs()

    adaptation = DataLoader(
        get_domain_dataset(domain, TRAIN_TRANSFORMS, domain_root),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        drop_last=True, pin_memory=True,
    )
    evaluation = DataLoader(
        get_domain_dataset(domain, EVAL_TRANSFORMS, domain_root),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    print(f"{domain} (target): {len(adaptation.dataset)} images")
    return adaptation, evaluation


def cycle(loader):
    """Yield batches forever, restarting the loader when it is exhausted."""
    while True:
        for batch in loader:
            yield batch


class DomainBalancedBatches:
    """Yield balanced source batches and optional unlabeled target images."""

    def __init__(self, source_loaders, target_loader=None, steps_per_epoch=None):
        self.source_loaders = source_loaders
        self.target_loader = target_loader
        self.domains = list(source_loaders)
        self.steps_per_epoch = steps_per_epoch or max(len(l) for l in source_loaders.values())

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        sources = {domain: cycle(loader) for domain, loader in self.source_loaders.items()}
        target = cycle(self.target_loader) if self.target_loader is not None else None

        for _ in range(self.steps_per_epoch):
            images, labels, domain_ids = [], [], []
            for index, domain in enumerate(self.domains):
                # one batch per domain: [8, 3, 224, 224] and [8]
                batch_images, batch_labels = next(sources[domain])[:2]
                images.append(batch_images)
                labels.append(batch_labels)
                domain_ids.append(torch.full((batch_images.size(0),), index, dtype=torch.long))

            # [0] keeps only the images: the target's labels must not be used.
            target_images = next(target)[0] if target is not None else None
            # cat over the 3 domains: 3 x [8, ...] -> [24, ...]
            yield (torch.cat(images), torch.cat(labels), torch.cat(domain_ids), target_images)
