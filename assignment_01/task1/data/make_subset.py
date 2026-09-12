import os
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from assignment_01.task1.data.transforms import apply_universal_transforms

import config
from sklearn.model_selection import train_test_split
import numpy as np

task_config = config.TaskConfig(task_name='task1')

def get_train_val_subsets():
    
    train_ds_full = datasets.STL10(root=task_config.TASK_DATASET_DIR, 
                                   split='train', 
                                   download=True,
                                   transform=apply_universal_transforms)
    
    labels = train_ds_full.classes
    task_config.set_num_classes(len(labels))
    print(f"STL-10 labels: {labels}")
    print(f"Full train set size: {len(train_ds_full)}")

    train_indices, val_indices = train_test_split(
        list(range(len(train_ds_full))), 
        test_size=0.2, 
        stratify=[train_ds_full.targets[i] for i in range(len(train_ds_full))],
        random_state=task_config.SEED)

    train_subset = torch.utils.data.Subset(train_ds_full, train_indices)
    val_subset = torch.utils.data.Subset(train_ds_full, val_indices)

    train_loader = DataLoader(train_subset, batch_size=task_config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=task_config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train Samples: {len(train_subset)} | Val Samples: {len(val_subset)}")
    
    return train_loader, val_loader


def get_test_subset():

    test_ds_full = datasets.STL10(root=task_config.TASK_DATASET_DIR, split='test', download=True)

    if os.path.exists(task_config.SELECTED_INDICES_FILE):
        selected_indices = np.load(task_config.SELECTED_INDICES_FILE).tolist()
        print(f"Loaded selected indices from {task_config.SELECTED_INDICES_FILE}")
    else:
        selected_indices = []
        for class_idx in range(task_config.NUM_CLASSES):
            class_indices = [i for i, label in enumerate(test_ds_full.targets) if label == class_idx]
            
            np.random.default_rng(seed=task_config.SEED).shuffle(class_indices)
            
            available_count = len(class_indices)
            
            if available_count < task_config.TARGET_PER_CLASS:
                print(f"Not enough samples for class {class_idx}. Required: {task_config.TARGET_PER_CLASS}, Available: {available_count}")
                
                choosen_for_class = class_indices
                imbalance_count = task_config.TARGET_PER_CLASS - available_count
                print(f"Imbalance count for class {class_idx}: {imbalance_count}")
                imbalance_documented = True
            else:
                choosen_for_class = class_indices[:config.TARGET_PER_CLASS]
                imbalance_documented = False
        
        selected_indices.extend(choosen_for_class)
            
        if not imbalance_documented:
            print(f"Selected {len(choosen_for_class)} samples for class {class_idx}.")
            
        np.save(task_config.SELECTED_INDICES_FILE, np.array(selected_indices))
        print(f"Saved selected indices to {task_config.SELECTED_INDICES_FILE}")

    test_subset = torch.utils.data.Subset(test_ds_full, selected_indices)
    print(f"Test subset size: {len(test_subset)}")

