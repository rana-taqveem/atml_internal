import os
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, TensorDataset
from assignment_01.task1.data.transforms import *
from assignment_01.task1.config import task_config
from assignment_01.task1.data.download import prepare_stl10
import numpy as np

def get_test_subset(
    transformation_type = 'baseline',
    *,
    hue_rotation_angle = 0,
    translation_x = 0,
    translation_y = 0,):
    
    allowed_transformations = ['baseline', 'grey_scale', 'hue', 'translation', 'patch_shuffle', 'style_transfer']
    
    if transformation_type not in allowed_transformations:
        raise ValueError(f"Invalid transformation_type. Allowed values are: {allowed_transformations}")
    
    prepare_stl10(task_config.TASK_DATASET_DIR)
    test_ds_full = datasets.STL10(root=task_config.TASK_DATASET_DIR,
                                  split='test', 
                                  download=False,
                                  transform=apply_universal_transforms)
    
    if os.path.exists(task_config.SELECTED_INDICES_FILE):
        selected_indices = np.load(task_config.SELECTED_INDICES_FILE).tolist()
        print(f"Loaded selected indices from {task_config.SELECTED_INDICES_FILE}")
    else:
        rng = np.random.default_rng(seed=task_config.SEED)
        selected_indices = []
        for class_idx in range(task_config.NUM_CLASSES):
            class_indices = np.flatnonzero(
                np.array(test_ds_full.labels) == class_idx)
            
            rng.shuffle(class_indices)
            
            choosen = class_indices[:task_config.TARGET_PER_CLASS]
            available_count = len(class_indices)
            
            if len(choosen) < task_config.TARGET_PER_CLASS:
                print(f"Not enough samples for class {class_idx}. Required: {task_config.TARGET_PER_CLASS}, Available: {available_count}")
                
            selected_indices.extend(choosen)
       
    np.save(task_config.SELECTED_INDICES_FILE, np.array(selected_indices))
    print(f"Saved selected indices to {task_config.SELECTED_INDICES_FILE}")
    
    selected_labels = np.asarray(test_ds_full.labels)[selected_indices]
    class_counts = np.bincount(selected_labels, minlength=task_config.NUM_CLASSES)
    print(f"Class counts in selected subset: {class_counts}")
    
    if (len(selected_indices) < task_config.TOTAL_TARGET_IMAGES 
        or not np.all(class_counts == task_config.TARGET_PER_CLASS)):
        print(f"Warning: Not enough samples to meet the target of {task_config.TOTAL_TARGET_IMAGES} images. Only {len(selected_indices)} were selected.")
        
    patch_shuffler = None
    if transformation_type == 'patch_shuffle':
        patch_shuffler = PatchShuffler(patch_size=566, seed=task_config.SEED)
    
    images = []
    labels = []
    permutations = []
    
    for image_id in selected_indices:
        image, label = test_ds_full[image_id]
        
        if transformation_type == 'grey_scale':
            image = apply_grey_scale(image)
            
        elif transformation_type == 'hue':
            image = apply_fixed_hue_rotation(image, 
                                                hue_rotation_angle)
        
        elif transformation_type == 'translation':
            image = apply_translation(image, 
                                        translation_x, 
                                        translation_y)
        
        elif transformation_type == 'patch_shuffle':
            image, permutation = patch_shuffler.shuffle_patches(image)
            permutations.append(permutation)
        
        images.append(image)
        labels.append(label)   
        
    dataset = TensorDataset(torch.stack(images), 
                            torch.tensor(labels, dtype=torch.long), 
                            torch.tensor(selected_indices, dtype=torch.long))  
    
    metadata = {
        "image_ids": selected_indices,
        "transformations": transformation_type,
        "seed": task_config.SEED,
    }
    
    if transformation_type == 'patch_shuffle':
        metadata["permutations"] = permutations
        metadata["patch_size"] = patch_shuffler.patch_size
    
    elif transformation_type == 'hue':
        metadata["hue_rotation_angle"] = hue_rotation_angle
    elif transformation_type == 'translation':
        metadata["translation_x"] = translation_x
        metadata["translation_y"] = translation_y
        
    return dataset, metadata
