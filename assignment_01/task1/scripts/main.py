import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import Subset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from config import task_config
from assignment_01.task1.data.transforms import apply_universal_transforms
from data.transforms import apply_normalization
from models.backbones import (
    Resnet50Backbone, 
    Torchvision_Vit_B_16_Backbone, 
    Openai_Clip_Backbone,
    LinearClassifier
)

from sklearn.metrics.pairwise import cosine_similarity

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {DEVICE}")


def get_train_val_dataloaders():
    
    train_ds_full = datasets.STL10(root=task_config.TASK_DATASET_DIR, 
                                   split='train', 
                                   download=True,
                                   transform=apply_universal_transforms)
    
    class_names = train_ds_full.classes
    task_config.set_num_classes(len(class_names))

    train_indices, val_indices = train_test_split(
        list(range(len(train_ds_full))), 
        test_size=task_config.TRAIN_VAL_SPLIT, 
        stratify=train_ds_full.labels,
        random_state=task_config.SEED)

    train_subset = Subset(train_ds_full, train_indices)
    val_subset = Subset(train_ds_full, val_indices)

    train_loader = DataLoader(train_subset, batch_size=task_config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=task_config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"STL-10 Data Setup Complete:")
    print(f" -> Total Train Samples: {len(train_subset)} ({len(train_loader)} batches)")
    print(f" -> Total Val Samples:   {len(val_subset)} ({len(val_loader)} batches)")
    
    return train_loader, val_loader

def train_one_epoch(model, loader, criterion, optimizer, scaler=None):
    model.train()

    total_loss = 0.0
    accurate_predictions = 0
    sample_count = 0
    
    for images, labels in loader:
      optimizer.zero_grad()
      images = images.to(DEVICE).float()
      labels = labels.to(DEVICE)

      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
    
      _, predicted_classes = torch.max(outputs.data, 1)
      accurate_predictions += (predicted_classes == labels).sum().item()

      total_loss += loss.item() * images.size(0)
      sample_count += images.size(0)

    avg_loss = total_loss / sample_count
    accuracy = 100 * accurate_predictions / sample_count

    return avg_loss, accuracy

def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    accurate_predictions = 0
    sample_count = 0

    with torch.no_grad():
      for images, labels in loader:
        images = images.to(DEVICE).float()
        labels = labels.to(DEVICE)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        _, predicted_classes = torch.max(outputs.data, 1)
        accurate_predictions += (predicted_classes == labels).sum().item()

        total_loss += loss.item() * images.size(0)
        sample_count += images.size(0)

    avg_loss = total_loss / sample_count
    accuracy = 100 * accurate_predictions / sample_count

    return avg_loss, accuracy

def train_model_head(model, model_name, train_loader, val_loader, criterion, optimizer, num_epochs=7, scaler=None, early_stopping_patience=5):
  
  TRAIN_LOSS = 'train_loss'
  TRAIN_ACC = 'train_acc'
  VAL_LOSS = 'val_loss'
  VAL_ACC = 'val_acc'

  results = { TRAIN_LOSS: [], TRAIN_ACC: [],
              VAL_LOSS: [],   VAL_ACC : [] }
  
  best_val_acc = 0.0
  best_weights = None
  epoch_since_improvement = 0

  for epoch in range(num_epochs):
      
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler=scaler)
    
    results[TRAIN_LOSS].append(train_loss)
    results[TRAIN_ACC].append(train_acc)

    val_loss, val_acc = evaluate(model, val_loader, criterion=criterion)
    results[VAL_LOSS].append(val_loss)
    results[VAL_ACC].append(val_acc)

    print(f"Epoch: {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss  : {val_loss:.4f}  , Val Acc  : {val_acc:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_weights = model.state_dict().copy()
        epoch_since_improvement = 0
    else:
        epoch_since_improvement += 1
    
    if epoch_since_improvement >= early_stopping_patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

  if best_weights is not None:
    model.load_state_dict(best_weights)
    print(f"Loaded best model weights with Val Acc: {best_val_acc:.2f}%")
    
  torch.save(model.state_dict(), task_config.MODEL_WEIGHTS_DIR + f"/{model_name}_{best_val_acc:.2f}.pth")
  print(f"Saved optimized best classifier head cleanly to {task_config.MODEL_WEIGHTS_DIR}/{model_name}_{best_val_acc:.2f}.pth")
        
  return results

def extract_backbone_features(model, loader, cache_path=None):

    if cache_path and os.path.exists(cache_path):
        data = torch.load(cache_path)
        return data['X'], data['y']

    model.to(DEVICE)
    model.eval()
    feats, labels = [], []
    
    with torch.no_grad():
        for images, y in tqdm(loader, desc="extracting features"):
            images = images.to(DEVICE)
            x = model(images)

            feats.append(x.cpu())
            labels.append(y)

    X = torch.cat(feats, dim=0)
    y_all = torch.cat(labels, dim=0)

    if cache_path:
        torch.save({'X': X, 'y': y_all}, cache_path)
        print(f"Saved features to {cache_path}")

    return X, y_all

def train(model, model_name, train_loader, val_loader, num_classes):

    train_cache_path=f"{task_config.TASK_FEATURES_DIR}/{model_name}_train_features.pt"
    val_cache_path=f"{task_config.TASK_FEATURES_DIR}/{model_name}_val_features.pt"
    
    X_train, y_train = extract_backbone_features(model, train_loader, cache_path=train_cache_path)
    X_val, y_val = extract_backbone_features(model, val_loader, cache_path=val_cache_path)

    train_loader_f = DataLoader(TensorDataset(X_train, y_train), batch_size=task_config.BATCH_SIZE, shuffle=True)
    val_loader_f = DataLoader(TensorDataset(X_val, y_val), batch_size=task_config.BATCH_SIZE, shuffle=False)
    
    in_features = X_train.shape[1]
    classifier_head = nn.Linear(in_features, num_classes).to(DEVICE)

    optimizer = optim.AdamW(classifier_head.parameters(), lr=task_config.LEARNING_RATE, weight_decay=task_config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    train_loader_f = DataLoader(TensorDataset(X_train, y_train), batch_size=task_config.BATCH_SIZE, shuffle=True)
    val_loader_f = DataLoader(TensorDataset(X_val, y_val), batch_size=task_config.BATCH_SIZE, shuffle=False)

    results = train_model_head(classifier_head, model_name, train_loader_f, val_loader_f, criterion, optimizer, num_epochs=task_config.NUM_EPOCHS)

    checkpoint_path = os.path.join(task_config.TASK_CHECKPOINTS_DIR, f"{model_name}_head.pth")
    torch.save(classifier_head.state_dict(), checkpoint_path)
    print(f"Saved classifier head to {checkpoint_path}")
    
    del model, classifier_head
    torch.cuda.empty_cache()
    gc.collect()
    
    return results


def main():
    
    task_config.init_env()
    train_loader, val_loader = get_train_val_dataloaders()
    
    resnet = Resnet50Backbone()
    vit = Torchvision_Vit_B_16_Backbone()
    clip = Openai_Clip_Backbone()
    
    print("Starting training for ResNet50 ...")
    resnet_results = train(resnet, "resnet50", train_loader, val_loader, task_config.NUM_CLASSES)
    
    print("Starting training for ViT-B/16 ...")
    vit_results = train(vit, "vit_b_16", train_loader, val_loader, task_config.NUM_CLASSES)
    
    print("Starting training for CLIP ViT-B/32 ...")
    clip_results = train(clip, "clip_vit_b_32", train_loader, val_loader, task_config.NUM_CLASSES)
    
    print("All models trained and evaluated.")
    
if __name__ == "__main__":
    main()
