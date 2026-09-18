import argparse
import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import Subset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from assignment_01.task1.data.conflict_dataset import ConflictDataset
from assignment_01.task1.data.make_subset import get_test_subset
from assignment_01.task1.data.download import prepare_stl10
from assignment_01.task1.config import task_config
from assignment_01.task1.data.transforms import apply_universal_transforms
from assignment_01.task1.data.conflict_dataset import ConflictDataset
from assignment_01.task1.models.backbones import (
    Resnet50Backbone, 
    Torchvision_Vit_B_16_Backbone, 
    Openai_Clip_Backbone,
    LinearClassifier
)
from assignment_01.task1.data.translation_dataset import (
    TranslationDataset,
    get_translation_conditions,
    get_translation_dataset,
)

import open_clip


# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {DEVICE}")

def get_train_val_dataloaders():
    prepare_stl10(task_config.TASK_DATASET_DIR)
    train_ds_full = datasets.STL10(root=task_config.TASK_DATASET_DIR, 
                                   split='train', 
                                   download=False,
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
      for batch in loader:
        images, labels = batch[:2]  # Unpack only the first two elements (images and labels)
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
        best_weights = copy.deepcopy(model.state_dict())
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
  print(f"Saved optimized best classifier head baselinely to {task_config.MODEL_WEIGHTS_DIR}/{model_name}_{best_val_acc:.2f}.pth")
        
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

    torch.manual_seed(task_config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(task_config.SEED)
        
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

def test_and_report(model, test_loader, criterion, results):

  test_loss, test_acc = evaluate(model, test_loader, criterion)

  y_true = []
  y_pred = []
  all_image_ids = []
  all_features = []
  all_logits = []
  
  model.eval()
  with torch.no_grad():
    for images, labels, image_ids in test_loader:
      images = images.to(DEVICE)
      labels = labels.to(DEVICE)

      features = model[0](images)  # Extract features using the backbone
      outputs = model[1](features)  # Classify using the classifier head
  
      _, predictions = torch.max(outputs.data, 1)

      y_true.extend(labels.cpu().tolist())
      y_pred.extend(predictions.cpu().tolist())
      
      all_image_ids.append(image_ids.cpu())
      all_features.append(features.cpu())
      all_logits.append(outputs.cpu())

  print("-" * 30)
  print(f"FINAL TEST RESULT")
  print(f"Test Loss: {test_loss:.4f}")
  print(f"Test Accuracy: {test_acc:.2f}%")
  print("-" * 30)

#   classes_list = [str(c) for c in classes]

#   cr = classification_report(y_true, y_pred, target_names=classes_list, zero_division=0)
#   print("\nClassification Report:\n", cr)

  if not all_features:
      raise ValueError("No features were extracted. Please check the model and data loader.")
  
  image_ids = torch.cat(all_image_ids, dim=0)
  features = torch.cat(all_features, dim=0)
  logits = torch.cat(all_logits, dim=0)
  
  accuracy_numbers = {"accuracy": test_acc, "loss": test_loss}
  results.update({
      "accuracy_numbers": accuracy_numbers,
      "y_true": y_true,
      "y_pred": y_pred,
      "image_ids": image_ids,
      "features": features,
      "logits": logits,    
  })
  
  results_file = f"{results["model_name"]}_{results["transformation"]}_results.pt"
  torch.save(results, os.path.join(task_config.TASK_RESULTS_DIR, results_file))
  return results
    
def plot_loss_acc(results, TRAIN_LOSS='train_loss', TRAIN_ACC='train_acc', VAL_LOSS='val_loss', VAL_ACC='val_acc'):
  plt.figure(figsize=(6,4))
  plt.plot(results[TRAIN_LOSS], label='Training Loss')
  plt.plot(results[VAL_LOSS], label='Validation Loss')
  plt.legend()

  plt.figure(figsize=(6,4))
  plt.plot(results[TRAIN_ACC], label='Training Accuracy')
  plt.plot(results[VAL_ACC], label='Validation Accuracy')
  plt.legend()

def plot_class_scatter(ax, coords_2d, labels, classes, title):

    labels_np = labels.numpy() if torch.is_tensor(labels) else labels
    for class_idx, class_name in enumerate(classes):
        mask = labels_np == class_idx
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], s=8, alpha=0.6, label=class_name)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


@torch.no_grad()
def prepare_clip_text_features(clip_backbone):
    model = clip_backbone.model
    model.to(DEVICE)
    model.eval()

    device = next(model.parameters()).device
    
    prompts = [
        task_config.CLIP_PROMPT.format(class_name) 
        for class_name in task_config.STL10_CLASSES
    ]
    
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    tokens = tokenizer(prompts).to(device)
    
    
    text_features = model.encode_text(tokens).float()
    text_features = F.normalize(text_features, dim=-1)
    
    logit_scale = model.logit_scale.exp().item()

    return text_features.cpu(), logit_scale, prompts

@torch.no_grad()
def make_clip_zero_shot_predictions(clip_result, text_features, logit_scale, prompts):
    
    image_features = F.normalize(clip_result["features"].float().cpu(), dim=-1)
    text_features = F.normalize(text_features.float().cpu(), dim=-1)
    
    logits = logit_scale * (image_features @ text_features.T)
    
    probs = F.softmax(logits, dim=-1)
    confidence, predictions = torch.max(probs, dim=-1)
    
    labels = torch.as_tensor(clip_result["y_true"], dtype=torch.long, device="cpu")
    return {
        "model_name": "clip_vit_b_32_zero_shot",
        "backbone_name": task_config.CLIP_VIT_B_32,
        "pretrained": "openai",
        "head_checkpoint": None,
        "classes": list(clip_result["classes"]),
        "transformation": clip_result["transformation"],
        "metadata": copy.deepcopy(clip_result["metadata"]),
        "prompts": list(prompts),
        "logit_scale": float(logit_scale),

        "accuracy_numbers": {
            "accuracy": (
                100 * (predictions == labels).float().mean().item()
            ),
            "loss": F.cross_entropy(logits, labels).item(),
        },

        "image_ids": clip_result["image_ids"].clone(),
        "y_true": labels.tolist(),
        "y_pred": predictions.tolist(),
        "features": image_features,
        "logits": logits,
        "confidence": confidence,
    }

@torch.no_grad()
def infer_cue_conflicts(model, loader, model_name):
    model.eval()
    
    feature_batches = []
    logit_batches = []
    conflict_ids = []
    
    collected = {
        "content_id": [],
        "style_id": [],
        "content_label": [],
        "style_label": [],
    }
    
    for batch in loader:
        images = batch["image"].to(DEVICE)
        
        features = model[0](images)
        logits = model[1](features)
        
        feature_batches.append(features.cpu())
        logit_batches.append(logits.cpu())
        conflict_ids.extend(batch["conflict_id"])
        
        for key in collected.keys():
            collected[key].append(batch[key].cpu())
            
    if not conflict_ids:
        raise ValueError("No conflict IDs were collected. Please check the data loader and model.")
    
    if len(set(conflict_ids)) != len(conflict_ids):
        raise ValueError("Duplicate conflict IDs found in the collected data.")
    
    features = torch.cat(feature_batches, dim=0)
    logits = torch.cat(logit_batches, dim=0)
    
    combined = { key: torch.cat(batches, dim=0) 
                for key, batches in collected.items()
            }
    
    checkpoint_path = os.path.join(task_config.TASK_CHECKPOINTS_DIR, f"{model_name}_head.pth")
    
    return {
        "model_name": model_name,
        "head_checkpoint": os.path.abspath(checkpoint_path),
        "classes": list(task_config.STL10_CLASSES),
        "transformation": "cue_conflict",
        "conflict_ids": conflict_ids,
        "content_ids": combined["content_id"],
        "style_ids": combined["style_id"],
        "content_labels": combined["content_label"],
        "style_labels": combined["style_label"],
        "features": features,
        "logits": logits,
        "y_pred": logits.argmax(dim=1).tolist(),
    }
       
def run_translation_inference(
    baseline_dataset, baseline_metadata, models,
    text_features, logit_scale, prompts,
):
    for condition in get_translation_conditions():
        dataset = TranslationDataset(
            baseline_dataset=baseline_dataset,
            translate_x=condition["translation_x"],
            translate_y=condition["translation_y"],
        )

        loader = DataLoader(
            dataset,
            batch_size=task_config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        metadata = {
            **baseline_metadata,
            **condition,
            "transformations": condition["condition_id"],
        }

        clip_result = None

        for model_name, model in models:
            result = initialize_results(model_name, metadata)
            result = test_and_report(
                model, loader, nn.CrossEntropyLoss(), result
            )

            if model_name == task_config.CLIP_VIT_B_32:
                clip_result = result

        if clip_result is None:
            raise RuntimeError("CLIP translation features are missing.")

        zero_shot_result = make_clip_zero_shot_predictions(
            clip_result, text_features, logit_scale, prompts
        )

        filename = (
            f"{zero_shot_result['model_name']}_"
            f"{condition['condition_id']}_results.pt"
        )

        torch.save(
            zero_shot_result,
            os.path.join(task_config.TASK_RESULTS_DIR, filename),
        )
        
@torch.no_grad()
def make_clip_conflict_zero_shot(clip_result, text_features, logit_scale, prompts):
    image_features = F.normalize(clip_result["features"].float().cpu(), dim=1)
    text_features = F.normalize(text_features.float().cpu(), dim=1)
    
    logits = logit_scale * (image_features @ text_features.T)
    
    probabilities  = F.softmax(logits, dim=-1)
    confidence, predictions = torch.max(probabilities , dim=-1)
    
    result = copy.deepcopy(clip_result)
    
    result.update({
        "model_name": f"{task_config.CLIP_VIT_B_32}_zero_shot",
        "backbone_name": task_config.CLIP_VIT_B_32,
        "pretrained": "openai",
        "head_checkpoint": None,
        "prompts": list(prompts),
        "logit_scale": float(logit_scale),
        "features": image_features,
        "logits": logits,
        "confidence": confidence,
        "y_pred": predictions.tolist(),
    })

    return result
def main():
    
    parser = argparse.ArgumentParser(description="Train and Evaluate Models on STL-10 Dataset")
    parser.add_argument("--mode", choices=["train", "infer", "dry-run"], required=True, help="Mode: 'train' to train models, 'infer' to run inference or 'dry-run' to check setup without training or inference")
    args = parser.parse_args()

    if args.mode == "train":
        print("Starting training...")
        start_training()
    elif args.mode == "infer":
        print("Starting inference...")
        start_inference()
    elif args.mode == "dry-run":
            print("All set! dry run passed. You can now run the script with --mode train or --mode infer to proceed.")
    else:
        print("Please specify a mode: --train or --infer")
        parser.print_help()

def start_training():
    task_config.init_env()
    train_loader, val_loader = get_train_val_dataloaders()
    
    resnet = Resnet50Backbone()
    vit = Torchvision_Vit_B_16_Backbone()
    clip = Openai_Clip_Backbone()
    
    print("Starting training for ResNet50 ...")
    resnet_results = train(resnet, task_config.RESNET50, train_loader, val_loader, task_config.NUM_CLASSES)
    
    print("Starting training for ViT-B/16 ...")
    vit_results = train(vit, task_config.VIT_B_16, train_loader, val_loader, task_config.NUM_CLASSES)
    
    print("Starting training for CLIP ViT-B/32 ...")
    clip_results = train(clip, task_config.CLIP_VIT_B_32, train_loader, val_loader, task_config.NUM_CLASSES)
    
    print("All models trained and evaluated.")
    

def start_inference():
    task_config.init_env()

    org_testset, org_metadata     = get_test_subset(transformation_type='baseline')
    gray_testset, gray_metadata   = get_test_subset(transformation_type='grey_scale')
    hue_testset, hue_metadata     = get_test_subset(transformation_type='hue', hue_rotation_angle=30)
    # trans_testset, trans_metadata = get_test_subset(transformation_type='translation', translation_x=8, translation_y=0)
    patch_testset, patch_metadata = get_test_subset(transformation_type='patch_shuffle')
    
    orginal_loader = DataLoader(org_testset,   batch_size=task_config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    grey_loader    = DataLoader(gray_testset,  batch_size=task_config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    hue_loader     = DataLoader(hue_testset,   batch_size=task_config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    # trans_loader   = DataLoader(trans_testset, batch_size=task_config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    patch_loader   = DataLoader(patch_testset, batch_size=task_config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    resnet = get_classifier_head(Resnet50Backbone(), model_name=task_config.RESNET50)
    vit = get_classifier_head(Torchvision_Vit_B_16_Backbone(), model_name=task_config.VIT_B_16)
    clip = get_classifier_head(Openai_Clip_Backbone(), model_name=task_config.CLIP_VIT_B_32)

    text_features, logit_scale, prompts = prepare_clip_text_features(clip[0])
    print("Text feature shape:", text_features.shape)
    print("Logit scale:", logit_scale)
    print("Prompts:", prompts)
    
    ### baseline images

    print("Starting inference on baseline with ResNet50 ...")
    resnet_org_result = initialize_results(task_config.RESNET50, org_metadata)
    test_and_report(resnet, orginal_loader, nn.CrossEntropyLoss(), resnet_org_result)
    
    print("Starting inference on baseline with ViT-B/16 ...")
    vit_org_result = initialize_results(task_config.VIT_B_16, org_metadata)
    test_and_report(vit, orginal_loader, nn.CrossEntropyLoss(), vit_org_result)

    print("Starting inference on baseline with CLIP ViT-B/32 ...")
    clip_org_result = initialize_results(task_config.CLIP_VIT_B_32, org_metadata)
    clip_org_result =  test_and_report(clip, orginal_loader, nn.CrossEntropyLoss(), clip_org_result)
    
    clip_zero_shot_result = make_clip_zero_shot_predictions(clip_org_result, text_features, logit_scale, prompts)
    results_file = f"{clip_zero_shot_result["model_name"]}_{clip_zero_shot_result["transformation"]}_results.pt"
    torch.save(clip_zero_shot_result, os.path.join(task_config.TASK_RESULTS_DIR, results_file))
    
    ### grey scale images
    print("Starting inference on grayscale images with ResNet50 ...")
    resnet_gray_result = initialize_results(task_config.RESNET50, gray_metadata)
    resnet_gray_result = test_and_report(resnet, grey_loader, nn.CrossEntropyLoss(), resnet_gray_result)
    
    print("Starting inference on grayscale images with ViT-B/16 ...")
    vit_gray_result = initialize_results(task_config.VIT_B_16, gray_metadata)
    vit_gray_result = test_and_report(vit, grey_loader, nn.CrossEntropyLoss(), vit_gray_result)

    print("Starting inference on grayscale images with CLIP ViT-B/32 ...")
    clip_gray_result = initialize_results(task_config.CLIP_VIT_B_32, gray_metadata)
    clip_gray_result = test_and_report(clip, grey_loader, nn.CrossEntropyLoss(), clip_gray_result)

    clip_zero_shot_result = make_clip_zero_shot_predictions(clip_gray_result, text_features, logit_scale, prompts)
    results_file = f"{clip_zero_shot_result["model_name"]}_{clip_zero_shot_result["transformation"]}_results.pt"
    torch.save(clip_zero_shot_result, os.path.join(task_config.TASK_RESULTS_DIR, results_file))
    
    
    ### hue-rotated images
    print("Starting inference on hue-rotated images with ResNet50 ...")
    resnet_hue_result = initialize_results(task_config.RESNET50, hue_metadata)
    resnet_hue_result = test_and_report(resnet, hue_loader, nn.CrossEntropyLoss(), resnet_hue_result)
    
    print("Starting inference on hue-rotated images with ViT-B/16 ...")
    vit_hue_result = initialize_results(task_config.VIT_B_16, hue_metadata)
    vit_hue_result = test_and_report(vit, hue_loader, nn.CrossEntropyLoss(), vit_hue_result)

    print("Starting inference on hue-rotated images with CLIP ViT-B/32 ...")
    clip_hue_result = initialize_results(task_config.CLIP_VIT_B_32, hue_metadata)
    clip_hue_result = test_and_report(clip, hue_loader, nn.CrossEntropyLoss(), clip_hue_result)

    clip_zero_shot_result = make_clip_zero_shot_predictions(clip_hue_result, text_features, logit_scale, prompts)
    results_file = f"{clip_zero_shot_result["model_name"]}_{clip_zero_shot_result["transformation"]}_results.pt"
    torch.save(clip_zero_shot_result, os.path.join(task_config.TASK_RESULTS_DIR, results_file))
    
    
    ### translated images
    # print("Starting inference on translated images with ResNet50 ...")
    # resnet_trans_result = initialize_results(task_config.RESNET50, trans_metadata)
    # resnet_trans_result = test_and_report(resnet, trans_loader, nn.CrossEntropyLoss(), resnet_trans_result)

    # print("Starting inference on translated images with ViT-B/16 ...")
    # vit_trans_result = initialize_results(task_config.VIT_B_16, trans_metadata)
    # vit_trans_result = test_and_report(vit, trans_loader, nn.CrossEntropyLoss(), vit_trans_result)

    # print("Starting inference on translated images with CLIP ViT-B/32 ...")
    # clip_trans_result = initialize_results(task_config.CLIP_VIT_B_32, trans_metadata)
    # clip_trans_result = test_and_report(clip, trans_loader, nn.CrossEntropyLoss(), clip_trans_result)

    # clip_zero_shot_result = make_clip_zero_shot_predictions(clip_trans_result, text_features, logit_scale, prompts)
    # results_file = f"{clip_zero_shot_result["model_name"]}_{clip_zero_shot_result["transformation"]}_results.pt"
    # torch.save(clip_zero_shot_result, os.path.join(task_config.TASK_RESULTS_DIR, results_file))
    
    
    run_translation_inference(
        baseline_dataset=org_testset,
        baseline_metadata=org_metadata,
        models=[
            (task_config.RESNET50, resnet),
            (task_config.VIT_B_16, vit),
            (task_config.CLIP_VIT_B_32, clip),
        ],
        text_features=text_features,
        logit_scale=logit_scale,
        prompts=prompts,
    )
        
    ### patch-shuffled images
    print("Starting inference on patch-shuffled images with ResNet50 ...")
    resnet_patch_result = initialize_results(task_config.RESNET50, patch_metadata)
    resnet_patch_result = test_and_report(resnet, patch_loader, nn.CrossEntropyLoss(), resnet_patch_result)

    print("Starting inference on patch-shuffled images with ViT-B/16 ...")
    vit_patch_result = initialize_results(task_config.VIT_B_16, patch_metadata)
    vit_patch_result = test_and_report(vit, patch_loader, nn.CrossEntropyLoss(), vit_patch_result)

    print("Starting inference on patch-shuffled images with CLIP ViT-B/32 ...")
    clip_patch_result = initialize_results(task_config.CLIP_VIT_B_32, patch_metadata)
    clip_patch_result = test_and_report(clip, patch_loader, nn.CrossEntropyLoss(), clip_patch_result)

    clip_zero_shot_result = make_clip_zero_shot_predictions(clip_patch_result, text_features, logit_scale, prompts)
    results_file = f"{clip_zero_shot_result["model_name"]}_{clip_zero_shot_result["transformation"]}_results.pt"
    torch.save(clip_zero_shot_result, os.path.join(task_config.TASK_RESULTS_DIR, results_file))
    

    conflict_dateset = ConflictDataset(task_config.TASK_CONFLICTS_DIR)
    
    conflict_loader = DataLoader(conflict_dateset, batch_size=task_config.BATCH_SIZE, shuffle=False, num_workers=0)

    conflict_models = [
        (task_config.RESNET50, resnet),
        (task_config.VIT_B_16, vit),
        (task_config.CLIP_VIT_B_32, clip)
    ]
    
    clip_conflict_result = None
    
    for model_name, model in conflict_models:
        
        result = infer_cue_conflicts(model, conflict_loader, model_name)
        
        result["metadata"] = {
            "dataset_dir": str(conflict_dateset.dataset_dir.resolve()),
            "manifest_path": str(
                (conflict_dateset.dataset_dir / "manifest.json").resolve()
            ),
            "n_conflicts": len(conflict_dateset),
        }
        
        os.makedirs(task_config.TASK_RESULTS_DIR, exist_ok=True)

        result_path = os.path.join(
            task_config.TASK_RESULTS_DIR,
            f"{model_name}_cue_conflict_results.pt",
        )
        torch.save(result, result_path)

        if model_name == task_config.CLIP_VIT_B_32:
            clip_conflict_result = result

        print(f"Saved conflict inference: {result_path}")
        
    if clip_conflict_result is None:
        raise RuntimeError("CLIP conflict features were not collected.")
    
    zero_shot_result  = make_clip_conflict_zero_shot(clip_conflict_result, 
                                                    text_features, logit_scale, 
                                                    prompts)
    result_path = os.path.join(
        task_config.TASK_RESULTS_DIR,
        f"{zero_shot_result['model_name']}_cue_conflict_results.pt",
    )
    torch.save(zero_shot_result, result_path)
    print(f"Saved zero-shot conflict inference: {result_path}")
     
    print("All models evaluated.")

def initialize_results(model_name, metadata):
    
    checkpoint_path = os.path.join(task_config.TASK_CHECKPOINTS_DIR, f"{model_name}_head.pth")
    
    return {
        "model_name": model_name,
        "head_checkpoint": os.path.abspath(checkpoint_path),
        "classes": task_config.STL10_CLASSES,
        "transformation": metadata["transformations"],
        "metadata": copy.deepcopy(metadata),
        "accuracy_numbers": {
            "accuracy": None,
            "loss": None
          },
      "y_true": None,
      "y_pred": None,
      "image_ids": None,
      "features": None,
      "logits": None,  
    }


def get_classifier_head(backbone, model_name):
    
    checkpoint_dir = os.path.join(task_config.TASK_CHECKPOINTS_DIR, f"{model_name}_head.pth")
    
    state = torch.load(checkpoint_dir, map_location=DEVICE, weights_only=True)
    
    num_classes, in_features = state['weight'].shape
    
    head_resent = nn.Linear(in_features, num_classes).to(DEVICE)
    head_resent.load_state_dict(state)
    
    classifier_head = nn.Sequential(backbone, head_resent).to(DEVICE).eval()
    return classifier_head

def generate_cue_conflict_images():
    
    from assignment_01.task1.data.make_cue_conflicts import generate_cue_conflicts
    from assignment_01.task1.data.transforms import StyleTransferModel
    
    style_transfer_model = StyleTransferModel()
    
    style_transfer_model.load_state_dict(torch.load(task_config.STYLE_TRANSFER_MODEL_PATH, map_location=DEVICE))
    style_transfer_model.to(DEVICE).eval()
    generate_cue_conflicts()
    
if __name__ == "__main__":
    main()
