import argparse
import hashlib
import json
import os
import platform
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
from assignment_01.task1.data.download import prepare_stl10, prepare_conflict_dataset
from assignment_01.task1.config import task_config
from assignment_01.task1.data.transforms import apply_universal_transforms
from assignment_01.task1.models.backbones import (
    Resnet50Backbone,
    Torchvision_Vit_B_16_Backbone,
    Openai_Clip_Backbone,
    LinearClassifier
)
from assignment_01.task1.data.translation_dataset import (
    TranslationDataset,
    get_translation_conditions,
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

  y_true = []
  y_pred = []
  all_image_ids = []
  all_features = []
  all_logits = []

  # One pass collects features and logits; loss and accuracy come from the
  # same logits, so the backbone does not run twice per condition.
  model.eval()
  with torch.no_grad():
    for images, labels, image_ids in test_loader:
      images = images.to(DEVICE).float()

      features = model[0](images)  # Extract features using the backbone
      outputs = model[1](features)  # Classify using the classifier head

      _, predictions = torch.max(outputs.data, 1)

      y_true.extend(labels.cpu().tolist())
      y_pred.extend(predictions.cpu().tolist())

      all_image_ids.append(image_ids.cpu())
      all_features.append(features.cpu())
      all_logits.append(outputs.cpu())

  if not all_logits:
    raise ValueError("No images were evaluated. Please check the data loader.")

  label_tensor = torch.as_tensor(y_true, dtype=torch.long)
  logit_tensor = torch.cat(all_logits, dim=0).float()
  test_loss = criterion(logit_tensor, label_tensor).item()
  test_acc = 100 * (logit_tensor.argmax(dim=1) == label_tensor).float().mean().item()

  print("-" * 30)
  print(f"{results['model_name']} | {results['transformation']}")
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
        images = batch["image"].to(DEVICE).float()

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
    
    checkpoint_path = get_head_checkpoint_path(model_name)

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
       
def run_condition_inference(loader, metadata, models, text_features, logit_scale, prompts):
    """Evaluate every trained head and zero-shot CLIP on one image condition.

    Each method writes {model_name}_{transformation}_results.pt. Zero-shot
    CLIP reuses the CLIP head run's image embeddings, so both CLIP decision
    methods see identical features.
    """
    clip_result = None

    for model_name, model in models:
        print(f"Starting inference on {metadata['transformations']} with {model_name} ...")
        result = initialize_results(model_name, metadata)
        result = test_and_report(model, loader, nn.CrossEntropyLoss(), result)

        if model_name == task_config.CLIP_VIT_B_32:
            clip_result = result

    if clip_result is None:
        raise RuntimeError(f"CLIP features are missing for {metadata['transformations']}.")

    zero_shot_result = make_clip_zero_shot_predictions(clip_result, text_features, logit_scale, prompts)
    results_file = f"{zero_shot_result['model_name']}_{zero_shot_result['transformation']}_results.pt"
    torch.save(zero_shot_result, os.path.join(task_config.TASK_RESULTS_DIR, results_file))
    print(f"zero-shot accuracy ({metadata['transformations']}): "
          f"{zero_shot_result['accuracy_numbers']['accuracy']:.2f}%")

def run_translation_inference(
    baseline_dataset, baseline_metadata, models,
    text_features, logit_scale, prompts,
):
    """Twelve translated conditions: 8/16/32 px in four cardinal directions.

    Zero displacement is the baseline run itself (apply_translation with a
    zero offset returns the input crop unchanged), so it is not re-evaluated.
    """
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

        run_condition_inference(loader, metadata, models, text_features, logit_scale, prompts)

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
INFERENCE_STEPS = ["baseline", "color", "translation", "patch", "conflict"]

def main():

    parser = argparse.ArgumentParser(description="Train and Evaluate Models on STL-10 Dataset")
    parser.add_argument("--mode", choices=["train", "infer", "dry-run"], required=True, help="Mode: 'train' to train models, 'infer' to run inference or 'dry-run' to check setup without training or inference")
    parser.add_argument("--steps", nargs="+", choices=INFERENCE_STEPS, default=INFERENCE_STEPS,
                        help="Inference steps to run (default: all). Translation reuses the baseline subset.")
    parser.add_argument("--conflict-dir", default=None,
                        help="Finalized cue-conflict folder containing manifest.json "
                             "(default: task_config.TASK_CONFLICT_DATASET_DIR).")
    parser.add_argument("--skip-head-check", action="store_true",
                        help="Skip re-checking each head's validation accuracy before inference.")
    args = parser.parse_args()

    if args.mode == "train":
        print("Starting training...")
        start_training()
    elif args.mode == "infer":
        print("Starting inference...")
        start_inference(steps=args.steps, conflict_dir=args.conflict_dir,
                        check_heads=not args.skip_head_check)
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


def start_inference(steps=INFERENCE_STEPS, conflict_dir=None, check_heads=True):
    task_config.init_env()
    steps = list(steps)

    resnet = get_classifier_head(Resnet50Backbone(), model_name=task_config.RESNET50)
    vit = get_classifier_head(Torchvision_Vit_B_16_Backbone(), model_name=task_config.VIT_B_16)
    clip = get_classifier_head(Openai_Clip_Backbone(), model_name=task_config.CLIP_VIT_B_32)

    models = [
        (task_config.RESNET50, resnet),
        (task_config.VIT_B_16, vit),
        (task_config.CLIP_VIT_B_32, clip),
    ]

    head_check = verify_heads_on_validation(models) if check_heads else None

    text_features, logit_scale, prompts = prepare_clip_text_features(clip[0])
    print("Text feature shape:", text_features.shape)
    print("Logit scale:", logit_scale)
    print("Prompts:", prompts)

    save_run_manifest(steps, conflict_dir, logit_scale, prompts, head_check)

    def make_loader(dataset):
        return DataLoader(dataset, batch_size=task_config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    org_testset, org_metadata = None, None
    if {"baseline", "translation"} & set(steps):
        org_testset, org_metadata = get_test_subset(transformation_type='baseline')

    ### baseline images
    if "baseline" in steps:
        run_condition_inference(make_loader(org_testset), org_metadata, models,
                                text_features, logit_scale, prompts)

    ### grey scale and hue-rotated images
    if "color" in steps:
        gray_testset, gray_metadata = get_test_subset(transformation_type='grey_scale')
        run_condition_inference(make_loader(gray_testset), gray_metadata, models,
                                text_features, logit_scale, prompts)
        del gray_testset

        hue_testset, hue_metadata = get_test_subset(transformation_type='hue',
                                                    hue_rotation_angle=task_config.HUE_ROTATION_ANGLE)
        run_condition_inference(make_loader(hue_testset), hue_metadata, models,
                                text_features, logit_scale, prompts)
        del hue_testset

    ### translated images (12 conditions; 0 px is the baseline)
    if "translation" in steps:
        run_translation_inference(
            baseline_dataset=org_testset,
            baseline_metadata=org_metadata,
            models=models,
            text_features=text_features,
            logit_scale=logit_scale,
            prompts=prompts,
        )

    ### patch-shuffled images
    if "patch" in steps:
        patch_testset, patch_metadata = get_test_subset(transformation_type='patch_shuffle')
        run_condition_inference(make_loader(patch_testset), patch_metadata, models,
                                text_features, logit_scale, prompts)
        del patch_testset

    ### cue-conflict images
    if "conflict" in steps:
        run_conflict_inference(conflict_dir, models, text_features, logit_scale, prompts)

    print("All models evaluated.")

def run_conflict_inference(conflict_dir, models, text_features, logit_scale, prompts):

    conflict_dir = prepare_conflict_dataset(conflict_dir or task_config.TASK_CONFLICT_DATASET_DIR)
    conflict_dateset = ConflictDataset(conflict_dir)
    print(f"Loaded {len(conflict_dateset)} selected conflicts from {conflict_dateset.dataset_dir}")

    conflict_loader = DataLoader(conflict_dateset, batch_size=task_config.BATCH_SIZE, shuffle=False, num_workers=0)

    clip_conflict_result = None

    for model_name, model in models:

        result = infer_cue_conflicts(model, conflict_loader, model_name)

        result["metadata"] = {
            "dataset_dir": str(conflict_dateset.dataset_dir.resolve()),
            "manifest_path": str(
                (conflict_dateset.dataset_dir / "manifest.json").resolve()
            ),
            "n_conflicts": len(conflict_dateset),
            "alphas": sorted({float(record["alpha"]) for record in conflict_dateset.records}),
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

def get_head_checkpoint_path(model_name):
    """Locate one trained head's weight file.

    Checks task_config.MODEL_WEIGHTS_DIR first (where --mode train saves new
    heads, on Drive on Colab), then falls back to the copies already
    committed in the repo (task_config.PRETRAINED_HEADS_DIR). Returns the
    Drive path if neither exists, so the caller's FileNotFoundError names
    the primary expected location.
    """
    filename = task_config.HEAD_WEIGHT_FILES[model_name]
    drive_path = os.path.join(task_config.MODEL_WEIGHTS_DIR, filename)
    if os.path.isfile(drive_path):
        return drive_path

    repo_path = os.path.join(task_config.PRETRAINED_HEADS_DIR, filename)
    if os.path.isfile(repo_path):
        return repo_path

    return drive_path

def expected_validation_accuracy(model_name):
    """Validation accuracy recorded in the head filename, e.g. resnet50_97.30.pth."""
    stem = os.path.splitext(task_config.HEAD_WEIGHT_FILES[model_name])[0]
    return float(stem.rsplit("_", 1)[1])

def verify_heads_on_validation(models, tolerance_pp=0.2):
    """Re-evaluate backbone + head on the seed-6304 validation split.

    This confirms that the loaded head file matches the current backbone
    before any test-time evidence is produced. It fails when a head's
    accuracy differs from the value in its filename by more than the tolerance.
    """
    _, val_loader = get_train_val_dataloaders()
    report = {}

    for model_name, model in models:
        _, val_acc = evaluate(model, val_loader, nn.CrossEntropyLoss())
        expected = expected_validation_accuracy(model_name)
        report[model_name] = {"validation_accuracy_pct": val_acc, "expected_pct": expected}
        print(f"Head check {model_name}: val acc {val_acc:.2f}% (expected {expected:.2f}%)")

        if abs(val_acc - expected) > tolerance_pp:
            raise RuntimeError(
                f"{model_name}: validation accuracy {val_acc:.2f}% does not match the "
                f"head file ({expected:.2f}%). Check HEAD_WEIGHT_FILES and the backbone."
            )
    return report

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def save_run_manifest(steps, conflict_dir, logit_scale, prompts, head_check):
    """Record the settings and software behind this inference run."""
    import sklearn
    import torchvision

    manifest = {
        "steps": list(steps),
        "seed": task_config.SEED,
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "classes": list(task_config.STL10_CLASSES),
        "test_subset": {
            "selected_indices_file": task_config.SELECTED_INDICES_FILE,
            "total_target_images": task_config.TOTAL_TARGET_IMAGES,
        },
        "heads": {
            model_name: {
                "path": get_head_checkpoint_path(model_name),
                "sha256": file_sha256(get_head_checkpoint_path(model_name)),
                "expected_validation_accuracy_pct": expected_validation_accuracy(model_name),
            }
            for model_name in task_config.HEAD_WEIGHT_FILES
        },
        "head_validation_check": head_check,
        "backbones": {
            task_config.RESNET50: "torchvision resnet50 ResNet50_Weights.IMAGENET1K_V2, global-average-pooled feature",
            task_config.VIT_B_16: "torchvision vit_b_16 ViT_B_16_Weights.IMAGENET1K_V1, final class token",
            task_config.CLIP_VIT_B_32: "open_clip ViT-B-32 pretrained=openai (quick_gelu), L2-normalized image embedding",
        },
        "interventions": {
            "input": "224x224 RGB in [0, 1]; model normalization applied once inside each backbone",
            "grey_scale": "ITU-R 601-2 luma replicated to 3 channels (torchvision Grayscale)",
            "hue": {"rotation_degrees": task_config.HUE_ROTATION_ANGLE, "implementation": "torchvision adjust_hue"},
            "translation": {"conditions": get_translation_conditions(), "padding": "reflect, 32 px, shifted crop"},
            "patch_shuffle": {"grid": "4x4", "patch_size": task_config.PATCH_SIZE, "seed": task_config.SEED,
                              "identity_permutation": "rejected and redrawn"},
            "cue_conflict_dir": conflict_dir or task_config.TASK_CONFLICT_DATASET_DIR,
        },
        "zero_shot": {"prompts": list(prompts), "logit_scale": float(logit_scale)},
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "open_clip": open_clip.__version__,
            "sklearn": sklearn.__version__,
        },
    }

    path = os.path.join(task_config.TASK_RESULTS_DIR, "run_manifest.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(f"Saved run manifest: {path}")

def initialize_results(model_name, metadata):

    checkpoint_path = get_head_checkpoint_path(model_name)

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

    checkpoint_dir = get_head_checkpoint_path(model_name)

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
