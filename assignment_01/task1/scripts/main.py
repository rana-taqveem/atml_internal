import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.manifold import TSNE
import seaborn as sns

import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
from sklearn.metrics.pairwise import cosine_similarity

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {DEVICE}")



TRAIN_LOSS = 'train_loss'
TRAIN_ACC = 'train_acc'
VAL_LOSS = 'val_loss'
VAL_ACC = 'val_acc'

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(model, loader, criterion, optimizer, scaler=None):
    model.train()

    total_loss = 0.0
    accurate_predictions = 0
    sample_count = 0
    for images, labels in loader:
      optimizer.zero_grad()
      images = images.to(DEVICE).float()
      labels = labels.to(DEVICE)

      if scaler is not None:
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
          outputs = model(images)
          loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
      else:
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
        outputs = model(images)

        labels = labels.to(DEVICE)
        loss = criterion(outputs, labels)
        _, predicted_classes = torch.max(outputs.data, 1)
        accurate_predictions += (predicted_classes == labels).sum().item()

        total_loss += loss.item() * images.size(0)
        sample_count += images.size(0)

    avg_loss = total_loss / sample_count
    accuracy = 100 * accurate_predictions / sample_count

    return avg_loss, accuracy

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=7, scaler=None):
  results = {
      TRAIN_LOSS: [],
      TRAIN_ACC: [],
      VAL_LOSS: [],
      VAL_ACC : []
  }

  for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler=scaler)
    results[TRAIN_LOSS].append(train_loss)
    results[TRAIN_ACC].append(train_acc)

    val_loss, val_acc = evaluate(model, val_loader, criterion=criterion)
    results[VAL_LOSS].append(val_loss)
    results[VAL_ACC].append(val_acc)

    print(f"Epoch: {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss {val_loss:.2f}% Val Acc:{val_acc:.2f}%")

  return results

def test_and_report(model, test_loader, criterion, classes):

  test_loss, test_acc = evaluate(model, test_loader, criterion)

  y_true = []
  y_pred = []

  model.eval()
  with torch.no_grad():
    for images, labels in test_loader:
      images = images.to(DEVICE)
      labels = labels.to(DEVICE)

      outputs = model(images)
      _, predictions = torch.max(outputs.data, 1)

      y_true.extend(labels.cpu().tolist())
      y_pred.extend(predictions.cpu().tolist())

  print("-" * 30)
  print(f"FINAL TEST RESULT")
  print(f"Test Loss: {test_loss:.4f}")
  print(f"Test Accuracy: {test_acc:.2f}%")
  print("-" * 30)

  classes_list = [str(c) for c in classes]

  cr = classification_report(y_true, y_pred, target_names=classes_list, zero_division=0)
  print("\nClassification Report:\n", cr)

  return test_loss, test_acc, y_true, y_pred

def plot_loss_acc(results):
  plt.figure(figsize=(6,4))
  plt.plot(results[TRAIN_LOSS], label='Training Loss')
  plt.plot(results[VAL_LOSS], label='Validation Loss')
  plt.legend()

  plt.figure(figsize=(6,4))
  plt.plot(results[TRAIN_ACC], label='Training Accuracy')
  plt.plot(results[VAL_ACC], label='Validation Accuracy')
  plt.legend()

def extract_features(model, loader, cache_path=None):

    if cache_path and os.path.exists(cache_path):
        data = torch.load(cache_path)
        return data['X'], data['y']

    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for images, y in tqdm(loader, desc="extracting features"):

            images = images.to(DEVICE)
            x = model.conv1(images)
            x = model.maxpool(model.relu(model.bn1(x)))

            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x).flatten(1)

            feats.append(x.cpu().half())
            labels.append(y)

    X = torch.cat(feats)
    y_all = torch.cat(labels)

    if cache_path:
        torch.save({'X': X, 'y': y_all}, cache_path)

    return X, y_all

def train_head_on_features(X_train, y_train, X_val, y_val, num_classes, num_epochs=NUM_EPOCHS, lr=1e-3, batch_size=256, normalize_features=False):

    in_features = X_train.shape[1]
    if normalize_features:
        head = nn.Sequential(nn.BatchNorm1d(in_features), nn.Linear(in_features, num_classes)).to(DEVICE)
    else:
        head = nn.Linear(in_features, num_classes).to(DEVICE)

    optimizer = optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader_f = DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader_f = DataLoader(torch.utils.data.TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    results = train_model(head, train_loader_f, val_loader_f, criterion, optimizer, num_epochs=num_epochs)

    return head, results

def extract_stage_features(model, loader, cache_path=None):

    if cache_path and os.path.exists(cache_path):
        data = torch.load(cache_path)
        return data['X'], data['y']

    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for images, y in tqdm(loader, desc="extracting layer3 features"):
            images = images.to(DEVICE)

            x = model.conv1(images)
            x = model.maxpool(model.relu(model.bn1(x)))
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)

            feats.append(x.cpu().half())
            labels.append(y)

    X = torch.cat(feats)
    y_all = torch.cat(labels)

    if cache_path:
        torch.save({'X': X, 'y': y_all}, cache_path)

    return X, y_all

def build_final_block_trainer(pretrained, normalize_input=None):

    if normalize_input is None:
        normalize_input = not pretrained

    weights = models.ResNet152_Weights.DEFAULT if pretrained else None
    base = models.resnet152(weights=weights).to(DEVICE)

    for p in base.parameters():
        p.requires_grad = False

    base.eval()

    if normalize_input:
        in_channels = base.layer3[-1].bn3.num_features
        base.layer4 = nn.Sequential(nn.BatchNorm2d(in_channels), base.layer4).to(DEVICE)

    fc = nn.Linear(base.fc.in_features, NUM_CLASSES).to(DEVICE)
    head_model = FinalBlockHead(base.layer4, base.avgpool, fc).to(DEVICE)

    for p in head_model.parameters():
        p.requires_grad = True

    return base, head_model

def collect_multistage_features(model, loader, max_samples=1000):

    taps = {}

    def make_hook(name):
        def hook(module, inp, out):
            taps[name] = out
        return hook

    handles = [
        model.layer1.register_forward_hook(make_hook("early")),
        model.layer3.register_forward_hook(make_hook("middle")),
        model.avgpool.register_forward_hook(make_hook("late")),
    ]

    early_feats, middle_feats, late_feats = [], [], []
    all_labels, all_preds = [], []
    seen = 0

    model.eval()
    with torch.no_grad():
        for images, labels in loader:

            if seen >= max_samples:
                break

            images = images.to(DEVICE)
            outputs = model(images)

            preds = outputs.argmax(dim=1)

            early_feats.append(taps["early"].mean(dim=[2, 3]).cpu())    # (batch, 256)
            middle_feats.append(taps["middle"].mean(dim=[2, 3]).cpu())  # (batch, 1024)
            late_feats.append(taps["late"].flatten(1).cpu())            # (batch, 2048)

            all_labels.append(labels)
            all_preds.append(preds.cpu())
            seen += images.size(0)

    for h in handles:
        h.remove()

    return {
        "early": torch.cat(early_feats)[:max_samples],
        "middle": torch.cat(middle_feats)[:max_samples],
        "late": torch.cat(late_feats)[:max_samples],
        "labels": torch.cat(all_labels)[:max_samples],
        "preds": torch.cat(all_preds)[:max_samples],
    }

def plot_class_scatter(ax, coords_2d, labels, classes, title):

    labels_np = labels.numpy() if torch.is_tensor(labels) else labels
    for class_idx, class_name in enumerate(classes):
        mask = labels_np == class_idx
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], s=8, alpha=0.6, label=class_name)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

