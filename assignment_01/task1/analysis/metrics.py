import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calcultate_metrics(result):
    
    labels = torch.as_tensor(result["y_true"], dtype=torch.long, device="cpu")
    
    logits = torch.as_tensor(result["logits"], dtype=torch.float32, device="cpu")
    
    if logits.ndim != 2 or labels.ndim != 1:
        raise ValueError("Logits must be a 2D tensor and labels must be a 1D tensor.")
    
    if len(labels) == 0 or logits.shape[0] != len(labels):
        raise ValueError("Labels must not be empty and must match the number of logits.")
    
    if not torch.isfinite(logits).all() or not torch.isfinite(labels).all():
        raise ValueError("Logits and labels must not contain NaN or Inf values.")
    
    num_classes = logits.shape[1]
    
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError(f"Labels must be in the range [0, {num_classes - 1}].")
    
    probabilities = torch.softmax(logits, dim=1)
    
    confidence, predictions = torch.max(probabilities, dim=1)
    
    accuracy = (predictions == labels).float().mean().item()
    
    macro_f1 = f1_score(labels.cpu().numpy(), 
                        predictions.cpu().numpy(), 
                        average='macro', 
                        zero_division=0,
                        labels=list(range(num_classes)))
    
    return{
         "n_images": len(labels),
        "accuracy_pct": 100 * accuracy,
        "macro_f1": float(macro_f1),
        "mean_max_confidence": confidence.mean().item(),
    }