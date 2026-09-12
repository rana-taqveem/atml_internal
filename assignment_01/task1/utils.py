

import os
import torch
from torch.utils.data import DataLoader

def get_data_dir(task_name: str):
    if os.path.exists('/content/drive/MyDrive'):
        print("Detected an already mounted Google Drive workspace.")
        base_dir = f"/content/drive/MyDrive/assignment_01/{task_name}"
    else:
        try:
            from google.colab import drive
            print("Google Drive not detected. Attempting to mount...")
            drive.mount('/content/drive', force_remount=True)
            base_dir = f"/content/drive/MyDrive/assignment_01/{task_name}"
        except Exception as e:
            print("Google Drive mounting failed or non-interactive. Using local storage path.")
            base_dir = f"/content/assignment_01/{task_name}"
        
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def save_checkpoint(state_dict, name):
    path = f'{DATA_DIR}/checkpoints/{name}.pt'
    torch.save(state_dict, path)
    print(f"saved -> {path}")

def load_checkpoint(name, map_location=None):
    path = f'{DATA_DIR}/checkpoints/{name}.pt'
    return torch.load(path, map_location=map_location or DEVICE) if os.path.exists(path) else None