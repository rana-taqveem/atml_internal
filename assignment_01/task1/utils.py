

import os
import torch
from torch.utils.data import DataLoader

DATA_DIR = '/content/drive/MyDrive/ATML/assignment_01/'

def get_data_dir(task_name: str):
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        DATA_DIR = f'/content/drive/MyDrive/ATML/assignment_01/{task_name}'
    except ImportError:
        DATA_DIR = f'./{task_name}'  # not on Colab

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(f'{DATA_DIR}/checkpoints', exist_ok=True)
    os.makedirs(f'{DATA_DIR}/features', exist_ok=True)
    os.makedirs(f'{DATA_DIR}/dataset', exist_ok=True)

    TORCH_HOME = f'{DATA_DIR}/torch_cache'
    os.environ['TORCH_HOME'] = TORCH_HOME

    print(f"DATA_DIR={DATA_DIR}")
    
    return DATA_DIR

def save_checkpoint(state_dict, name):
    path = f'{DATA_DIR}/checkpoints/{name}.pt'
    torch.save(state_dict, path)
    print(f"saved -> {path}")

def load_checkpoint(name, map_location=None):
    path = f'{DATA_DIR}/checkpoints/{name}.pt'
    return torch.load(path, map_location=map_location or DEVICE) if os.path.exists(path) else None