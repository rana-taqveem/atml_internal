

import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader

def get_data_dir(task_name: str):
    # Colab sets this environment variable in its runtime.
    running_in_colab = bool(os.environ.get("COLAB_RELEASE_TAG"))

    if running_in_colab:
        from google.colab import drive

        mount_point = "/content/drive"

        # Reuse an existing mount.
        if not os.path.isdir(f"{mount_point}/MyDrive"):
            drive.mount(mount_point)

        base_dir = (
            Path(mount_point)
            / "MyDrive"
            / "ATML"
            / "assignment_01"
            / task_name
        )
    else:
        # Local runs never attempt to connect to Google Drive.
        base_dir = (
            Path(__file__).resolve().parent
            / "local_runs"
            / task_name
        )

    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir)

def is_running_in_colab():
    # Colab sets this environment variable in its runtime.
    return bool(os.environ.get("COLAB_RELEASE_TAG"))

def get_conflict_dataset_dir(task_dir):

    if is_running_in_colab():
        return os.path.join(task_dir, "conflict_dataset")

    return str(Path(__file__).resolve().parent / "data" / "conflict_dataset")

def save_checkpoint(state_dict, name):
    path = f'{DATA_DIR}/checkpoints/{name}.pt'
    torch.save(state_dict, path)
    print(f"saved -> {path}")

def load_checkpoint(name, map_location=None):
    path = f'{DATA_DIR}/checkpoints/{name}.pt'
    return torch.load(path, map_location=map_location or DEVICE) if os.path.exists(path) else None
