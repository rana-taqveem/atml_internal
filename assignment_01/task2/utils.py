

import os
from pathlib import Path


def is_running_in_kaggle():
    """Kaggle sets this in notebook and batch runtimes alike."""
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or os.path.isdir("/kaggle/working")


def get_data_dir(task_name: str):
    # Colab sets this environment variable in its runtime.
    running_in_colab = bool(os.environ.get("COLAB_RELEASE_TAG"))

    if is_running_in_kaggle():
        # /kaggle/working is the only writable location that survives to the
        # session's output; anything written elsewhere is lost when the kernel
        # stops.
        base_dir = Path("/kaggle/working") / "atml" / task_name

    elif running_in_colab:
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
