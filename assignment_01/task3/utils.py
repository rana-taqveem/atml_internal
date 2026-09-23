"""Data-root resolution for Task 3.

Identical logic to Task 1 and Task 2: Drive on Colab, a local folder otherwise.
Only the results and checkpoints live here; PACS itself is shared with Task 2.
"""

import os
from pathlib import Path

from assignment_01.task2.utils import is_running_in_colab, is_running_in_kaggle


def get_data_dir(task_name: str):
    if is_running_in_kaggle():
        # /kaggle/working is the only writable location that survives to the
        # session's output; anything written elsewhere is lost when the kernel
        # stops.
        base_dir = Path("/kaggle/working") / "atml" / task_name

    elif is_running_in_colab():
        from google.colab import drive

        mount_point = "/content/drive"
        if not os.path.isdir(f"{mount_point}/MyDrive"):
            drive.mount(mount_point)

        base_dir = Path(mount_point) / "MyDrive" / "ATML" / "assignment_01" / task_name
    else:
        base_dir = Path(__file__).resolve().parent / "local_runs" / task_name

    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir)
