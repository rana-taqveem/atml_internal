"""Task 4 open-set recognition settings.

CIFAR-100 is evaluation-only; thresholds are calibrated on CIFAR-10 validation.
"""

import os

import torch

from assignment_01.task4 import utils


class Task4Config:

    def __init__(self, task_name="task4", num_classes=10):
        self.task_name = task_name
        self.TASK_DIR = utils.get_data_dir(task_name)
        self.TASK_DATASET_DIR = os.path.join(self.TASK_DIR, "dataset")
        self.TASK_RESULTS_DIR = os.path.join(self.TASK_DIR, "results")
        self.MODEL_WEIGHTS_DIR = os.path.join(self.TASK_DIR, "model_weights")
        self.CACHE_DIR = os.path.join(self.TASK_DIR, "cache")

        ## PROTOCOL
        self.SEED = 6304
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.NUM_CLASSES = num_classes            # the ten CIFAR-10 classes
        self.TRAIN_VAL_SPLIT = 0.10               # stratified 90/10 of the train partition

        ## OPTIMIZATION (Vanilla and GCSC)
        self.LEARNING_RATE = 0.1
        self.MOMENTUM = 0.9
        self.WEIGHT_DECAY = 5e-4
        self.NUM_EPOCHS = 100
        self.BATCH_SIZE = 128
        self.SCHEDULE = "cosine"

        ## AUGMENTATION
        self.CROP_SIZE = 32
        self.CROP_PADDING = 4
        self.RANDAUGMENT_NUM_OPS = 2              # GCSC only
        self.RANDAUGMENT_MAGNITUDE = 9            # GCSC only

        ## PROSER (fine-tune from the selected Vanilla checkpoint)
        self.PROSER_NUM_DUMMY = 5                 # appended dummy classifiers
        self.PROSER_EPOCHS = 50
        self.PROSER_LR = 1e-3
        self.PROSER_BETA = 1.0                    # classifier-placeholder loss weight
        self.PROSER_GAMMA = 0.1                   # data-placeholder loss weight
        self.PROSER_MIXUP_ALPHA = 2.0             # lambda ~ Beta(2, 2)
        self.PROSER_MIXUP_LAYER = "layer2"

        ## SCORES
        self.MAHALANOBIS_EPSILON = 1e-6           # added to every diagonal entry

        ## EVALUATION
        # Calibrated to accept about 95% of known validation examples.
        self.ACCEPT_PERCENTILE = 95.0

        ## DATA: CIFAR-10 knowns, CIFAR-100 unknowns (evaluation only)
        self.CIFAR10_CLASSES = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ]
        self.NEAR_UNKNOWN_CLASSES = [
            "bus", "pickup_truck", "motorcycle", "tractor",
            "wolf", "fox", "leopard", "camel",
        ]
        self.FAR_UNKNOWN_CLASSES = [
            "bottle", "bowl", "chair", "clock",
            "keyboard", "mushroom", "sunflower", "wardrobe",
        ]

        # Shared preprocessing for known and unknown inputs.
        self.NORMALIZE_MEAN = (0.4914, 0.4822, 0.4465)
        self.NORMALIZE_STD = (0.2470, 0.2435, 0.2616)

    def init_env(self):
        for directory in (self.TASK_DIR, self.TASK_DATASET_DIR, self.TASK_RESULTS_DIR,
                          self.MODEL_WEIGHTS_DIR, self.CACHE_DIR):
            os.makedirs(directory, exist_ok=True)
        os.environ["TORCH_HOME"] = os.path.join(self.TASK_DIR, "torch_cache")
        print(f"Initialized directories for task: {self.task_name}")

    def checkpoint_path(self, method):
        return os.path.join(self.MODEL_WEIGHTS_DIR, f"{method}_best.pth")


task_config = Task4Config()
