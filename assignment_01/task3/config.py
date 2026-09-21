"""Task 3 settings: domain generalization on PACS.

The assignment requires reusing the Task 2 PACS protocol unchanged: the same
source splits, ResNet-18 initialization, seven-class head, domain-balanced
sampling, optimizer settings, epoch budget, early stopping rule and seed 6304.
Those values are therefore imported from the Task 2 config rather than
restated here, so the two tasks cannot drift apart.

Only the output directories and the two method-specific settings are new.

Sketch is the unseen target: no Sketch image may be loaded during training,
source-side diagnostics, checkpoint selection or hyperparameter selection.
Checkpoints are selected on mean macro-F1 over the three source validation
domains, exactly as in Task 2.
"""

import os

from assignment_01.task2.config import task_config as task2_config
from assignment_01.task3 import utils


class Task3Config:

    def __init__(self, task_name="task3"):
        self.task_name = task_name
        self.TASK_DIR = utils.get_data_dir(task_name)
        self.TASK_RESULTS_DIR = os.path.join(self.TASK_DIR, "results")
        self.MODEL_WEIGHTS_DIR = os.path.join(self.TASK_DIR, "model_weights")

        # PACS lives with Task 2; Task 3 reuses the same copy and the same splits.
        self.TASK_DATASET_DIR = task2_config.TASK_DATASET_DIR
        self.TASK2_WEIGHTS_DIR = task2_config.MODEL_WEIGHTS_DIR
        self.ERM_CHECKPOINT = os.path.join(self.TASK2_WEIGHTS_DIR, "erm_best.pth")

        ## Inherited protocol (do not change without changing Task 2 as well)
        self.SEED = task2_config.SEED
        self.DEVICE = task2_config.DEVICE
        self.NUM_CLASSES = task2_config.NUM_CLASSES
        self.SOURCE_DOMAINS = task2_config.SOURCE_DOMAINS
        self.TARGET_DOMAIN = task2_config.TARGET_DOMAIN      # final evaluation only
        self.PACS_CLASSES = task2_config.PACS_CLASSES
        self.LEARNING_RATE = task2_config.LEARNING_RATE
        self.WEIGHT_DECAY = task2_config.WEIGHT_DECAY
        self.NUM_EPOCHS = task2_config.NUM_EPOCHS
        self.EARLY_STOPPING_PATIENCE = task2_config.EARLY_STOPPING_PATIENCE
        self.SOURCE_BATCH_PER_DOMAIN = task2_config.SOURCE_BATCH_PER_DOMAIN
        self.TRAIN_VAL_SPLIT = task2_config.TRAIN_VAL_SPLIT
        self.GRAD_CLIP_NORM = task2_config.GRAD_CLIP_NORM
        self.MMD_BANDWIDTH_MULTIPLIERS = task2_config.MMD_BANDWIDTH_MULTIPLIERS

        ## DAN-DG: pairwise alignment of the three observed source domains
        self.LAMBDA_DG = 1.0                  # main comparison; study varies {0.1, 1, 10}

        ## SAM: non-adaptive sharpness-aware minimization
        self.SAM_RHO = 0.05                   # main comparison; study varies {0.01, 0.05, 0.1}

        ## Diagnostics
        self.SHARPNESS_RADIUS = 0.05          # perturbation radius for the sharpness proxy
        self.SHARPNESS_BATCH_PER_DOMAIN = 32  # fixed validation batch, seed 6304
        self.SEPARABILITY_C = 1.0             # logistic regression regularization
        self.SEPARABILITY_TEST_SIZE = 0.30    # 70/30 split

    def init_env(self):
        for directory in (self.TASK_DIR, self.TASK_RESULTS_DIR, self.MODEL_WEIGHTS_DIR):
            os.makedirs(directory, exist_ok=True)
        os.environ["TORCH_HOME"] = os.path.join(self.TASK_DIR, "torch_cache")
        print(f"Initialized directories for task: {self.task_name}")


task_config = Task3Config()
