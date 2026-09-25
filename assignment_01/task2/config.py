import os
import torch
from assignment_01.task2 import utils


class TaskConfig:
    """Task 2 settings: unsupervised domain adaptation on PACS.

    Hyperparameters below follow the assignment specification. Values that
    still need to be decided or confirmed against the PDF are marked TODO.
    """

    def __init__(self, task_name: str, num_classes: int = 7):
        self.task_name = task_name
        self.TASK_DIR = utils.get_data_dir(task_name)
        self.TASK_CHECKPOINTS_DIR = os.path.join(self.TASK_DIR, 'checkpoints')
        self.TASK_FEATURES_DIR = os.path.join(self.TASK_DIR, 'features')
        self.TASK_DATASET_DIR = os.path.join(self.TASK_DIR, 'dataset')
        self.TASK_RESULTS_DIR = os.path.join(self.TASK_DIR, 'results')
        self.MODEL_WEIGHTS_DIR = os.path.join(self.TASK_DIR, 'model_weights')

        ## OPTIMIZATION HYPERPARAMETERS (PA1 Task 2)
        self.LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = 1e-4
        self.NUM_EPOCHS = 30
        self.EARLY_STOPPING_PATIENCE = 5      # on mean source-validation macro-F1
        self.SOURCE_BATCH_PER_DOMAIN = 8      # 8 per source domain -> 24 source
        self.TARGET_BATCH_SIZE = 24           # equal total source/target batch size

        ## DATASET CONFIGURATIONS
        self.SEED = 6304
        self.TRAIN_VAL_SPLIT = 0.20           # stratified, within each source domain
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.NUM_CLASSES = num_classes        # PACS has 7 classes

        self.SOURCE_DOMAINS = ("photo", "art_painting", "cartoon")
        self.TARGET_DOMAIN = "sketch"
        self.PACS_CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]

        ## MODEL
        self.RESNET18 = "resnet18"            # ResNet18_Weights.IMAGENET1K_V1
        self.RESIZE = 256
        self.CROP = 224

        # Shared clipping limits unstable adversarial updates.
        self.GRAD_CLIP_NORM = 1.0

        ## DAN (MMD alignment)
        self.DAN_LAMBDA_MMD = 1.0                    # main comparison; study varies {0.1, 1, 10}
        self.MMD_BANDWIDTH_MULTIPLIERS = (0.5, 1.0, 2.0)  # x median pairwise squared distance

        ## DANN / CDAN (adversarial alignment)
        self.DISCRIMINATOR_HIDDEN = 256
        self.DISCRIMINATOR_DROPOUT = 0.5
        self.DOMAIN_LOSS_WEIGHT = 1.0                # unit weight, per the assignment
        self.DANN_MAX_ALPHA = 1.0                    # study varies {0.25, 0.5, 1}

    def init_env(self):
        for directory in (self.TASK_DIR, self.TASK_CHECKPOINTS_DIR, self.TASK_FEATURES_DIR,
                          self.TASK_DATASET_DIR, self.TASK_RESULTS_DIR, self.MODEL_WEIGHTS_DIR):
            os.makedirs(directory, exist_ok=True)

        os.environ['TORCH_HOME'] = os.path.join(self.TASK_DIR, 'torch_cache')
        print(f"Initialized directories for task: {self.task_name}")

    def set_num_classes(self, num_classes: int):
        self.NUM_CLASSES = num_classes
        print(f"Set NUM_CLASSES to {self.NUM_CLASSES}")


task_config = TaskConfig(task_name='task2')
