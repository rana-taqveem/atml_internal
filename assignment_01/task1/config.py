import os
import torch
import utils



## OPTIMIZATION HYPERPARAMETERS
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5
BATCH_SIZE = 64
NUM_EPOCHS = 20

## DATASET CONFIGURATIONS
SEED = 6304
TRAIN_VAL_SPLIT = 0.20
TOTAL_TARGET_IMAGES = 500
TARGET_PER_CLASS = TOTAL_TARGET_IMAGES // NUM_CLASSES

class TaskConfig:
    
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.TASK_DIR = utils.get_data_dir(task_name)
        self.TASK_CHECKPOINTS_DIR = os.path.join(self.TASK_DIR, 'checkpoints')
        self.TASK_FEATURES_DIR = os.path.join(self.TASK_DIR, 'features')
        self.TASK_DATASET_DIR = os.path.join(self.TASK_DIR, 'dataset')
        self.TASK_RESULTS_DIR = os.path.join(self.TASK_DIR, 'results')
        self.TASK_TRAIN_DIR = os.path.join(self.TASK_DIR, 'train')
        self.TASK_TEST_DIR = os.path.join(self.TASK_DIR, 'test')
        
        self.SELECTED_INDICES_FILE = os.path.join(self.TASK_DIR, 'selected_indices.npy')
        self.CUE_CONFLICT_DIR = os.path.join(self.TASK_DIR, 'cue_conflict')

    def init_env(self):
        os.makedirs(self.TASK_DIR, exist_ok=True)
        os.makedirs(self.TASK_CHECKPOINTS_DIR, exist_ok=True)
        os.makedirs(self.TASK_FEATURES_DIR, exist_ok=True)
        os.makedirs(self.TASK_DATASET_DIR, exist_ok=True)
        os.makedirs(self.TASK_RESULTS_DIR, exist_ok=True)
        os.makedirs(self.CUE_CONFLICT_DIR, exist_ok=True)

        TORCH_HOME = os.path.join(self.TASK_DIR, 'torch_cache')
        os.environ['TORCH_HOME'] = TORCH_HOME
        
        print(f"Initialized directories for task: {self.task_name}")
        
    def set_num_classes(self, num_classes: int):
        global NUM_CLASSES
        NUM_CLASSES = num_classes
        print(f"Set NUM_CLASSES to {NUM_CLASSES}")  
      