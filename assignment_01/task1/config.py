import os
import torch
import utils

class TaskConfig:
    
    def __init__(self, task_name: str, num_classes: int = 10):
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
        
        self.MODEL_WEIGHTS_DIR = os.path.join(self.TASK_DIR, 'model_weights')
        
        ## OPTIMIZATION HYPERPARAMETERS
        self.LEARNING_RATE = 1e-3
        self.WEIGHT_DECAY = 1e-4
        self.EARLY_STOPPING_PATIENCE = 5
        self.BATCH_SIZE = 64
        self.NUM_EPOCHS = 50
        
        ## DATASET CONFIGURATIONS
        self.SEED = 6304
        self.TRAIN_VAL_SPLIT = 0.20
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.TOTAL_TARGET_IMAGES = 500
        self.NUM_CLASSES = num_classes
        self.TARGET_PER_CLASS = self.TOTAL_TARGET_IMAGES // self.NUM_CLASSES

    def init_env(self):
        os.makedirs(self.TASK_DIR, exist_ok=True)
        os.makedirs(self.TASK_CHECKPOINTS_DIR, exist_ok=True)
        os.makedirs(self.TASK_FEATURES_DIR, exist_ok=True)
        os.makedirs(self.TASK_DATASET_DIR, exist_ok=True)
        os.makedirs(self.TASK_RESULTS_DIR, exist_ok=True)
        os.makedirs(self.MODEL_WEIGHTS_DIR, exist_ok=True)
        os.makedirs(self.CUE_CONFLICT_DIR, exist_ok=True)

        TORCH_HOME = os.path.join(self.TASK_DIR, 'torch_cache')
        os.environ['TORCH_HOME'] = TORCH_HOME
        
        print(f"Initialized directories for task: {self.task_name}")
        
    def set_num_classes(self, num_classes: int):
        self.NUM_CLASSES = num_classes
        self.TARGET_PER_CLASS = self.TOTAL_TARGET_IMAGES // self.NUM_CLASSES
        print(f"Set NUM_CLASSES to {self.NUM_CLASSES}")  
      

task_config = TaskConfig(task_name='task1')