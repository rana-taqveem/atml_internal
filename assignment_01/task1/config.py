import os
from pathlib import Path
import torch
from assignment_01.task1  import utils

class TaskConfig:
    
    def __init__(self, task_name: str, num_classes: int = 10):
        self.task_name = task_name
        self.TASK_DIR = utils.get_data_dir(task_name)
        self.TASK_CHECKPOINTS_DIR = os.path.join(self.TASK_DIR, 'checkpoints')
        self.TASK_FEATURES_DIR = os.path.join(self.TASK_DIR, 'features')
        self.TASK_DATASET_DIR = os.path.join(self.TASK_DIR, 'dataset')
        self.TASK_RESULTS_DIR = os.path.join(self.TASK_DIR, 'results')
        self.TASK_CONFLICT_DATASET_DIR = utils.get_conflict_dataset_dir(self.TASK_DIR)
        self.TASK_TRAIN_DIR = os.path.join(self.TASK_DIR, 'train')
        self.TASK_TEST_DIR = os.path.join(self.TASK_DIR, 'test')
        
        self.SELECTED_INDICES_FILE = os.path.join(self.TASK_DIR, 'selected_indices.npy')
        self.CUE_CONFLICT_DIR = os.path.join(self.TASK_DIR, 'cue_conflict')
        
        self.MODEL_WEIGHTS_DIR = os.path.join(self.TASK_DIR, 'model_weights')
        # Repository fallback for pretrained classifier heads.
        self.PRETRAINED_HEADS_DIR = str(
            Path(__file__).resolve().parent / 'results' / 'baseline' / 'weights'
        )
        
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
        
        self.RESNET50 = "resnet50"
        self.VIT_B_16 = "vit_b_16"
        self.CLIP_VIT_B_32 = "clip_vit_b_32"
        
        ## INTERVENTION SETTINGS
        self.HUE_ROTATION_ANGLE = 30          # degrees, the additional color intervention
        self.PATCH_SIZE = 56                  # 4x4 grid on 224x224 images
        self.TRANSLATION_DISPLACEMENTS = (8, 16, 32)

        ## TRAINED CLASSIFIER HEADS (inside MODEL_WEIGHTS_DIR)
        # Filenames record the validation accuracy checked before inference.
        self.HEAD_WEIGHT_FILES = {
            self.RESNET50: "resnet50_97.30.pth",
            self.VIT_B_16: "vit_b_16_97.90.pth",
            self.CLIP_VIT_B_32: "clip_vit_b_32_98.40.pth",
        }

        self.NUM_CONFLICT_IMAGES = 200
        self.CLASS_PAIRS = [
            (0, 1),  # airplane / bird
            (1, 2),  # bird / car
            (7, 9),  # monkey / truck
            (2, 4),  # car / deer
            (1, 3),  # bird / cat
        ]
                
        # CLIP Specific Prompt
        self.CLIP_PROMPT = "a photo of a {}."
        self.STL10_CLASSES = [
            "airplane", "bird", "car", "cat", "deer", 
            "dog", "horse", "monkey", "ship", "truck"
        ]

    def init_env(self):
        os.makedirs(self.TASK_DIR, exist_ok=True)
        os.makedirs(self.TASK_CHECKPOINTS_DIR, exist_ok=True)
        os.makedirs(self.TASK_FEATURES_DIR, exist_ok=True)
        os.makedirs(self.TASK_DATASET_DIR, exist_ok=True)
        os.makedirs(self.TASK_RESULTS_DIR, exist_ok=True)
        os.makedirs(self.MODEL_WEIGHTS_DIR, exist_ok=True)
        os.makedirs(self.CUE_CONFLICT_DIR, exist_ok=True)
        os.makedirs(self.TASK_CONFLICT_DATASET_DIR, exist_ok=True)

        TORCH_HOME = os.path.join(self.TASK_DIR, 'torch_cache')
        os.environ['TORCH_HOME'] = TORCH_HOME
        
        print(f"Initialized directories for task: {self.task_name}")
        
    def set_num_classes(self, num_classes: int):
        self.NUM_CLASSES = num_classes
        self.TARGET_PER_CLASS = self.TOTAL_TARGET_IMAGES // self.NUM_CLASSES
        print(f"Set NUM_CLASSES to {self.NUM_CLASSES}")  
      

task_config = TaskConfig(task_name='task1')
