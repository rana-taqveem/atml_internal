import torch
import torch.nn as nn
from torchvision import models
import config
import open_clip

from data.transforms import apply_normalization

task_config = config.TaskConfig(task_name='task1')

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {DEVICE}")

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes = 10):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
    

class resnet50_backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(resnet50_backbone, self).__init__()
        
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        self.model = models.resnet50(weights=weights).to(DEVICE)
        self.model.fc = nn.Identity()  # Remove the final fully connected layer

        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        x = apply_normalization(x, model_type='resnet50')
        return self.model(x)
    
    
class torchvision_vit_b_16_backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(torchvision_vit_b_16_backbone, self).__init__()
        
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        self.model = models.vit_b_16(weights=weights).to(DEVICE)
        self.model.heads = nn.Identity()  
        
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = apply_normalization(x, model_type='vit_b_16')
        return self.model(x)
    
    def get_cls_token(self, x):
        
        x = apply_normalization(x, model_type='vit_b_16')
        shape = x.shape[0]    
        x = self.model._process_input(x) 
        batch_class_token = self.model.cls_token.expand(shape, -1, -1)
        x = torch.cat((batch_class_token, x), dim=1)
        x = self.model.encoder(x)
        return x[:, 0]
        
    
class openai_clip_backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(openai_clip_backbone, self).__init__()
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        self.model = model
        self.model.visual.proj = nn.Identity()  
        
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        x = apply_normalization(x, model_type='clip_vit_b_32')
        img_embedding = self.model.encode_image(x)
        img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
        
        return img_embedding
    