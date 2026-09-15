import torch
import open_clip
import torch.nn as nn
from torchvision import models

from assignment_01.task1.config import task_config
from assignment_01.task1.data.transforms import apply_normalization

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes = 10):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
    
class Resnet50Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(Resnet50Backbone, self).__init__()
        
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        self.model = models.resnet50(weights=weights).to(task_config.DEVICE)
        self.model.fc = nn.Identity()
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        x = apply_normalization(x, model_type=task_config.RESNET50)
        return self.model(x)
       
class Torchvision_Vit_B_16_Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(Torchvision_Vit_B_16_Backbone, self).__init__()
        
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        self.model = models.vit_b_16(weights=weights).to(task_config.DEVICE)
        self.model.heads = nn.Identity() 
        
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = apply_normalization(x, model_type=task_config.VIT_B_16)
        return self.model(x)
    
    def get_cls_token(self, x):
        
        x = apply_normalization(x, model_type=task_config.VIT_B_16)
        shape = x.shape[0]    
        x = self.model._process_input(x) 
        batch_class_token = self.model.cls_token.expand(shape, -1, -1)
        x = torch.cat((batch_class_token, x), dim=1)
        x = self.model.encoder(x)
        return x[:, 0]
         
class Openai_Clip_Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(Openai_Clip_Backbone, self).__init__()
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        x = apply_normalization(x, model_type=task_config.CLIP_VIT_B_32)
        img_embedding = self.model.encode_image(x)
        img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
        
        return img_embedding
    