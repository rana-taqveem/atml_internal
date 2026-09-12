import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

UNIVERSAL_TRANSFORMS = T.Compose([
   
    T.Lambda(lambda img: img.convert("RGB")),
    T.Resize((224, 224)),
    T.ToTensor(),
])


RESNET_NORMALIZATION = T.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])

CLIP_NORMALIZATION = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                 std=[0.26862954, 0.26130258, 0.27577711])

TV_VIT_B_16_NORMALIZATION = T.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])

def apply_universal_transforms(image):
    return UNIVERSAL_TRANSFORMS(image)

def apply_normalization(image_tensor, model_type):
    if model_type == 'resnet50':
        return RESNET_NORMALIZATION(image_tensor)
    elif model_type == 'clip_vit_b_32':
        return CLIP_NORMALIZATION(image_tensor)
    elif model_type == 'vit_b_16':
        return TV_VIT_B_16_NORMALIZATION(image_tensor)
    else:
        raise ValueError("Invalid model type. Please choose from 'resnet50', 'clip_vit_b_32', or 'vit_b_16'.")       
       
def apply_grey_scale(image_tensor):
    return T.Grayscale(num_output_channels=3)(image_tensor) 

def apply_color_bias(image_tensor, bias_type='red'):
    
    biased_image = image_tensor.clone()
    channel_dim = 1 if image_tensor.ndim == 4 else 0
    
    if bias_type == 'red':
        biased_image.select(channel_dim, 1).zero_()
        biased_image.select(channel_dim, 2).zero_()
    elif bias_type == 'green':
        biased_image.select(channel_dim, 0).zero_()
        biased_image.select(channel_dim, 2).zero_()
    elif bias_type == 'blue':
        biased_image.select(channel_dim, 0).zero_()
        biased_image.select(channel_dim, 1).zero_()
    else:
        raise ValueError("Invalid bias type. Please choose from 'red', 'green', or 'blue'.")

    return biased_image 

def apply_fixed_hue_rotation(image_tensor, angle_degrees):
    
    hue_factor = angle_degrees / 360.0
    return F.adjust_hue(image_tensor, hue_factor)

def apply_palette_transfer(image_tensor, target_palette_mean: torch.Tensor):
    
    is_batch = image_tensor.ndim == 4
    
    view_shape = (-1, 3, 1, 1) if is_batch else (3, 1, 1)
    spatial_dims = (2, 3) if is_batch  else (1, 2)
    
    input_mean = image_tensor.mean(dim=spatial_dims, keepdim=True)
    
    target_mean = target_palette_mean.to(image_tensor.device).view(view_shape)
    
    color_shift = target_mean - input_mean
    transformed_image = image_tensor + color_shift
    return torch.clamp(transformed_image, 0.0, 1.0)
    
