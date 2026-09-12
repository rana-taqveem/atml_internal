


from email.mime import image

import torchvision.transforms as T


# Define a set of universal transformations to be applied to all images
# this is a global variable that can be used across different modules
# and is more efficient than defining the same transformations multiple times
UNIVERSAL_TRANSFORMS = T.Compose([
    T.ToPILImage(),
    T.ConvertImageMode('RGB'),
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