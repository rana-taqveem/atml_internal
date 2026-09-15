import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from copy import deepcopy
import torch.nn as nn


from assignment_01.task1.config import task_config


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
    if model_type == task_config.RESNET50:
        return RESNET_NORMALIZATION(image_tensor)
    elif model_type == task_config.CLIP_VIT_B_32:
        return CLIP_NORMALIZATION(image_tensor)
    elif model_type == task_config.VIT_B_16:
        return TV_VIT_B_16_NORMALIZATION(image_tensor)
    else:
        raise ValueError("Invalid model type. Please choose from 'resnet50', 'clip_vit_b_32', or 'vit_b_16'.")             

def apply_grey_scale(image_tensor):
    return T.Grayscale(num_output_channels=3)(image_tensor) 

def apply_fixed_hue_rotation(image_tensor, angle_degrees):

    if not -180 <= angle_degrees <= 180:
        raise ValueError("Angle must be between -180 and 180 degrees.")
    
    hue_factor = angle_degrees / 360.0
    return F.adjust_hue(image_tensor, hue_factor)

def apply_translation(image_tensor, translate_x, translate_y):
    
    if image_tensor.ndim not in [3, 4]:
        raise ValueError("Input tensor must be 3D (C, H, W) or 4D (N, C, H, W).")
    
    if not isinstance(translate_x, int) or not isinstance(translate_y, int):
        raise ValueError("Translation values must be integers.")
    
    
    padding = 32
    
    if abs(translate_x) > padding or abs(translate_y) > padding:
        raise ValueError(f"Translation values must be within the range [-{padding}, {padding}].")
    
    height, width = image_tensor.shape[-2:]
    
    if height <= padding or width <= padding:
        raise ValueError("Image dimensions must be greater than the padding size.")
    
    padded_image = F.pad(image_tensor, 
                         padding=[padding]*4,
                         padding_mode='reflect')
    
    top = padding - translate_y
    left = padding - translate_x
    
    translated_image = padded_image[..., 
                                    top:top+height,
                                    left:left+width
                                    ]
   
    return translated_image
   
def show_transformation(image_tensor, transformation_func, *args, **kwargs):
    transformed_image = transformation_func(image_tensor, *args, **kwargs)
    
    import matplotlib.pyplot as plt
    
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_tensor.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Original Image")
    
    axes[0].axis('off')
    axes[1].imshow(transformed_image.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title("Transformed Image")
    
    axes[1].axis('off')
    plt.title(f"Transformation: {transformation_func.__name__}")
    plt.show()
    
class PatchShuffler:
    
    def __init__(self, patch_size = 56, seed = 6304):
        
        if not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError("Patch size must be a positive integer.")
        
        self.patch_size = patch_size
        
        self.num_patches = (224 // patch_size) ** 2
        
        # Initialize a random generator with a fixed seed for reproducibility
        self.gererator = torch.Generator()
        self.gererator.manual_seed(seed)  # For reproducibility

    def __call__(self, image_tensor):
        if image_tensor.ndim != 3:
            raise ValueError("Input tensor must be 3D (C, H, W).")
        
        C, H, W = image_tensor.shape
        p = self.patch_size
        
        if H % p != 0 or W % p != 0:
            raise ValueError("Image dimensions must be divisible by the patch size.")
        
        # Calculate the number of patches along height and width
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
    
        # Enforce the assignment's 4×4 grid.
        if num_patches_h != 4 or num_patches_w != 4:
            raise ValueError(
                "Expected a 4x4 grid. Use patch_size=56 for 224x224 images."
            )
            
        # Extract non-overlapping patches: [C, 4, 4, p, p].
        patches = (image_tensor.unfold(1, self.patch_size, self.patch_size)
                              .unfold(2, self.patch_size, self.patch_size))
        
        # Arrange memory consecutively and flatten the grid into 16 patches.
        patches = patches.contiguous().view(C, self.num_patches, self.patch_size, self.patch_size)
        
        identity = torch.arange(self.num_patches)
        
         # Draw a random order containing every patch index exactly once.
        permuted_indices = torch.randperm(self.num_patches, generator=self.gererator)
        
        # Draw again if the complete arrangement is unchanged
        while torch.equal(permuted_indices, identity):
            permuted_indices = torch.randperm(self.num_patches, generator=self.gererator)
        
        # Put indexing values on the same device as the image.
        indices = permuted_indices.to(image_tensor.device)
        
        # Restore separate grid-row and grid-column dimensions.
        shuffled_patches = patches[:, indices]
        
        # Order axes as: channel, grid row, pixel row, grid column, pixel column.
        shuffled_image = shuffled_patches.view(C, num_patches_h, num_patches_w, self.patch_size, self.patch_size)
        
        # Order axes as: channel, grid row, pixel row, grid column, pixel column.
        shuffled_image = shuffled_image.permute(0, 1, 3, 2, 4)
        
        # Combine the height and width dimensions into an image.
        shuffled_image = shuffled_image.contiguous().view(C, H, W)
        
        return shuffled_image, permuted_indices
    

class StyleTransferModel(nn.Module):
    
    def __init__(self, encoder_path, decoder_path):
        super(StyleTransferModel, self).__init__()
        
         # Load the dependency only when style transfer is requested.
        from assignment_01.task1.data.external.adain import net
        from assignment_01.task1.data.external.adain.function import (
            adaptive_instance_normalization,
        )

        self.adain = adaptive_instance_normalization
        
        self.encoder = encoder_path
        self.decoder = decoder_path
        self.device = task_config.DEVICE
        
        full_encoder = deepcopy(net.vgg)
        self.decoder = deepcopy(net.decoder)
        
        encoder_weights = torch.load(
            encoder_path,
            map_location="cpu",
            weights_only=True
        )
        
        decoder_weights = torch.load(
            decoder_path,
            map_location="cpu",
            weights_only=True
        )
    
        full_encoder.load_state_dict(encoder_weights)
        self.decoder.load_state_dict(decoder_weights)
        
        self.encoder = nn.Sequential(*list(full_encoder.children())[:31])
        
        self.encoder.to(self.device).eval()
        self.decoder.to(self.device).eval()
        
        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(False)
            
    @torch.no_grad()
    def __call__(self, content_image, style_image, alpha=0.5):

        if not (0 <= alpha <= 1):
            raise ValueError("Alpha must be in the range [0, 1]")
        
        for image in (content_image, style_image):
            if image.shape[1] != (3, 224, 224):
                raise ValueError("Input images must have shape (3, 224, 224)")
            
            if not torch.is_floating_point(image):
                raise TypeError("Input images must be of floating point type")
            
            if not torch.isfinite(image).all():
                raise ValueError("Input images must not contain NaN or Inf values")
            
            if image.min() < 0 or image.max() > 1:
                raise ValueError("Input images must be normalized to the range [0, 1]")
            
        content_image = content_image.unsqueeze(0).to(
            device = self.device,
            dtype = torch.float32
        )
        
        style_image = style_image.unsqueeze(0).to(
            device = self.device,
            dtype = torch.float32
        )
        
        content_features = self.encoder(content_image)
        style_features = self.encoder(style_image)
        
        transferred_features = self.adain(content_features, style_features)
        mixed_features = alpha * transferred_features + (1 - alpha) * content_features
    
        output_image = self.decoder(mixed_features)
        return output_image.squeeze(0).clamp(0, 1).cpu()