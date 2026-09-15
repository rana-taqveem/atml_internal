import torch

def generate_cue_conflicts(dataset, 
                           content_id, 
                           style_id, 
                           stylizer,
                           alpha=0.5):
    
    style_image, style_lablel = dataset[style_id]
    content_image, content_label = dataset[content_id]
    
    if content_label == style_lablel:
        raise ValueError("Content and style images must have different class labels.")


    conflict_image = stylizer(content_image, 
                              style_image, 
                              alpha=alpha)
    
    metadata = {
        "content_id": int(content_id),
        "style_id": int(style_id),
        "content_label": int(content_label),
        "style_label": int(style_lablel),
        "alpha": float(alpha),
        "accepted": None,
        "rejection_reason": None
    }
        
    return conflict_image, metadata