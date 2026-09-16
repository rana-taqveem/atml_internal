import json
from pathlib import Path
import numpy as np
from torchvision.datasets import STL10
from torchvision.utils import save_image

from assignment_01.task1.data.make_subset import get_test_subset
from assignment_01.task1.config import task_config
from assignment_01.task1.data.transforms import StyleTransferModel, apply_universal_transforms
from assignment_01.task1.data.make_cue_conflicts import generate_cue_conflicts

def main():
    
    task_config.init_env()
    
    task_dir = Path(__file__).resolve().parents[1]
    weights_dir = task_dir /"data"/ "external" / "model_weights"
    
    encoder_path = weights_dir / "vgg_normalised.pth"
    decoder_path = weights_dir / "decoder.pth"
    
    stylizer = StyleTransferModel(encoder_path, decoder_path)
    
    # Load the STL10 test dataset
    test_dataset = STL10(root=task_config.TASK_DATASET_DIR, split='test', download=False, transform=apply_universal_transforms)
    
    indices_path = Path(task_config.SELECTED_INDICES_FILE)

    if not indices_path.is_file():
        print("Preparing the fixed test subset and saving its image IDs...")
        get_test_subset(transformation_type="original")

    selected_indices = np.load(indices_path).tolist()
    content_class, style_class = task_config.CLASS_PAIRS[0]  # Example: (0, 1)
    
    content_id = next(
        image_id for image_id in selected_indices 
        if int(test_dataset.labels[image_id]) == content_class
    )
    
    style_id = next(
        image_id for image_id in selected_indices
        if int(test_dataset.labels[image_id]) == style_class
    )
    
    conflict_image, metadata = generate_cue_conflicts(
        dataset=test_dataset,
        content_id=content_id,
        style_id=style_id,
        stylizer=stylizer,
        alpha=0.5
    )
    
    conflict_id = f"conflict_{content_id}_{style_id}"
    conflict_image_path = str(Path(task_config.TASK_CONFLICT_DATASET_DIR) / f"{conflict_id}.png")
    
    metadata["conflict_id"] = conflict_id
    metadata["image_path"] = conflict_image_path

    save_image(conflict_image, conflict_image_path)
    
    content_image, _ = test_dataset[content_id]
    style_image, _ = test_dataset[style_id]
    
    save_image([content_image, style_image, conflict_image], 
               str(Path(task_config.TASK_CONFLICT_DATASET_DIR) / f"preview_{conflict_id}.png"),
               nrow=3)
    
    with (Path(task_config.TASK_CONFLICT_DATASET_DIR) / f"{conflict_id}.json").open("w", encoding="utf-8") as file: json.dump(metadata, file, indent=2)

    print("Saved candidate:", conflict_id)
    print("Preview order: content | style | generated conflict")
    
if __name__ == "__main__":
    main()