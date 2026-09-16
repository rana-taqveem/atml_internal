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
    
    # conflict_image, metadata = generate_cue_conflicts(
    #     dataset=test_dataset,
    #     content_id=content_id,
    #     style_id=style_id,
    #     stylizer=stylizer,
    #     alpha=0.5
    # )
    
    candidates_per_direction = 40
    alpha = 1.0
    rng = np.random.default_rng(task_config.SEED)

    # Separate production candidates from earlier pilot images.
    output_dir = (
        Path(task_config.TASK_CONFLICT_DATASET_DIR)
        / "candidates_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    # Group original dataset IDs by class.
    ids_by_class = {
        class_id: [
            int(image_id)
            for image_id in selected_indices
            if int(test_dataset.labels[image_id]) == class_id
        ]
        for class_id in range(task_config.NUM_CLASSES)
    }

    records = []
    sampling_plan = []

    for class_a, class_b in task_config.CLASS_PAIRS:
        directions = [
            (class_a, class_b),
            (class_b, class_a),
        ]

        for content_class, style_class in directions:
            # All distinct content/style combinations for this direction.
            combinations = [
                (content_id, style_id)
                for content_id in ids_by_class[content_class]
                for style_id in ids_by_class[style_class]
            ]

            if len(combinations) < candidates_per_direction:
                raise ValueError(
                    f"Not enough combinations for "
                    f"{content_class} -> {style_class}"
                )

            # Save the full random order so later candidates can extend it.
            order = rng.permutation(len(combinations))
            ordered_pairs = [
                combinations[int(index)] for index in order
            ]

            sampling_plan.append({
                "content_class": content_class,
                "style_class": style_class,
                "ordered_pairs": ordered_pairs,
            })

    # Save the plan before expensive image generation.
    with (output_dir / "sampling_plan.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump({
            "seed": task_config.SEED,
            "alpha": alpha,
            "candidates_per_direction": candidates_per_direction,
            "groups": sampling_plan,
        }, file, indent=2)

    for group in sampling_plan:
        content_class = group["content_class"]
        style_class = group["style_class"]

        for candidate_order, (content_id, style_id) in enumerate(
            group["ordered_pairs"][:candidates_per_direction]
        ):
            conflict_image, metadata = generate_cue_conflicts(
                dataset=test_dataset,
                content_id=content_id,
                style_id=style_id,
                stylizer=stylizer,
                alpha=alpha,
            )

            conflict_id = (
                f"c{content_class}_s{style_class}"
                f"_{candidate_order:04d}"
                f"_{content_id}_{style_id}"
            )

            image_name = f"{conflict_id}.png"
            preview_name = f"preview_{conflict_id}.png"

            save_image(conflict_image, output_dir / image_name)

            content_image, _ = test_dataset[content_id]
            style_image, _ = test_dataset[style_id]

            save_image(
                [content_image, style_image, conflict_image],
                output_dir / preview_name,
                nrow=3,
            )

            metadata.update({
                "conflict_id": conflict_id,
                "pair": sorted([content_class, style_class]),
                "direction": f"{content_class}->{style_class}",
                "candidate_order": candidate_order,
                "image_path": image_name,
                "preview_path": preview_name,
                "selected_for_evaluation": False,
            })

            # Save each record immediately to preserve completed work.
            with (output_dir / f"{conflict_id}.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(metadata, file, indent=2)

            records.append(metadata)
            print(f"Saved candidate {len(records)}: {conflict_id}")

    with (output_dir / "manifest.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(records, file, indent=2)

    print("Candidates generated:", len(records))
    print("Review directory:", output_dir)
    
if __name__ == "__main__":
    main()