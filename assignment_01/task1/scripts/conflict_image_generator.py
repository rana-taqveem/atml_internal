"""Generate a balanced pool of cue-conflict candidates for visual review.

For five unordered class pairs, generate both directions with 40 candidates
per direction: 5 * 2 * 40 = 400 candidates.

Each candidate contains:
    content.png   -- clean content input, resized to 224x224
    style.png     -- clean style input, resized to 224x224
    conflict.png  -- generated image
    preview.png   -- content | style | conflict
    metadata.json

Image paths in metadata are relative to the run directory.
Candidates must be visually reviewed before classifier inference.
"""

import json
from pathlib import Path

import numpy as np
from torchvision.datasets import STL10
from torchvision.utils import save_image

from assignment_01.task1.config import task_config
from assignment_01.task1.data.download import prepare_stl10
from assignment_01.task1.data.make_cue_conflicts import generate_cue_conflicts
from assignment_01.task1.data.make_subset import get_test_subset
from assignment_01.task1.data.transforms import (
    StyleTransferModel,
    apply_universal_transforms,
)


ALPHA = 0.7
CANDIDATES_PER_DIRECTION = 40
TARGET_ACCEPTED_PER_DIRECTION = 20
RUN_VERSION = "v1"


def save_json(path, data):
    """Write metadata using ordinary JSON-compatible values."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def main():
    """Prepare inputs, save a seeded sampling plan, then generate candidates.

    Each direction samples distinct content/style combinations without
    replacement. Individual source images may occur in several combinations.

    The complete randomized combination order is saved so future generation
    can extend a direction that has too few visually accepted candidates.

    This script does not resume interrupted runs. Existing run directories
    cause an error to prevent overwriting images or review decisions.
    """
    task_config.init_env()

    if not 0 < ALPHA <= 1:
        raise ValueError("Production alpha must be greater than 0 and at most 1.")

    if CANDIDATES_PER_DIRECTION < TARGET_ACCEPTED_PER_DIRECTION:
        raise ValueError("Generate at least as many candidates as the target.")

    task_dir = Path(__file__).resolve().parents[1]
    weights_dir = task_dir / "data" / "external" / "model_weights"

    encoder_path = weights_dir / "vgg_normalised.pth"
    decoder_path = weights_dir / "decoder.pth"

    for weight_path in (encoder_path, decoder_path):
        if not weight_path.is_file():
            raise FileNotFoundError(f"Missing style-transfer weights: {weight_path}")

    alpha_tag = str(ALPHA).replace(".", "_")
    output_dir = (
        Path(task_config.TASK_CONFLICT_DATASET_DIR)
        / f"candidates_alpha_{alpha_tag}_{RUN_VERSION}"
    )

    if output_dir.exists():
        raise FileExistsError(
            f"Run directory already exists: {output_dir}\n"
            "Choose a new RUN_VERSION to preserve the existing run."
        )

    prepare_stl10(task_config.TASK_DATASET_DIR)

    indices_path = Path(task_config.SELECTED_INDICES_FILE)
    if not indices_path.is_file():
        print("Preparing the fixed test subset and saving its IDs...")
        get_test_subset(transformation_type="original")

    test_dataset = STL10(
        root=task_config.TASK_DATASET_DIR,
        split="test",
        download=False,
        transform=apply_universal_transforms,
    )

    selected_indices = np.load(indices_path, allow_pickle=False)

    if selected_indices.ndim != 1 or selected_indices.size == 0:
        raise ValueError("Selected image IDs must be a nonempty 1D array.")

    if not np.issubdtype(selected_indices.dtype, np.integer):
        raise ValueError("Selected image IDs must be integers.")

    if np.unique(selected_indices).size != selected_indices.size:
        raise ValueError("Selected image IDs contain duplicates.")

    if selected_indices.min() < 0 or selected_indices.max() >= len(test_dataset):
        raise ValueError("Selected image IDs are outside the test dataset.")

    selected_indices = selected_indices.tolist()
    class_names = list(task_config.STL10_CLASSES)

    if list(test_dataset.classes) != class_names:
        raise ValueError("Configured class order does not match STL10.")

    class_pairs = []
    seen_pairs = set()

    for pair in task_config.CLASS_PAIRS:
        if len(pair) != 2:
            raise ValueError("Each class pair must contain exactly two IDs.")

        class_a, class_b = pair

        if not all(
            isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_))
            for value in pair
        ):
            raise ValueError("Class IDs must be integers.")

        class_a, class_b = int(class_a), int(class_b)

        if not (
            0 <= class_a < len(class_names)
            and 0 <= class_b < len(class_names)
        ):
            raise ValueError("A configured class ID is out of range.")

        if class_a == class_b:
            raise ValueError("Content and style classes must differ.")

        pair_key = tuple(sorted((class_a, class_b)))

        if pair_key in seen_pairs:
            raise ValueError(f"Duplicate unordered class pair: {pair_key}")

        seen_pairs.add(pair_key)
        class_pairs.append(pair_key)

    if len(class_pairs) < 5:
        raise ValueError("Task 1 requires at least five unordered class pairs.")

    ids_by_class = {
        class_id: [
            image_id
            for image_id in selected_indices
            if int(test_dataset.labels[image_id]) == class_id
        ]
        for class_id in range(len(class_names))
    }

    rng = np.random.default_rng(task_config.SEED)
    sampling_plan = []

    for class_a, class_b in class_pairs:
        pair_name = f"{class_names[class_a]}__{class_names[class_b]}"

        for content_class, style_class in (
            (class_a, class_b),
            (class_b, class_a),
        ):
            combinations = [
                (content_id, style_id)
                for content_id in ids_by_class[content_class]
                for style_id in ids_by_class[style_class]
            ]

            if len(combinations) < CANDIDATES_PER_DIRECTION:
                raise ValueError(
                    f"Not enough combinations for "
                    f"{class_names[content_class]} content / "
                    f"{class_names[style_class]} style."
                )

            random_order = rng.permutation(len(combinations))
            ordered_pairs = [
                combinations[int(index)] for index in random_order
            ]

            sampling_plan.append({
                "pair": [class_a, class_b],
                "pair_name": pair_name,
                "content_class": content_class,
                "style_class": style_class,
                "direction_name": (
                    f"{class_names[content_class]}_content__"
                    f"{class_names[style_class]}_style"
                ),
                "ordered_pairs": ordered_pairs,
            })

    stylizer = StyleTransferModel(
        encoder_path=encoder_path,
        decoder_path=decoder_path,
    )

    output_dir.mkdir(parents=True, exist_ok=False)

    review_protocol = {
        "version": "visual_review_v1",
        "accept_only_if": [
            "Source objects are interpretable and style texture is visible.",
            "The output preserves recognizable content shape.",
            "Transferred texture is visible on the content object, "
            "not merely a color change.",
            "Severe artifacts or background-dominated transfer do not "
            "make the intended conflict ambiguous.",
        ],
        "rejection_reasons": [
            "source_ambiguous",
            "shape_lost",
            "weak_texture",
            "background_dominates",
            "severe_artifacts",
            "ambiguous",
        ],
        "selection_rule": (
            "After visual review, select the first target number of "
            "accepted candidates in candidate_order within each direction."
        ),
        "model_predictions_used_for_selection": False,
    }

    save_json(output_dir / "review_protocol.json", review_protocol)

    save_json(output_dir / "sampling_plan.json", {
        "seed": int(task_config.SEED),
        "alpha": ALPHA,
        "classes": class_names,
        "selected_image_ids": selected_indices,
        "candidates_per_direction": CANDIDATES_PER_DIRECTION,
        "target_accepted_per_direction": TARGET_ACCEPTED_PER_DIRECTION,
        "encoder_path": str(encoder_path),
        "decoder_path": str(decoder_path),
        "groups": sampling_plan,
    })

    records = []
    expected_total = len(sampling_plan) * CANDIDATES_PER_DIRECTION

    for group in sampling_plan:
        content_class = group["content_class"]
        style_class = group["style_class"]

        candidate_pairs = group["ordered_pairs"][:CANDIDATES_PER_DIRECTION]

        for candidate_order, (content_id, style_id) in enumerate(candidate_pairs):
            conflict_image, metadata = generate_cue_conflicts(
                dataset=test_dataset,
                content_id=content_id,
                style_id=style_id,
                stylizer=stylizer,
                alpha=ALPHA,
            )

            conflict_id = (
                f"c{content_class}_s{style_class}"
                f"_{candidate_order:04d}_{content_id}_{style_id}"
            )

            candidate_dir = (
                output_dir
                / group["pair_name"]
                / group["direction_name"]
                / f"candidate_{candidate_order:04d}_{content_id}_{style_id}"
            )
            candidate_dir.mkdir(parents=True, exist_ok=False)

            content_image, _ = test_dataset[content_id]
            style_image, _ = test_dataset[style_id]

            content_path = candidate_dir / "content.png"
            style_path = candidate_dir / "style.png"
            conflict_path = candidate_dir / "conflict.png"
            preview_path = candidate_dir / "preview.png"
            metadata_path = candidate_dir / "metadata.json"

            save_image(content_image, content_path)
            save_image(style_image, style_path)
            save_image(conflict_image, conflict_path)

            save_image(
                [content_image, style_image, conflict_image],
                preview_path,
                nrow=3,
            )

            metadata.update({
                "conflict_id": conflict_id,
                "pair": group["pair"],
                "pair_name": group["pair_name"],
                "direction": f"{content_class}->{style_class}",
                "direction_name": group["direction_name"],
                "candidate_order": candidate_order,
                "content_class_name": class_names[content_class],
                "style_class_name": class_names[style_class],
                "content_image_path": content_path.relative_to(
                    output_dir
                ).as_posix(),
                "style_image_path": style_path.relative_to(
                    output_dir
                ).as_posix(),
                "image_path": conflict_path.relative_to(
                    output_dir
                ).as_posix(),
                "preview_path": preview_path.relative_to(
                    output_dir
                ).as_posix(),
                "review_protocol": review_protocol["version"],
                "accepted": None,
                "rejection_reason": None,
                "selected_for_evaluation": False,
            })

            save_json(metadata_path, metadata)

            # The manifest indexes metadata rather than duplicating review decisions.
            records.append({
                "conflict_id": conflict_id,
                "metadata_path": metadata_path.relative_to(
                    output_dir
                ).as_posix(),
            })

            print(
                f"[{len(records)}/{expected_total}] "
                f"Saved {conflict_id}"
            )

    save_json(output_dir / "manifest.json", records)

    print("\nGeneration complete.")
    print("Candidates generated:", len(records))
    print("Output directory:", output_dir)
    print("Preview order: content | style | conflict")
    print("All candidates remain unreviewed.")
    print("Review each metadata.json before selecting inference images.")


if __name__ == "__main__":
    main()