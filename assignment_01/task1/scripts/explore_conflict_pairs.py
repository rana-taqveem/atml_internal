"""Generate ten candidates for every ordered STL-10 class pair.

This is exploratory data for visual pair selection, not the final evaluation
set. No classifier predictions or automatic acceptance decisions are used.

The same content examples are reused across style classes for comparability.
Each class also supplies a fixed, independently sampled set of style images.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from torchvision.datasets import STL10
from torchvision.utils import save_image

from assignment_01.task1.config import task_config
from assignment_01.task1.data.download import prepare_stl10
from assignment_01.task1.data.make_subset import get_test_subset
from assignment_01.task1.data.make_cue_conflicts import generate_cue_conflicts
from assignment_01.task1.data.transforms import (
    StyleTransferModel,
    apply_universal_transforms,
)


def save_json(path, data):
    """Replace a JSON file atomically after writing its temporary copy."""
    temporary_path = path.with_suffix(".json.tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    temporary_path.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--images-per-direction", type=int, default=10)
    parser.add_argument("--run-version", default="v1")
    args = parser.parse_args()

    if not 0 < args.alpha <= 1:
        raise ValueError("Alpha must be greater than 0 and at most 1.")

    if args.images_per_direction < 1:
        raise ValueError("Images per direction must be positive.")

    task_config.init_env()

    alpha_tag = str(args.alpha).replace(".", "_")
    output_dir = (
        Path(task_config.TASK_CONFLICT_DATASET_DIR)
        / f"all_pairs_alpha_{alpha_tag}_{args.run_version}"
    )

    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}\n"
            "Use another --run-version to avoid overwriting your review."
        )

    task_dir = Path(__file__).resolve().parents[1]
    weights_dir = task_dir / "data" / "external" / "model_weights"

    encoder_path = weights_dir / "vgg_normalised.pth"
    decoder_path = weights_dir / "decoder.pth"

    for path in (encoder_path, decoder_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing weights: {path}")

    prepare_stl10(task_config.TASK_DATASET_DIR)

    indices_path = Path(task_config.SELECTED_INDICES_FILE)
    if not indices_path.is_file():
        get_test_subset(transformation_type="original")

    dataset = STL10(
        root=task_config.TASK_DATASET_DIR,
        split="test",
        download=False,
        transform=apply_universal_transforms,
    )

    selected_ids = np.load(indices_path, allow_pickle=False)

    if selected_ids.ndim != 1 or selected_ids.size == 0:
        raise ValueError("Selected IDs must be a nonempty 1D array.")

    if not np.issubdtype(selected_ids.dtype, np.integer):
        raise ValueError("Selected IDs must be integers.")

    if np.unique(selected_ids).size != selected_ids.size:
        raise ValueError("Selected IDs contain duplicates.")

    if selected_ids.min() < 0 or selected_ids.max() >= len(dataset):
        raise ValueError("Selected IDs are outside the test dataset.")

    class_names = list(task_config.STL10_CLASSES)

    if list(dataset.classes) != class_names:
        raise ValueError("Configured classes do not match dataset class order.")

    rng = np.random.default_rng(task_config.SEED)
    content_ids_by_class = {}
    style_ids_by_class = {}

    for class_id in range(len(class_names)):
        available_ids = [
            int(image_id)
            for image_id in selected_ids
            if int(dataset.labels[image_id]) == class_id
        ]

        if len(available_ids) < args.images_per_direction:
            raise ValueError(
                f"Not enough selected images for {class_names[class_id]}."
            )

        content_ids_by_class[class_id] = rng.choice(
            available_ids,
            size=args.images_per_direction,
            replace=False,
        ).tolist()

        style_ids_by_class[class_id] = rng.choice(
            available_ids,
            size=args.images_per_direction,
            replace=False,
        ).tolist()

    groups = []

    for content_class in range(len(class_names)):
        for style_class in range(len(class_names)):
            if content_class == style_class:
                continue

            groups.append({
                "content_class": content_class,
                "style_class": style_class,
                "combinations": list(zip(
                    content_ids_by_class[content_class],
                    style_ids_by_class[style_class],
                )),
            })

    stylizer = StyleTransferModel(
        encoder_path=encoder_path,
        decoder_path=decoder_path,
    )

    output_dir.mkdir(parents=True, exist_ok=False)

    save_json(output_dir / "sampling_plan.json", {
        "purpose": "visual_pair_exploration",
        "seed": int(task_config.SEED),
        "alpha": args.alpha,
        "classes": class_names,
        "images_per_direction": args.images_per_direction,
        "selected_image_ids": selected_ids.tolist(),
        "content_ids_by_class": content_ids_by_class,
        "style_ids_by_class": style_ids_by_class,
        "encoder_path": str(encoder_path),
        "decoder_path": str(decoder_path),
        "groups": groups,
    })

    save_json(output_dir / "review_protocol.json", {
        "purpose": "Compare visual feasibility of class pairs in both directions.",
        "accept_only_if": [
            "The source content and style are visually interpretable.",
            "The generated content object remains recognizable.",
            "Style texture visibly transfers to the content object; "
            "color change alone is insufficient.",
            "Severe artifacts or background-only transfer do not dominate.",
        ],
        "rejection_reasons": [
            "source_ambiguous",
            "shape_lost",
            "weak_texture",
            "background_dominates",
            "severe_artifacts",
            "ambiguous",
        ],
        "model_predictions_used": False,
    })

    manifest = []
    total = len(groups) * args.images_per_direction

    for group in groups:
        content_class = group["content_class"]
        style_class = group["style_class"]

        content_name = class_names[content_class]
        style_name = class_names[style_class]

        direction_dir = (
            output_dir
            / f"{content_name}_content"
            / f"{style_name}_style"
        )

        for index, (content_id, style_id) in enumerate(group["combinations"]):
            candidate_id = (
                f"c{content_class}_s{style_class}"
                f"_{index:03d}_{content_id}_{style_id}"
            )

            candidate_dir = direction_dir / candidate_id
            candidate_dir.mkdir(parents=True, exist_ok=False)

            conflict_image, metadata = generate_cue_conflicts(
                dataset=dataset,
                content_id=content_id,
                style_id=style_id,
                stylizer=stylizer,
                alpha=args.alpha,
            )

            content_image, _ = dataset[content_id]
            style_image, _ = dataset[style_id]

            paths = {
                "content_image_path": candidate_dir / "content.png",
                "style_image_path": candidate_dir / "style.png",
                "image_path": candidate_dir / "conflict.png",
                "preview_path": candidate_dir / "preview.png",
            }

            save_image(content_image, paths["content_image_path"])
            save_image(style_image, paths["style_image_path"])
            save_image(conflict_image, paths["image_path"])

            save_image(
                [content_image, style_image, conflict_image],
                paths["preview_path"],
                nrow=3,
            )

            metadata.update({
                "conflict_id": candidate_id,
                "purpose": "visual_pair_exploration",
                "pair": sorted([content_class, style_class]),
                "direction": f"{content_class}->{style_class}",
                "candidate_order": index,
                "content_class_name": content_name,
                "style_class_name": style_name,
                "selected_for_evaluation": False,
                **{
                    key: path.relative_to(output_dir).as_posix()
                    for key, path in paths.items()
                },
            })

            metadata_path = candidate_dir / "metadata.json"
            save_json(metadata_path, metadata)

            manifest.append({
                "conflict_id": candidate_id,
                "metadata_path": metadata_path.relative_to(
                    output_dir
                ).as_posix(),
            })

            # Preserve an index of completed candidates during long runs.
            save_json(output_dir / "manifest.json", manifest)

            print(
                f"[{len(manifest)}/{total}] "
                f"{content_name} content <- {style_name} style: "
                f"{candidate_id}",
                flush=True,
            )

    print("\nExploration complete.")
    print("Generated candidates:", len(manifest))
    print("Output directory:", output_dir)
    print("Preview order: content | style | conflict")
    print("All candidates remain unreviewed.")


if __name__ == "__main__":
    main()