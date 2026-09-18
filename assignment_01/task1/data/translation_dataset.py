from torch.utils.data import Dataset

from assignment_01.task1.data.transforms import apply_translation


def get_translation_conditions():
    """Define four directions at each nonzero displacement."""
    directions = {
        "right": (1, 0),
        "left": (-1, 0),
        "down": (0, 1),
        "up": (0, -1),
    }

    conditions = []

    for displacement in (8, 16, 32):
        for direction, (unit_x, unit_y) in directions.items():
            conditions.append({
                "condition_id": f"translation_{direction}_{displacement}",
                "displacement": displacement,
                "direction": direction,
                "translation_x": displacement * unit_x,
                "translation_y": displacement * unit_y,
            })

    return conditions


class TranslationDataset(Dataset):
    """Apply one fixed translation to an existing baseline dataset.

    The baseline dataset must return (image, label, image_id), where image
    is an unnormalized tensor with shape [C, H, W].

    Translation preserves the label and image ID. Images are transformed
    when accessed, so the complete translated dataset is not stored.
    """

    def __init__(self, baseline_dataset, translate_x, translate_y):
        for value in (translate_x, translate_y):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("Translation offsets must be integers.")

            if abs(value) > 32:
                raise ValueError("Translation offsets must be within [-32, 32].")

        self.baseline_dataset = baseline_dataset
        self.translate_x = translate_x
        self.translate_y = translate_y

    def __len__(self):
        return len(self.baseline_dataset)

    def __getitem__(self, index):
        image, label, image_id = self.baseline_dataset[index]

        translated_image = apply_translation(
            image,
            self.translate_x,
            self.translate_y,
        )

        return translated_image, label, image_id