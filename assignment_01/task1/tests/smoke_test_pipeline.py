"""End-to-end synthetic smoke test of Task 1 inference and reporting.

Runs the real start_inference() and report.run_all() code paths, but with
  * a fake STL-10 (deterministic random 96x96 images, label = index % 10),
  * tiny stub backbones with the real feature sizes (2048 / 768 / 512),
  * the real finalized cue-conflict folder and the real 500 selected test IDs.

It verifies wiring, file naming, ID alignment and every report section; the
numbers it produces are meaningless. No downloads are needed.

    python -m assignment_01.task1.tests.smoke_test_pipeline [--work-dir DIR]
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from assignment_01.task1.config import task_config

TASK1_DIR = Path(__file__).resolve().parents[1]
REAL_CONFLICT_DIR = Path(task_config.TASK_CONFLICT_DATASET_DIR)


class FakeSTL10:
    """Stand-in for torchvision STL10 with the same indexing and fields."""

    classes = list(task_config.STL10_CLASSES)

    def __init__(self, root=None, split="test", download=False, transform=None):
        self.split = split
        self.transform = transform
        size = 8000 if split == "test" else 5000
        self.labels = np.arange(size) % 10

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        rng = np.random.default_rng(index + (0 if self.split == "test" else 10_000))
        base = rng.integers(0, 256, size=(12, 12, 3), dtype=np.uint8)
        image = Image.fromarray(base).resize((96, 96), Image.NEAREST)
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[index])


class StubBackbone(nn.Module):
    """Small fixed conv net: [B,3,224,224] -> [B, dim]."""

    def __init__(self, dim, seed, normalize=False):
        super().__init__()
        torch.manual_seed(seed)
        self.conv = nn.Conv2d(3, 32, kernel_size=8, stride=8)
        self.proj = nn.Linear(32 * 4, dim)
        self.normalize = normalize
        self.requires_grad_(False)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.adaptive_avg_pool2d(x, 2).flatten(1)
        x = self.proj(x)
        return F.normalize(x, dim=-1) if self.normalize else x


class StubClipModel(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(3)
        self.token_embedding = nn.Embedding(49408, 512)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(100.0), dtype=torch.float32))
        self.requires_grad_(False)

    def encode_text(self, tokens):
        return self.token_embedding(tokens).mean(dim=1)


def make_stub(dim, seed, clip=False):
    def factory():
        backbone = StubBackbone(dim, seed, normalize=clip)
        if clip:
            backbone.model = StubClipModel()
        return backbone
    return factory


def point_config_to(work_dir):
    task_config.TASK_DIR = str(work_dir)
    for name, sub in [("TASK_CHECKPOINTS_DIR", "checkpoints"), ("TASK_FEATURES_DIR", "features"),
                      ("TASK_DATASET_DIR", "dataset"), ("TASK_RESULTS_DIR", "results"),
                      ("TASK_CONFLICT_DATASET_DIR", "conflict_dataset"),
                      ("MODEL_WEIGHTS_DIR", "model_weights"), ("CUE_CONFLICT_DIR", "cue_conflict")]:
        setattr(task_config, name, str(work_dir / sub))
    task_config.SELECTED_INDICES_FILE = str(work_dir / "selected_indices.npy")
    task_config.BATCH_SIZE = 50


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"  ok  {message}")


def run(work_dir, conflict_dir=REAL_CONFLICT_DIR):
    point_config_to(work_dir)
    task_config.init_env()

    from assignment_01.task1.data import make_subset
    from assignment_01.task1.scripts import main
    from assignment_01.task1.analysis import report

    for module in (make_subset, main):
        module.datasets.STL10 = FakeSTL10
        module.prepare_stl10 = lambda root: None

    main.Resnet50Backbone = make_stub(2048, 0)
    main.Torchvision_Vit_B_16_Backbone = make_stub(768, 1)
    main.Openai_Clip_Backbone = make_stub(512, 2, clip=True)

    from assignment_01.task1.data.download import prepare_conflict_dataset
    conflict_dir = prepare_conflict_dataset(conflict_dir)
    if not (conflict_dir / "sampling_plan.json").is_file():
        raise FileNotFoundError(
            f"No sampling_plan.json under {conflict_dir}, and no conflict_dataset.zip to extract there. "
            "Pass --conflict-dir pointing at your conflict_dataset folder (extracted, or containing "
            "conflict_dataset.zip)."
        )

    # Use the real 500 test IDs so conflicts pair with clean content rows.
    plan = json.loads((conflict_dir / "sampling_plan.json").read_text(encoding="utf-8"))
    np.save(task_config.SELECTED_INDICES_FILE, np.array(plan["selected_image_ids"], dtype=np.int64))

    # Random heads, saved under names carrying their own validation accuracy.
    _, val_loader = main.get_train_val_dataloaders()
    head_files = {}
    for model_name, factory in [(task_config.RESNET50, main.Resnet50Backbone),
                                (task_config.VIT_B_16, main.Torchvision_Vit_B_16_Backbone),
                                (task_config.CLIP_VIT_B_32, main.Openai_Clip_Backbone)]:
        backbone = factory()
        torch.manual_seed(len(model_name))
        head = nn.Linear(backbone.proj.out_features, 10)
        _, val_acc = main.evaluate(nn.Sequential(backbone, head).to(main.DEVICE), val_loader, nn.CrossEntropyLoss())
        head_files[model_name] = f"{model_name}_{val_acc:.2f}.pth"
        torch.save(head.state_dict(), Path(task_config.MODEL_WEIGHTS_DIR) / head_files[model_name])
    task_config.HEAD_WEIGHT_FILES = head_files

    print("\n== Intervention invariants")
    baseline, _ = make_subset.get_test_subset("baseline")
    patch_a, patch_meta = make_subset.get_test_subset("patch_shuffle")
    patch_b, _ = make_subset.get_test_subset("patch_shuffle")
    grey, _ = make_subset.get_test_subset("grey_scale")
    clean_images, patch_images = baseline.tensors[0], patch_a.tensors[0]
    check(torch.equal(patch_images, patch_b.tensors[0]), "patch shuffle is reproducible (seed 6304)")
    check(all(torch.equal(clean_images[i].flatten().sort().values, patch_images[i].flatten().sort().values)
              for i in range(len(clean_images))), "patch shuffle preserves every pixel value")
    check(all(not torch.equal(p, torch.arange(16)) for p in patch_meta["permutations"]),
          "no identity permutations")
    check(torch.equal(grey.tensors[0][:, 0], grey.tensors[0][:, 1]), "grayscale keeps 3 equal channels")
    check(report.check_zero_translation_identity(), "zero translation returns the input")
    del baseline, patch_a, patch_b, grey

    print("\n== Inference")
    main.start_inference(conflict_dir=str(conflict_dir), check_heads=True)

    result_files = sorted(p.name for p in Path(task_config.TASK_RESULTS_DIR).glob("*_results.pt"))
    check(len(result_files) == 4 * 17, f"68 result files written (4 methods x 17 conditions), got {len(result_files)}")

    print("\n== Report")
    output_dir = Path(task_config.TASK_RESULTS_DIR) / "analysis"
    completeness = report.run_all(task_config.TASK_RESULTS_DIR, conflict_dir, output_dir)
    check(not completeness["skipped_sections"], "no report section skipped")
    check(not completeness["missing_result_files"], "no missing result files")

    translation = list(__import__("csv").DictReader((output_dir / "translation_summary.csv").open()))
    zero = [r for r in translation if r["displacement"] == "0"]
    check(len(zero) == 4 and all(float(r["mean_consistency_pct"]) == 100.0 for r in zero),
          "0px translation has 100% consistency for every method")

    cue = list(__import__("csv").DictReader((output_dir / "cue_conflict_report.csv").open()))
    check(len(cue) == 4 and all(int(r["n_total"]) == 200 for r in cue), "cue report: 4 methods x 200 conflicts")
    check(all(int(r["n_shape"]) + int(r["n_texture"]) + int(r["n_other"]) == 200 for r in cue),
          "shape + texture + other = 200 for every method")

    for name in ["intervention_comparison_compact.csv", "translation_curves.png", "cue_conflict_examples.png",
                 "cue_conflict_dataset_audit.csv", "representation_stability.csv", "tsne_all_backbones.png",
                 "clip_head_vs_zero_shot.csv", "prediction_vs_feature.csv", "evidence_summary.md"]:
        check((output_dir / name).is_file(), f"wrote {name}")

    print(f"\nSmoke test passed. Outputs: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default=None, help="Keep outputs here (default: temporary folder).")
    parser.add_argument("--conflict-dir", default=str(REAL_CONFLICT_DIR),
                        help="Your extracted conflict_dataset folder (must contain sampling_plan.json). "
                             "Default assumes it lives in the repo at data/conflict_dataset, which is "
                             "only true locally; on Colab pass the path to your unzipped Drive copy.")
    args = parser.parse_args()

    if args.work_dir:
        work = Path(args.work_dir)
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        run(work, conflict_dir=args.conflict_dir)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            run(Path(tmp), conflict_dir=args.conflict_dir)
