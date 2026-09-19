# Assignment 1

## Task 1: Inductive biases and feature representations

STL-10, frozen backbones (torchvision ResNet-50 `IMAGENET1K_V2`, torchvision ViT-B/16
`IMAGENET1K_V1`, OpenCLIP ViT-B-32 `openai`), one linear head per backbone, plus
zero-shot CLIP with the prompt `a photo of a {class}.`. Seed 6304 throughout.

```
task1/
  config.py                 settings: seeds, optimizer, interventions, head files, paths
  utils.py                  data root (Google Drive on Colab, task1/local_runs locally)
  data/
    download.py             verified STL-10 download/extraction
    make_subset.py          500-image class-balanced test subset + grayscale/hue/patch inputs
    transforms.py           224x224 RGB inputs, grayscale, hue rotation, translation, 4x4 patch shuffle, AdaIN wrapper
    translation_dataset.py  12 translation conditions (8/16/32 px x 4 directions)
    make_cue_conflicts.py   one content/style stylization
    conflict_dataset.py     loader for the finalized cue-conflict selection
    conflict_dataset/       the 200 selected cue conflicts (PNG + JSON metadata)
    external/adain/         AdaIN network code (third party, see Attribution)
  models/backbones.py       frozen backbones; normalization is applied once, inside each wrapper
  scripts/
    main.py                 --mode train | infer
    conflict_image_generator.py  cue-conflict candidate generation
  analysis/
    metrics.py              top-1, macro-F1, confidence, paired consistency, cue counts/shape bias/coverage
    feature_similarity.py   cosine stability I_T and per-sample prediction/feature records
    representation.py       joint t-SNE per backbone
    report.py               builds every table and figure from saved inference results
  tests/smoke_test_pipeline.py   synthetic end-to-end check (no downloads)
```

### Running (from the `src/` folder that contains `assignment_01/`)

```bash
pip install torch torchvision open_clip_torch scikit-learn matplotlib tqdm pillow

# 1. Train the three linear heads (already done; saves model_weights/<model>_<val acc>.pth)
python -m assignment_01.task1.scripts.main --mode train

# 2. Inference on every condition: baseline, grayscale, hue, 12 translations,
#    patch shuffle and the 200 cue conflicts, for all four decision methods.
#    Each head is first re-checked on the validation split against the
#    accuracy in its filename (config.HEAD_WEIGHT_FILES).
python -m assignment_01.task1.scripts.main --mode infer \
    --conflict-dir <path to conflict_dataset>

#    Individual steps can be rerun: --steps baseline color translation patch conflict

# 3. Tables and figures -> <results>/analysis/
python -m assignment_01.task1.analysis.report --conflict-dir <path to conflict_dataset>

# Optional: synthetic end-to-end check of steps 2 and 3
python -m assignment_01.task1.tests.smoke_test_pipeline
```

On Colab the data root is `MyDrive/ATML/assignment_01/task1/`; `--conflict-dir` defaults to
`<data root>/conflict_dataset`.

### Outputs of `analysis/report.py`

| File | Content |
|---|---|
| `baseline_classification_report.csv` | clean top-1, macro-F1, mean max confidence (4 methods) |
| `intervention_comparison_compact.csv` / `_long.csv` | clean, grayscale, hue (30°) and patch shuffle: accuracy, change vs own baseline (pp), prediction consistency |
| `translation_directions.csv`, `translation_summary.csv`, `translation_curves.png`, `translation_directions.png` | accuracy and consistency vs displacement, per direction and averaged |
| `cue_conflict_report.csv`, `_directions.csv`, `_pairs.csv`, `_decisions.csv` | shape/texture/other counts, shape bias, coverage (pooled, per direction, per pair, per image) |
| `cue_conflict_dataset_audit.csv` | generated / selected / removed candidates per direction, alpha |
| `cue_conflict_examples.png` / `.csv` | seeded random examples of agreement, texture choices, disagreement and other-class failures |
| `clip_head_vs_zero_shot.csv` | CLIP head vs zero-shot on the same embeddings |
| `representation_stability.csv` / `.png`, `representation_samples.csv`, `prediction_vs_feature.csv` | cosine stability I_T per backbone and intervention; per-sample similarity next to prediction changes |
| `tsne_*.png`, `tsne_points.csv`, `tsne_settings.json` | one joint t-SNE per backbone (clean + grayscale, cue conflict, translation right 32 px, patch shuffle) |
| `evidence_summary.md`, `completeness.json` | key tables and a list of any missing results |

### Attribution

- `task1/data/external/adain/` (`net.py`, `function.py`) and the pretrained `vgg_normalised.pth` /
  `decoder.pth` weights are from Naoto Inoue's
  [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) (MIT License, copied in that folder),
  an implementation of Huang and Belongie (2017).
- Pretrained backbones: torchvision and [OpenCLIP](https://github.com/mlfoundations/open_clip).
