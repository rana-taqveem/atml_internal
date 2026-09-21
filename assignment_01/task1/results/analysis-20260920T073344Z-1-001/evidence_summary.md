# Task 1 evidence summary

Values come from the CSV files in this folder (shown with 2 decimals; the CSVs keep full precision).

## Clean baseline

| method | N | top-1 (%) | macro-F1 (%) | mean max confidence |
|---|---|---|---|---|
| ResNet-50 | 500 | 97.80 | 97.81 | 0.97 |
| ViT-B/16 | 500 | 98.80 | 98.80 | 0.97 |
| CLIP ViT-B/32 | 500 | 97.80 | 97.78 | 0.54 |
| CLIP ViT-B/32 zero-shot | 500 | 96.80 | 96.75 | 0.95 |

## Clean, color and patch-shuffle comparison (%)

| method | clean top-1 | gray top-1 | gray change (pp) | gray consistency | hue top-1 | hue change (pp) | hue consistency | patch top-1 | patch change (pp) | patch consistency |
|---|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 head | 97.80 | 96.40 | -1.40 | 97.00 | 96.20 | -1.60 | 97.00 | 90.00 | -7.80 | 91.80 |
| ViT-B/16 head | 98.80 | 96.60 | -2.20 | 96.60 | 98.00 | -0.80 | 98.40 | 92.60 | -6.20 | 92.80 |
| CLIP ViT-B/32 head | 97.80 | 95.40 | -2.40 | 95.60 | 97.80 | 0.00 | 97.40 | 84.00 | -13.80 | 83.60 |
| CLIP ViT-B/32 zero-shot | 96.80 | 94.40 | -2.40 | 96.80 | 95.80 | -1.00 | 98.40 | 84.00 | -12.80 | 84.40 |

## Translation (mean over four directions)

| method | px | accuracy (%) | consistency (%) | min acc | max acc |
|---|---|---|---|---|---|
| ResNet-50 | 0 | 97.80 | 100.00 | 97.80 | 97.80 |
| ResNet-50 | 8 | 97.40 | 99.25 | 97.20 | 97.60 |
| ResNet-50 | 16 | 97.00 | 98.80 | 96.60 | 97.20 |
| ResNet-50 | 32 | 97.25 | 98.60 | 97.00 | 97.80 |
| ViT-B/16 | 0 | 98.80 | 100.00 | 98.80 | 98.80 |
| ViT-B/16 | 8 | 98.55 | 99.10 | 98.20 | 98.80 |
| ViT-B/16 | 16 | 98.65 | 99.60 | 98.40 | 99.00 |
| ViT-B/16 | 32 | 98.30 | 98.85 | 98.00 | 98.40 |
| CLIP ViT-B/32 | 0 | 97.80 | 100.00 | 97.80 | 97.80 |
| CLIP ViT-B/32 | 8 | 97.65 | 97.60 | 97.40 | 98.00 |
| CLIP ViT-B/32 | 16 | 97.40 | 97.10 | 97.00 | 97.80 |
| CLIP ViT-B/32 | 32 | 97.10 | 97.80 | 96.60 | 97.60 |
| CLIP ViT-B/32 zero-shot | 0 | 96.80 | 100.00 | 96.80 | 96.80 |
| CLIP ViT-B/32 zero-shot | 8 | 95.70 | 97.50 | 94.60 | 96.80 |
| CLIP ViT-B/32 zero-shot | 16 | 95.20 | 96.55 | 94.40 | 95.80 |
| CLIP ViT-B/32 zero-shot | 32 | 95.30 | 97.35 | 95.00 | 96.00 |

## Cue conflicts (pooled over all selected conflicts)

| method | N | shape | texture | other | shape bias (%) | coverage (%) |
|---|---|---|---|---|---|---|
| ResNet-50 | 200 | 139 | 21 | 40 | 86.88 | 80.00 |
| ViT-B/16 | 200 | 179 | 4 | 17 | 97.81 | 91.50 |
| CLIP ViT-B/32 | 200 | 171 | 11 | 18 | 93.96 | 91.00 |
| CLIP ViT-B/32 zero-shot | 200 | 172 | 8 | 20 | 95.56 | 90.00 |

## Cue-conflict dataset audit

| content | style | generated | selected | removed | removed w/ reason | alpha |
|---|---|---|---|---|---|---|
| airplane | bird | 25 | 20 | 5 | 5 | 0.7 |
| bird | airplane | 25 | 20 | 5 | 5 | 0.7 |
| bird | car | 25 | 20 | 5 | 5 | 0.7 |
| bird | cat | 25 | 20 | 5 | 5 | 0.7 |
| car | bird | 25 | 20 | 5 | 5 | 0.7 |
| car | deer | 25 | 20 | 5 | 5 | 0.7 |
| cat | bird | 25 | 20 | 5 | 5 | 0.7 |
| deer | car | 25 | 20 | 5 | 5 | 0.7 |
| monkey | truck | 25 | 20 | 5 | 5 | 0.7 |
| truck | monkey | 25 | 20 | 5 | 5 | 0.7 |
| ALL | ALL | 250 | 200 | 50 | 50 | 0.7 |

## Texture class below top-1 (clean content image vs cue conflict)

| method | mean rank clean | mean rank conflict | top-2 clean (%) | top-2 conflict (%) | runner-up | shape, clean (%) | runner-up | shape, conflict (%) |
|---|---|---|---|---|---|---|
| ResNet-50 | 6.50 | 5.18 | 12.50 | 27.00 | 9.35 | 17.27 |
| ViT-B/16 | 6.10 | 5.24 | 13.50 | 20.50 | 11.73 | 19.55 |
| CLIP ViT-B/32 | 6.78 | 6.03 | 5.50 | 17.50 | 5.26 | 12.87 |
| CLIP ViT-B/32 zero-shot | 5.96 | 5.60 | 12.00 | 12.50 | 11.05 | 8.72 |

## Paired tests: interventions vs own baseline (Holm-corrected per family)

| method | condition | change (pp) | CI low | CI high | p (Holm) | sig. |
|---|---|---|---|---|---|---|
| ResNet-50 | grey_scale | -1.40 | -2.91 | 0.11 | 0.36 | False |
| ResNet-50 | hue | -1.60 | -3.06 | -0.14 | 0.29 | False |
| ResNet-50 | patch_shuffle | -7.80 | -10.15 | -5.45 | 0.00 | True |
| ViT-B/16 | grey_scale | -2.20 | -3.80 | -0.60 | 0.09 | False |
| ViT-B/16 | hue | -0.80 | -1.91 | 0.31 | 0.58 | False |
| ViT-B/16 | patch_shuffle | -6.20 | -8.45 | -3.95 | 0.00 | True |
| CLIP ViT-B/32 | grey_scale | -2.40 | -4.23 | -0.57 | 0.10 | False |
| CLIP ViT-B/32 | hue | 0.00 | -1.36 | 1.36 | 1.00 | False |
| CLIP ViT-B/32 | patch_shuffle | -13.80 | -17.11 | -10.49 | 0.00 | True |
| CLIP ViT-B/32 zero-shot | grey_scale | -2.40 | -3.85 | -0.95 | 0.01 | True |
| CLIP ViT-B/32 zero-shot | hue | -1.00 | -1.87 | -0.13 | 0.29 | False |
| CLIP ViT-B/32 zero-shot | patch_shuffle | -12.80 | -15.98 | -9.62 | 0.00 | True |

## Cue-conflict tests (Holm-corrected per family)

| family | comparison | method | effect (pp) | CI low | CI high | p (Holm) | sig. |
|---|---|---|---|---|---|---|---|
| texture_rank | texture class: conflict vs clean content image | ResNet-50 | 14.50 | 8.40 | 20.60 | 0.00 | True |
| texture_rank | texture class: conflict vs clean content image | ViT-B/16 | 7.00 | 1.54 | 12.46 | 0.04 | True |
| texture_rank | texture class: conflict vs clean content image | CLIP ViT-B/32 | 12.00 | 7.09 | 16.91 | 0.00 | True |
| texture_rank | texture class: conflict vs clean content image | CLIP ViT-B/32 zero-shot | 0.50 | -4.40 | 5.40 | 1.00 | False |
| shape_bias_model_vs_model | shape bias: vit_b_16 - resnet50 | resnet50 vs vit_b_16 | 10.94 | 6.16 | 16.17 | 0.00 | True |
| shape_bias_model_vs_model | shape bias: clip_vit_b_32 - resnet50 | resnet50 vs clip_vit_b_32 | 7.08 | 1.88 | 12.57 | 0.03 | True |
| shape_bias_model_vs_model | shape bias: clip_vit_b_32_zero_shot - resnet50 | resnet50 vs clip_vit_b_32_zero_shot | 8.68 | 3.45 | 14.22 | 0.01 | True |
| shape_bias_model_vs_model | shape bias: clip_vit_b_32 - vit_b_16 | vit_b_16 vs clip_vit_b_32 | -3.86 | -7.55 | -0.55 | 0.06 | False |
| shape_bias_model_vs_model | shape bias: clip_vit_b_32_zero_shot - vit_b_16 | vit_b_16 vs clip_vit_b_32_zero_shot | -2.26 | -5.44 | 0.55 | 0.25 | False |
| shape_bias_model_vs_model | shape bias: clip_vit_b_32_zero_shot - clip_vit_b_32 | clip_vit_b_32 vs clip_vit_b_32_zero_shot | 1.60 | -1.08 | 4.45 | 0.26 | False |

## Representation stability I_T (cosine)

| backbone | condition | N | I_T |
|---|---|---|---|
| ResNet-50 | grey_scale | 500 | 0.80 |
| ResNet-50 | hue | 500 | 0.90 |
| ResNet-50 | patch_shuffle | 500 | 0.70 |
| ResNet-50 | cue_conflict | 200 | 0.44 |
| ResNet-50 | translation_8px_mean | 2000 | 0.97 |
| ResNet-50 | translation_16px_mean | 2000 | 0.95 |
| ResNet-50 | translation_32px_mean | 2000 | 0.93 |
| ViT-B/16 | grey_scale | 500 | 0.74 |
| ViT-B/16 | hue | 500 | 0.88 |
| ViT-B/16 | patch_shuffle | 500 | 0.63 |
| ViT-B/16 | cue_conflict | 200 | 0.48 |
| ViT-B/16 | translation_8px_mean | 2000 | 0.95 |
| ViT-B/16 | translation_16px_mean | 2000 | 0.98 |
| ViT-B/16 | translation_32px_mean | 2000 | 0.95 |
| CLIP ViT-B/32 | grey_scale | 500 | 0.89 |
| CLIP ViT-B/32 | hue | 500 | 0.95 |
| CLIP ViT-B/32 | patch_shuffle | 500 | 0.75 |
| CLIP ViT-B/32 | cue_conflict | 200 | 0.78 |
| CLIP ViT-B/32 | translation_8px_mean | 2000 | 0.98 |
| CLIP ViT-B/32 | translation_16px_mean | 2000 | 0.96 |
| CLIP ViT-B/32 | translation_32px_mean | 2000 | 0.97 |

