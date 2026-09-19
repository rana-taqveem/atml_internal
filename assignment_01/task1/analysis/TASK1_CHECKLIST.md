# Task 1 implementation and evidence checklist

Source: `assignment_01/ATML-PA1.pdf`, Task 1, pages 2-5; shared submission rules also apply.
Last reviewed: 2026-09-19. Companion explanations: [study README](README.md).

Checked items mean the stated implementation step was observed, not that the final experiment is complete. Unchecked items can have partial code. Mark final evidence complete only after checking saved results. Training completion below is based on previously reviewed Colab logs; checkpoint files were not revalidated in this review.

## Status 2026-09-19: code complete, awaiting the Colab inference run

Ticked items are implemented and exercised by `tests/smoke_test_pipeline.py` (fake STL-10, stub backbones, real conflict folder and real 500 IDs). The smoke test passes end to end: 68 result files, every report section, 0 px = 100% consistency, shape+texture+other = 200. Evidence items (sections C/D "Run ...", tables, curves, interpretation) stay open until `main.py --mode infer` and `analysis/report.py` run on Colab with the real models.

Found in review:
- `checkpoints/*_head.pth` (local copies in results/baseline) are from an earlier run: 95.70 / 96.90 / 97.00% on the cached validation features. `model_weights/{resnet50_97.30, vit_b_16_97.90, clip_vit_b_32_98.40}.pth` reproduce the logged accuracies exactly. Inference now loads those files (config.HEAD_WEIGHT_FILES) and re-checks each head on the validation split before evaluating.
- Fixed: patch_size 566 -> 56 and nonexistent shuffle_patches call; missing get_translation_dataset import; undefined TASK_CONFLICTS_DIR; double forward pass in test_and_report; numpy ints in saved IDs (weights_only loading).
- Still needed from the author: rejection status/reasons for the 50 removed candidates (fill `rejection_reasons_TEMPLATE.csv` from the report output, save it as `conflict_dataset/rejection_reasons.csv`); hypotheses/metrics for each experimental choice; the t-SNE condition set (default: grayscale, cue conflict, translation right 32 px, patch shuffle) is a default, change `representation.DEFAULT_CONDITIONS` if preferred.

## A. Overall experimental setup

- [x] Select STL-10 as the dataset.
- [x] Verify the official training partition is split stratified 80/20 with seed 6304; preserve split identifiers.
- [x] Verify all required pretrained backbones and frozen final representations: ResNet-50 IMAGENET1K_V2 global-average-pooled features; ViT-B/16 IMAGENET1K_V1 final class token; OpenCLIP ViT-B-32 pretrained=openai normalized image embedding.
- [x] Verify head training settings: linear head, seed 6304, AdamW lr=1e-3, weight_decay=1e-4, maximum 50 epochs, early stopping after five epochs without improved validation accuracy.
- [x] Complete training runs and save three heads (previously reviewed Colab logs; CLIP activation fix followed by retraining).
- [x] Verify inference loads the intended best head checkpoints and matching backbone/cache versions; use eval mode and disabled gradients.
- [ ] Verify saved test selection: 500 official test images, class-balanced, seed 6304, stable IDs; document any shortage/imbalance.
- [x] Verify common 224x224 RGB intervention inputs and exactly one application of model-specific normalization.
- [x] Verify exact baseline/transformed image reuse across all methods, including randomized patch permutations.
- [x] Record dataset, class order, IDs, seeds, transformation settings, model/pretraining identity, head checkpoint and software environment.
- [ ] State hypotheses and metrics for experimental choices before interpreting outcomes: dataset, cue-conflict pairs/style strength, additional color intervention, representation visualization/settings.
- [x] Implement zero-shot CLIP text/scoring helpers in main.py.
- [x] Verify zero-shot uses fixed prompt `a photo of a {class}.`, normalized embeddings and scaled class similarities; no prompt search or fine-tuning.
- [ ] Evaluate all four decision methods: ResNet head, ViT head, CLIP head and zero-shot CLIP.

## B. Analysis implementation checklist

### B1. Standalone classification metrics

- [x] Implement top-1 accuracy, macro-F1 and mean maximum confidence.
- [x] Add supplementary per-class precision/recall/F1/support, macro precision/recall and configurable top-k.
- [x] Verify baseline metrics on a known synthetic example (2026-09-18): top-1 70%, macro-F1 71.1111%, mean maximum confidence 0.8482255; supplementary top-2 100%.
- [ ] Verify invalid inputs and edge cases after validation cleanup.
- [x] Clean up duplicated k validation; validate integer k and labels before casting; document consistent output units.
- [x] Connect metric calculation to saved inference results and export machine-readable tables.

### B2. baseline versus transformed comparison

- [x] Validate unique, nonempty, one-dimensional image IDs and matching ID sets.
- [x] Align labels/logits using each run's sorted ID indices.
- [x] Check class-order agreement, score row/column counts and aligned true labels.
- [x] Implement baseline/transformed accuracy, signed change/drop in percentage points, prediction consistency.
- [ ] Verify reordered IDs give identical answers; reject duplicates, missing IDs, incompatible labels and class orders.
- [x] Enforce or explicitly verify matching model, decision method and checkpoint in the calling layer.
- [x] Update stale comparison docstring: import syntax and labels.shape[1] issues have now been corrected.

### B3. Shape-versus-texture analysis — next lesson

- [x] Define result records with unique conflict ID, content/style IDs, content/shape label, style/texture label, accepted status, rejection reason, alpha and model predictions.
- [x] Validate accepted examples have distinct valid shape and texture labels and aligned predictions.
- [x] Count shape predictions, texture predictions and other predictions; verify their sum equals the number of evaluated conflicts (synthetic checks).
- [x] Compute shape bias (%) = 100*N_shape/(N_shape+N_texture).
- [x] Compute coverage (%) = 100*(N_shape+N_texture)/N_total.
- [x] Report undefined shape bias when N_shape+N_texture=0; reject empty evaluation sets.
- [x] Report generation accepted/rejected counts separately from model decision counts.
- [ ] Check toy cases, then evaluate the same accepted conflicts with all four methods.

### B4. Translation aggregation

- [x] Use paired comparisons for displacement 0, 8, 16 and 32 pixels in four cardinal directions.
- [x] Average accuracy and consistency across directions at each displacement; preserve individual direction results.
- [x] Plot both accuracy and consistency versus displacement (step 4 requests both).
- [x] Verify zero displacement reproduces baseline accuracy and 100% prediction consistency.

### B5. Representation analysis

- [x] Align transformed features with baseline counterparts using image/content IDs; handle repeated content images in cue-conflict records explicitly.
- [x] Calculate per-example cosine similarity and its mean for grayscale, cue conflict, translation and patch shuffle for each backbone.
- [x] Validate feature dimensions/finite values and define handling of zero-norm vectors.
- [x] Choose t-SNE or UMAP and record settings, fixed subset and seed.
- [x] Fit one joint baseline/transformed 2D projection per backbone; color by ground-truth/content class and mark condition separately.
- [x] Do not compare absolute coordinates between separately fitted backbones or treat 2D distances as exact original distances.
- [ ] Inspect class separation, condition mixing and transformed examples moving from baseline clusters.

## C. Intervention generation and final experimental evidence

### baseline and color

- [x] Inference code includes baseline, grayscale and fixed 30-degree hue conditions.
- [ ] Verify grayscale preserves three channels and hue changes color while preserving geometry; document what is changed/preserved.
- [ ] Run baseline evaluation and save top-1/macro-F1/confidence for all four methods; zero-shot confidence uses scaled similarities.
- [ ] Run grayscale and hue and save absolute performance, accuracy changes and paired consistency against each method's own baseline baseline.

### Cue conflicts

- [x] Single-conflict generation helper exists with content/style IDs, labels, alpha, accepted status and rejection reason.
- [ ] Verify pretrained stylizer/weights, output quality and reproducible generation.
- [x] Configure five unordered class pairs in CLASS_PAIRS; IDs match STL10_CLASSES.
- [x] Generate both directions for each configured pair, targeting 20 accepted examples per direction for 200 total.
- [x] Correct StyleTransferModel input shape check: compare the full image.shape with (3,224,224), not image.shape[1].
- [x] Define a visual rejection rule before evaluating models; recorded in CUE_CONFLICT_REVIEW.md. Application and final counts remain pending.
- [x] Produce at least 200 valid conflicts, balanced across pairs/directions as closely as possible.
- [ ] Save accepted/rejected totals and per-pair/direction counts with reasons, settings and image identifiers.
- [x] Integrate accepted-conflict inference for all four methods and produce counts, shape bias and coverage.
- [ ] Select informative agreements, disagreements or failures and show model predictions with images.

### Translation and patch structure

- [x] Verify translation implementation uses reflection padding followed by shifted crop.
- [x] Expand inference beyond the currently configured (8,0) translation to the full displacement/direction grid; avoid output filenames overwriting settings.
- [x] Verify patch intervention uses 4x4 cells (56x56 pixels at 224x224), preserves pixels and rejects identity permutations with seed 6304.
- [x] Generate one fixed permutation per image and reuse identical shuffled images across methods.
- [ ] Evaluate patch shuffle and report absolute accuracy, drop and consistency.

## D. Required outputs and interpretation

- [ ] Compact comparison table: baseline, grayscale, additional color and patch shuffle.
- [ ] Cue-conflict table: shape/texture/other counts, shape bias, coverage, accepted/rejected counts.
- [ ] Translation accuracy/consistency curves.
- [ ] Representation stability results for every required intervention and joint t-SNE/UMAP plots.
- [ ] Informative qualitative cue-conflict examples with predictions.
- [ ] Discuss what color and cue conflicts jointly suggest about shape/texture/color reliance, including coverage limitations.
- [ ] Discuss translation/patch responses, locality, spatial organization and positional sensitivity.
- [ ] Discuss at least one agreement or mismatch between prediction changes and feature changes; compare CLIP head versus zero-shot.
- [ ] Distinguish plausible architecture effects from confounds: pretraining data, supervision, augmentation, capacity and positional encoding.
- [ ] Avoid assuming CNN texture bias or ViT shape bias in advance; confidence after shuffling alone does not establish sensible recognition.

## E. Reproducibility and submission

- [x] Save configurations, splits/IDs, metrics, figures and commands so each reported number is traceable.
- [x] Document setup, data preparation, training and analysis commands in repository README; attribute materially reused external code.
- [ ] Preserve code/environment versions; keep raw datasets and unnecessary large checkpoints out of Git.
- [ ] Include Task 1 evidence in the shared 8-page NeurIPS-style report and public GitHub submission; report wording/interpretation remain the student's own under the assignment policy.

## Working sequence

Full implementation contract: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). It includes required inference integration, sample identity rules, all metric/plot outputs and end-to-end verification. Review counts remain distinct from prediction coverage. Documentation is prepared; remaining code is not marked complete.

2026-09-18 selection update: user's retained set finalized locally: 200 unique conflicts, exactly 20 per direction across the final five pairs. Manifest and acceptance flags now reflect retained selection; PNG integrity and archive verified. Pending: reasons/status for 50 excluded candidates, remote Drive archive synchronization, conflict dataset loader/inference and analysis. Visual validity follows user selection rather than a new assistant audit.

Current review: only standalone metrics and paired comparison functions exist in analysis Python files. Cue-conflict counting, direction aggregation, cosine stability, joint projection, result loading/validation and report exports remain to implement. All-pairs exploration is running per user; final pairs/strengths and selected accepted set are not yet frozen. Develop analysis using toy data without claiming final results are available. Next manual coding exercise is calculate_cue_conflict_metrics with three aligned label arrays and an explicit class count, followed by checks for known counts and the all-other case.

Next: learn and implement B3 shape/texture/other counting, then shape bias and coverage using a toy example. Full cue-conflict generation is a separate dependency for final results. Before trusting B1/B2 in experiments, finish their small synthetic verification checks. Then proceed through B4, B5 and final reporting/integration. Update this checklist as each implementation and evidence item is verified.
