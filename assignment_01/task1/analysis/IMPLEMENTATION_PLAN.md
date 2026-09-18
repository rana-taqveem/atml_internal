# End-to-end Task 1 analysis implementation

## Progress: Step 1 baseline verification, 2026-09-18

Translation refactor review: schedule moved into data/translation_dataset.py; old translation_conditions.py no longer exists. get_translation_dataset currently loops through all conditions but returns outside the loop, exposing only up/32 and discarding earlier loaders. main.py still uses the old single-translation block. Recommended manual correction is a generator yielding each loader/metadata pair and a matching consumer loop, extracted into a translation inference helper to reduce start_inference clutter.

Translation schedule is saved under data/translation_conditions.py (use this import path). Its 12 unique conditions and four directions per displacement passed a basic check. Next manual change removes the old single (+8,0) dataset/loader and replaces only the translated-images inference block with the schedule loop, using unique condition IDs in result metadata/filenames. Translation inference and aggregation remain unexecuted.

Directional export wiring is present in saved report.py. Translation review: reflect padding and shifted-crop code are present; main.py still schedules only (+8,0). Next manual increment defines shared translation conditions: baseline for displacement zero, plus four cardinal directions at 8/16/32 pixels (12 new conditions), each with a distinct condition filename ID. Inference-loop replacement, direction averaging, zero-identity check and plotting remain pending.

2026-09-19 directional summary test PASSED: two synthetic directions produced correct separate counts/ratios, including None bias for all-other predictions; direction totals summed to the input count. Next manual wiring collects direction rows inside build_cue_conflict_report and writes cue_conflict_directions.csv after all model checks. Actual CSV export and real inference remain unverified.

2026-09-19 cue report synthetic verification PASSED: four model rows each with one shape/texture/other, bias 50%, coverage 66.6667%; reordered records accepted, changed content source IDs, duplicate conflict IDs and class order rejected. Loader and writer were replaced with in-memory fixtures, so this does not verify actual files or CSV output. Next manual increment adds directional breakdowns using masks and the same cue metric functions.

2026-09-19 saved review confirms conflict_models tuple ordering and the None guard are fixed. Next manual increment exports the four cue summaries, comparing conflict-ID-to-source/label mappings and class order across methods before saving. Real inference, grouped cue summaries, review-count audit and qualitative examples remain pending.

2026-09-19 follow-up: zero-shot conflict helper is present and checkpoint metadata now uses _head.pth. Two execution blockers remain in start_inference: conflict_models tuples are still reversed relative to loop unpacking; the guard raises when clip_conflict_result is not None instead of when it is None. Manual corrections supplied; conflict execution/report integration still pending.

2026-09-19 conflict wiring review: infer_cue_conflicts name fixed, but checkpoint suffix still _conflict_inference.pth. conflict_models contains (model, name) while loop unpacks (name, model); fix tuples before execution. Next manual snippet adds a separate CLIP conflict zero-shot adapter preserving conflict identifiers/labels and replacing classifier scores with scaled image/text similarities, plus save wiring. No inference executed.

Conflict inference saved-code review: batch collection and output fields match the proposed structure. Function is named infer_cur_conflicts (rename to infer_cue_conflicts for forthcoming calls); checkpoint metadata incorrectly uses _conflict_inference.pth instead of the actual loaded _head.pth. Manual corrections and trained-head loader/save wiring supplied. No real conflict inference executed; zero-shot adapter remains pending.

User-run dataset smoke check passed: 200 records, first batch [4,3,224,224], aligned content/style labels and unique conflict strings shown. This verifies the first batch, not all image pixels. Next manual increment is inference on dictionary batches, preserving conflict_ids separately from repeated numeric content_ids; save logits/features and both labels for trained heads. Existing CLIP zero-shot helper assumes image_ids.clone() and y_true, so conflict zero-shot integration still needs an explicit adapter.

Saved ConflictDataset wrapper is present. The local directory is now data/conflict_dataset (the older misspelled path no longer exists). The metadata loader checks membership in entry['content_ids'], but manifest entries contain conflict_id and metadata_path. Replace this with record conflict_id != entry['conflict_id']. Image/batch runtime verification and inference integration remain pending.

Conflict loader verification blocked by unsaved/empty source: conflict_dataset.py is empty on disk at the latest read. No loader runtime verification claimed. Next supplied increment is a Dataset wrapper around the proposed load_conflict_records function; both must be saved before testing against the finalized selection.

Cue-summary adapter reviewed in saved report.py: import and score/label checks are present. It currently appears after the __main__ block; move that entry point below all definitions before calling cue analysis there. Next manual increment loads manifest references and accepted/selected per-image metadata from the finalized dataset; actual local root is data/conflixt_dataset. Dataset image loading and inference integration are still pending.

Cue metric formulas tested successfully on mixed, all-other, all-shape and all-texture cases; count conservation and fractional-label rejection passed. Checklist marks the metric calculations implemented, not conflict inference/evaluation complete. Next proposed adapter reads logits plus content_labels/style_labels from one conflict result and calculates a summary. Current main.py has no conflict inference records carrying these fields; dataset/inference integration remains pending.

Cue-count correction is present in saved code: all three inputs preserve dtype until validation. Next manual increment: calculate_cue_metrics wrapper returning counts, shape_bias_pct and coverage_pct, using None for undefined shape bias when every prediction is other. Formula implementation and tests remain pending.

Cue-count increment verified: actual count_cue_decisions passed one-of-each and all-other examples, and rejected empty, mismatched-length, identical-content/style and negative-label inputs. Validation regression: all three arrays are cast to torch.long before validation, silently converting prediction 0.9 into class 0. Remove dtype=torch.long in those three conversions; manual fix pending. Bias and coverage formulas remain pending.

Color synthetic check PASSED using actual analysis function definitions with an in-memory fixture loader: eight rows, reordered IDs, baseline/transformed accuracy both 50%, zero accuracy change and consistency 50% for grayscale / 0% for hue. Different model/checkpoint identifiers were rejected. This test bypassed file loading and did not exercise the latest CSV writer. Saved code includes duplicate save_classification_table definitions; remove the earlier one and retain the later shared-writer version. Color file export and real evidence remain pending. Next teaching step is cue-conflict shape/texture/other counts and their denominators.

Color-row follow-up: import syntax and unreachable return are fixed. Current loop uses `greyscale` instead of the inference condition `grey_scale`, and appends its summary outside the inner loop, retaining only hue (four rather than eight rows). Manual loop correction supplied before CSV integration. Next refactor extracts a shared `save_rows_csv` writer; color export remains unverified.

Color helper review: user consistently renamed the clean condition to `baseline` in main.py, make_subset.py and report.py, and comparator to `compare_with_baseline`. Preserve these names. Current report.py has an unclosed parenthesized import and an unreachable second return in summarize_intervention; manual corrections supplied. Next: collect eight intervention rows (four methods times grayscale/hue), validating a shared labeled test set across all loaded records. Earlier synthetic pass predates these latest edits; current file cannot run until import is closed.

Synthetic reporting integration PASSED: actual function definitions loaded without project configuration side effects; four invented .pt records loaded and exported as four CSV rows with expected accuracy, including reordered IDs. Duplicate IDs, different ID sets, mismatched per-ID labels and reversed class order were rejected. Windows sandbox denied the initial temporary-folder access; the approved rerun passed. This is synthetic verification, not real inference evidence. Next lesson: a shared clean-versus-intervention summary using compare_with_clean for grayscale/hue.

Output-path fallback is now present and correct in the saved coordinator. Next manual increment: validate shared class order, unique image IDs and per-ID ground-truth labels across the four baseline records, allowing different row orders. Runtime integration and real baseline evidence remain pending.

Follow-up: `task_config as TaskConfig` now imports the correct instance, so the configuration access issue is fixed despite the confusing alias. The saved function still requires output_path while its entry point omits it. Minimal remaining fix: give output_path a None default and derive its base directory from results_dir when omitted. Cross-run checks and execution remain pending.

Baseline coordinator review: user added `build_baseline_report`, but it references instance attributes through the `TaskConfig` class and requires an `output_path` argument absent from the entry-point call. Provide manual correction to use existing `task_config` instance and a single `results_dir` argument. Resolve these execution blockers before cross-run validation. No real report execution verified.

Latest increment: saved `load_inference_result` matches the proposed explicit-path, CPU-loading and identity-check implementation; `top_1_accuracy_pct` is now correctly named. Next manual implementation: `build_baseline_report` to load the four original-condition results and export `clean_baseline.csv`. Static review only; real-file loading, cross-run image/class comparability and CSV execution are not yet verified.

CSV increment reviewed: `report.py` now contains `save_classification_table`, matching the proposed implementation. Static review only; end-to-end export is not yet verified. Next: a result loader with explicit model/condition filenames, CPU loading and metadata checks. The summary still calls its percentage-valued accuracy column `top_1_accuracy`; renaming to `top_1_accuracy_pct` is pending. Real baseline files and run comparability remain to be verified.

Latest review: finite-logit validation is restored. The student named the summary helper `summarize_metrics`; subsequent callers should use that name. Next increment is collecting summary rows from result dictionaries before adding file loading and CSV export. Baseline execution/export remains pending.

Follow-up saved-code review: label validation now precedes integer conversion and top-k has separate type/range checks. The previous finite-logit check was removed during editing and must be restored. Next teaching increment: a small `summarize_result` adapter that converts one inference record into one named table row; CSV export follows afterward. No Python edits made by the assistant.

The existing `calcultate_metrics` passed an isolated synthetic check: top-1 70%, macro-F1 71.1111%, mean maximum confidence 0.8482255 and top-2 100%. The actual function was loaded without project configuration imports. No model inference or Python source edits were performed. This checks a known valid-input example, not all edge cases or real model performance.

Next: validate labels before integer conversion and consolidate top-k validation, test invalid inputs, then connect metrics to saved results. Step 1 remains in progress; exports and real baseline evidence are pending.

## Assignment-step audit, 2026-09-18

Use the six assignment steps as the teaching and implementation sequence, with shared helpers underneath them. This supersedes the earlier component-first teaching order.

| PA1 step | Observed implementation | Remaining |
| --- | --- | --- |
| 1 Clean | calcultate_metrics; CLIP normalized image/text similarities times exp(logit_scale), then softmax | Verify current metrics, wire into exported summaries, run all four methods |
| 2 Color | Three-channel grayscale; fixed hue rotation; compare_with_clean | Wire paired comparisons, record hypotheses/changes preserved, export absolute and delta metrics |
| 3 Cue conflict | User-selected 200 files finalized with balanced IDs | Dataset loader/inference carrying conflict and content/style IDs; counts, bias, coverage, group summaries and examples; omitted-candidate rejection audit |
| 4 Translation | Reflection pad and shifted crop implemented | main.py schedules only (8,0); add 0/8/16/32 and cardinal directions, safe condition filenames, averaging and plots |
| 5 Patch | 4x4 enforcement, seed, nonidentity rejection in PatchShuffler.__call__ | make_subset uses patch_size=566 and nonexistent shuffle_patches method; use 56 and call instance; verify pixel preservation/reuse and export comparisons |
| 6 Representations | Inference saves features; simple scatter helper exists | Pair IDs/content references, cosine scores, joint projection, fixed samples/settings and prediction-feature agreement analysis |

Shared artifact design: one raw result per method/condition, containing all sample IDs, labels, logits, features and metadata; classification table, paired sample table, cue decisions table, translation direction/aggregate tables, representation similarity/projection data and run completeness report are derived without rerunning inference. CLIP head and zero-shot share a backbone representation, so three backbone projections suffice while four decision methods are compared.

Concepts to teach: scaled CLIP similarity changes softmax concentration, not ranking; absolute accuracy and own-baseline percentage-point change answer distinct questions; chromatic information means color cues, not object geometry; grayscale changes color but hue rotation changes HSV hue while preserving spatial layout and HSV S/V (not guaranteed perceived luminance). Pooled cue bias uses pooled counts, not average group ratios. Pair conflicts with clean content features, allowing repeated content IDs.

Research-question evidence mapping: RQ1 uses color table plus cue counts/bias/coverage; RQ2 uses translation curves and patch comparisons; RQ3 joins per-sample prediction flips with cosine changes and compares CLIP decisions on shared features; RQ4 uses pretraining/model metadata and failure examples to discuss confounds without claiming architecture-only causality. No empirical answers exist until execution.

Status: implementation contract, 2026-09-18. Existing Python includes standalone classification metrics and paired clean comparisons only; this document does not claim the remaining components are implemented. User prefers learning through manual implementation unless direct edits are explicitly requested for the current work.

## Processing boundary

Generate fixed images -> validate selected datasets -> infer frozen models -> save raw outputs -> validate and align results -> compute metrics -> export tables/figures -> student interpretation.

Analysis runs from saved outputs without training or model downloads. It must fail clearly for incompatible records and report missing experiments rather than silently producing a complete-looking report.

## Result contract

All results: model_name, backbone_name/pretraining, head checkpoint identity when relevant, ordered classes, unique condition ID and settings, seed, logits [N,C], features [N,D], labels [N], sample identifiers. Preserve source/checkpoint/config fingerprints where available. Explicit units: accuracy/consistency/coverage/bias in percent; accuracy differences in percentage points; confidence and cosine similarity as fractions/raw similarity.

Ordinary conditions use original numeric image_ids. Cue conflicts additionally use unique string conflict_ids and aligned content_ids, style_ids, content_labels, style_labels, alpha, selected/accepted state and source manifest identity. Repeated content_ids are valid: map each conflict to its corresponding clean content row. Do not require conflict IDs to match ordinary image IDs. Do not use generic accuracy as the sole cue-conflict metric.

## Ordered implementation milestones

1. Validate label/logit/feature arrays and metadata, preserve compatibility with existing calcultate_metrics and compare_with_clean. Test ties, missing classes, malformed inputs and shuffled IDs. Remove configuration side effects from pure analysis modules.
2. Cue-conflict metrics: shape/texture/other counts, shape bias, coverage, per-example decisions, per-direction and per-pair aggregates. Return null for undefined shape bias. Compute pooled bias from summed counts, not an unweighted mean of group biases. Keep dataset review counts separate from model counts.
3. Selected conflict loader: read manifest references and authoritative metadata; require accepted and selected flags; verify paths, labels and IDs; load saved RGB PNGs without re-stylization. Return both intended labels and conflict/content IDs. Support explicit dataset root on Windows and Colab; no hardcoded extracted ZIP folder.
4. Inference integration: save features/logits for all four methods and all conditions; reuse CLIP image embeddings for zero-shot classification; preserve scaled similarities for confidence. Add every translation direction at 8,16,32 plus zero baseline. Correct patch_size=566 to 56. Include dx/dy in condition IDs and output filenames to avoid overwrite. Validate common inputs across models.
5. Translation metrics: require the complete expected direction grid, pair by IDs, report direction metrics and their equal-weight means; plot accuracy and consistency against displacement. Zero baseline has clean accuracy and 100% consistency.
6. Feature similarity: paired cosine per image and mean for grayscale, conflict, translation and patch shuffle. Require matching backbone/version and feature dimension; reject nonfinite or zero-norm vectors. Clean content matching for conflict permits repeated reference rows. Retain sample-level similarity for examining prediction/representation disagreements.
7. Representation visualization: one t-SNE fit per backbone on a fixed combined clean/transformed feature selection, with recorded sample IDs, seed, preprocessing and parameters. Color by ground-truth/content class, markers by condition. Reuse fitted coordinates for views; do not compare separately fitted coordinates across backbones. Avoid silently overweighting duplicated clean examples.
8. Reporting runner: load explicit run outputs, validate coverage of required methods/conditions, write classification/comparison, cue counts/bias/coverage, review counts, translation and cosine tables; serialize undefined values as JSON null. Save plot data/config along with figures. Record source files and hashes.
9. Qualitative review: export deterministic candidate galleries with content/style/conflict and all model predictions, including agreements, disagreements and other decisions. Never use these predictions to revise image acceptance. Student supplies interpretation.
10. End-to-end synthetic smoke test: fabricated complete run with known counts and reordered IDs, not real model conclusions. Then run actual inference/analysis and verify final checklist against evidence.

## Required artifacts

- Clean baseline table for three trained heads and zero-shot CLIP: top-1, macro-F1, mean maximum confidence.
- Compact clean/grayscale/hue/patch comparison table including own-baseline changes and consistency.
- Cue-conflict counts, shape bias and coverage overall and per pair/direction; separate generated/accepted/rejected/unreviewed/selected audit.
- Translation direction table and averaged accuracy/consistency curves for 0,8,16,32.
- Backbone cosine-stability tables and combined representation projections for required interventions.
- Qualitative conflict examples with predictions and sample-level prediction/feature-change data.
- Configurations, IDs, checkpoint identities, data paths/hashes and completeness report supporting reproducibility.

## Current blockers to final evidence

Selected 200 conflicts are finalized locally but need matching remote files and inference integration. Fifty excluded candidates lack documented review reasons/status. Current inference uses only translation (8,0); patch subset has a 566-pixel typo; translation filenames omit displacement. Final results cannot be declared complete until these are resolved and required runs exist.
