# Task 1 learning notes and question log

### Translation experiment schedule

Compare displacements 0,8,16,32 pixels, averaging four cardinal directions at each nonzero displacement. For the current crop implementation positive x moves content right and positive y down. Zero displacement is identical in all directions; reuse the clean baseline after verifying the transformation identity. Twelve nonzero conditions need unique IDs such as translation_right_8 so files do not overwrite. A shared schedule module should feed inference and analysis. Different directional sensitivities can cancel in the mean, so retain direction-level results too. Reflection padding and cropping preserve output shape but can crop edge content and introduce reflected boundary evidence.

### Collecting directional rows with extend

summarize_cue_directions returns a list of dictionaries. Use direction_rows.extend(...) inside the model loop to add each dictionary as an individual CSV row. append(...) would instead add a nested list, which DictWriter cannot use as a row. Save the overall and directional tables after all model consistency checks. With five pairs, two observed directions per pair, and four models, expect 40 directional rows; inspect n_total to verify the intended 20 conflicts per direction. The two-direction synthetic function test passed, including an all-other group with undefined bias.

### Directional cue summaries

The overall four-model cue report passed synthetic checks, including reordered conflicts and rejection of source-ID/duplicate-ID/class-order mismatches. Next, separate content=A/style=B from content=B/style=A. A Boolean mask selects the corresponding rows of logits and both label arrays together; reuse summarize_cue_conflicts on that subset. Directions may have different counts and behaviour despite a balanced overall dataset. Overall bias must be calculated from pooled counts, not the unweighted average of direction percentages. File I/O was bypassed in this synthetic report test.

### Conflict inference: batch dictionaries and row alignment

The user verified 200 selected conflicts and a first batch of shape [4,3,224,224]. A conflict inference loop reads batch['image'], obtains features from model[0] and scores from model[1], and collects both source labels and identifiers in the same loop. Tensor chunks concatenate along dim=0 (images), while string conflict IDs extend a list. Save content_ids separately: repeated source images are allowed, whereas each conflict_id must be unique. eval() controls evaluation behaviour and no_grad() disables gradient recording. The existing ordinary zero-shot helper expects image_ids and y_true, so do not pass the new conflict record directly until adapting that path.

### From a conflict manifest to aligned inference records

The finalized manifest is a list of conflict_id/metadata_path references. Each referenced JSON supplies image_path, content_label, style_label, content_id, style_id, accepted and selected_for_evaluation. Load only the approved selection in that manifest; do not glob all PNGs because the folders also hold originals and previews. First load/validate metadata, then implement a Dataset that loads one conflict image and returns its identifiers and both labels together. The local directory is spelled conflixt_dataset; the archive's internal root is conflict_dataset. Pass the root explicitly. Put the script's __main__ block below all function definitions so new report functions are defined when invoked.

### Cue metric verification and result adapter

Mixed (one each), all-other, all-shape and all-texture formula tests passed, as did fractional-label rejection. Next convert logits [N,C] into predicted class IDs [N] using argmax(dim=1), then pass those IDs and aligned content/style label arrays [N] to calculate_cue_metrics. Softmax is unnecessary for selecting the largest score. Both intended labels must lie within the C class columns. This adapter does not create conflict inference results; the loader/inference layer must preserve unique conflict IDs and both source labels for later use.

### Shape bias and coverage: different denominators

After validating aligned predictions/content/style labels, reuse count_cue_decisions. Shape bias is 100*n_shape/(n_shape+n_texture); coverage is 100*(n_shape+n_texture)/n_total. Empty input is already rejected by the counting function. When all predictions are other, coverage is zero and shape bias is undefined: return None, not zero (zero would imply texture-only choices among intended labels). None becomes null in JSON and a blank field with the current CSV writer; explain that blank as undefined. Keep counts alongside percentages. The dtype-preservation correction is now present in saved code; the percentage wrapper has only been proposed, not verified.

### Cue-count review: validate before casting

The three Boolean masks correctly count shape, texture and other decisions in tested valid inputs. However, dtype=torch.long in the initial conversions removes fractional information before the integer-valued check: prediction 0.9 becomes class 0 and was incorrectly counted as shape. Preserve the input dtype during validation. Equality comparisons and Boolean sums do not require integer casting afterward. Tests passed for one of each decision, all-other outcomes, and rejection of empty arrays, unequal lengths, identical content/style labels and negative labels. Fractional-input correction remains pending.

### Tested example: equal accuracy does not imply stable predictions

True labels [0,0,1,1], baseline predictions [0,1,1,0], grayscale predictions [1,0,1,0] and hue predictions [1,0,0,1] all give 50% accuracy. Grayscale preserves two of four baseline predictions (50% consistency); hue preserves none (0%). The current color-row builder passed these checks for all four decision methods, including reordered image IDs. File loading was replaced with in-memory fixtures for this test. Model/checkpoint mismatch checks passed. Duplicate function definitions do not combine: the later definition replaces the earlier name binding, so remove the older save_classification_table copy.

### Baseline reporting test and the next color-analysis lesson

The synthetic integration check passed: four saved result files produced four CSV rows; reordered images were allowed, while duplicate IDs, different image sets, mismatched labels and class order were rejected. No real model scores were measured. Next, compare each model's grayscale/hue result against its own baseline result. Accuracy compares predictions to true labels; consistency compares transformed predictions to baseline predictions. Equal accuracies do not imply identical predictions because different images can become correct or incorrect. Signed accuracy change is transformed minus baseline, in percentage points. Keep the same model and checkpoint when comparing interventions; different models belong in the baseline comparison instead.

### Why validate baseline records across models?

Equal sample counts do not imply equal test images. Compare unique ID sets, ordered class names and the true label attached to each ID. Sort each record's IDs and use its own sorting indices on its labels before comparison; different original row orders are allowed. Do not require predictions to agree across models, because those differences are what the experiment measures. This check verifies identity/label comparability, not pixel equality or checkpoint provenance. The output-path fallback in report.py has now been reviewed as present.

### Configuration class versus instance

`TaskConfig` is the class definition. Attributes assigned through `self` in its constructor, such as RESNET50 and TASK_RESULTS_DIR, belong to an instance. The project exposes the initialized `task_config` instance; report code should import and use it instead of reading those attributes on `TaskConfig`. Also ensure required function arguments match the call: the initial baseline coordinator required output_path but the entry point supplied only results_dir. Deriving the report path from results_dir removes this mismatch.

### Connecting the baseline report

The loader and percentage column name were reviewed in the saved files. The next orchestration function loops over the three trained-head model names and CLIP zero-shot, loads each original-condition result, then passes the four dictionaries to the shared CSV writer. It contains no new metric formula and performs no training or inference. Keep results_dir an argument so the same function works on local and Colab paths. A four-row CSV is not by itself proof of a fair comparison: matching sample IDs, labels and class order still need validation before interpreting real results.

### Loading saved inference results for analysis

`save_classification_table` is now present (static review). The next layer reads explicit model/condition result paths rather than every .pt file in a directory. Each result file stores predictions/features/metadata, whereas a head checkpoint stores learned parameters; loading a result does not rerun the model. Load tensors on CPU for analysis and check internal model/condition fields against the requested file. Missing files should raise a clear error rather than silently produce a partial baseline table. A successful file load alone does not verify that all four runs used the same images, class order or intended checkpoints; those comparisons remain part of baseline integration.

## Step 1 lesson: verify metrics before evaluating models (2026-09-18)

Follow-up: the student's saved label/top-k changes are present. Keep a finite-value check for logits as well as labels: NaN/infinite scores can produce NaN softmax probabilities. The next layer takes a raw inference record (IDs, labels, logits, features and metadata) and produces one summary row (model, condition, sample count and metrics). This reuses the same calculations across baseline and transformed conditions and does not require another model forward pass. Summary rows complement rather than replace raw records needed for paired/representation analysis.

A known-answer synthetic check separates calculation mistakes from model behaviour. We tested the current function with true labels [0,0,0,0,1,1,1,2,2,2] and predictions [0,0,0,1,0,1,1,0,2,2]. Seven of ten predictions are correct: top-1 accuracy is 70%. Per-class F1 scores are 2/3, 2/3 and 0.8, so macro-F1 is 71.1111%. Logits start at zero, assign score 2 to each true class, then score 3 to each predicted class. Softmax yields mean maximum confidence 0.8482255 and top-2 accuracy 100%. All four checks passed. These are invented test results, not experimental findings.

The input scores have shape [10,3]: ten image rows and three class columns. Softmax(dim=1) works across classes separately for each image. max(dim=1) produces ten winning probabilities and ten class IDs; mean() averages the winning probabilities. Confidence measures score concentration and need not equal accuracy.

Next validation lesson: casting 1.9 to an integer silently produces 1, so check labels are integer-valued before converting to torch.long. Top-k must also be an integer between 1 and the number of classes. Python implementation changes remain for the student to type; only these learning/progress documents were edited.

2026-09-18: Finalized the user's retained local selection in data/conflixt_dataset (actual folder spelling). Verified 200 unique conflicts, 20 in each of ten directions, readable RGB 224x224 source/style/conflict PNGs and all preview paths. Rebuilt stale 250-entry manifest to contain only 200 retained records; preserved manifest_before_finalization.json. Marked retained metadata accepted/selected with review_source=user_retained_selection. Saved selection_summary.json and image_sha256.json, and created verified data/conflict_dataset_final_200.zip with internal root conflict_dataset/. Fifty removed candidates have no documented rejection reasons, so exclusion is not automatically counted as visual rejection. This was an integrity/balance audit of the user's selection, not a new visual review of all 200 images. Remote Colab/Drive files were not inspected or changed; upload/extract the finalized archive there before inference.

Latest generation change: configured pairs are airplane/bird, bird/car, monkey/truck, car/deer and bird/cat. Production now defaults to 25 candidates per direction (50 per unordered pair, 250 total) at alpha=0.7. Outputs use flat_pairs_alpha_0_7_VERSION with one flat folder per pair; filter preview__*.png to review and identify directions from filenames. No candidate/direction subfolders in production. Select 20 valid examples per direction (40 per pair, 200 total), not twenty per pair. Existing runs are preserved; pilot layout remains separate. Mocked production verification checked 250 unique records, fifty previews per pair, no nested folders and valid relative metadata paths.

2026-09-17 strategy update: user has finalized airplane/bird as a pair and wants an exploratory grid across all ten STL-10 classes: each content class with the other nine style classes, ten candidates per direction (900 outputs; 45 unordered pairs). This exploration is separate from the final >=200 accepted conflicts. Use one explicitly recorded trial alpha and no classifier predictions; compare matched content exemplars across style classes to reduce source-selection differences. Keep source IDs, direction folders, previews and acceptance metadata. Ten candidates per direction are for pair screening; final five-pair selection still targets twenty accepted images per direction. Final pair quality must be assessed in both directions; no automated alpha or pair selection is implied.

2026-09-17: Generator now defaults to an alpha pilot. Run `python -m assignment_01.task1.scripts.conflict_image_generator --mode pilot --run-version v1`. Five fixed combinations per direction are compared at 0.5, 0.7, 0.85 and 1.0, yielding 50 comparison sheets and 200 variants for ten directions. Pair/direction/candidate folders hold content/style images, labeled comparison.png and per-alpha images/metadata. Record visual choices in direction_choices.json; it is a review record, not yet consumed by production. `--mode production` retains the previous fixed-alpha generator. Existing folders cause an error; use a new version rather than overwrite. Mock-stylizer verification checked both directions, all four strengths and saved paths; real Colab generation remains to be run.

See [cue-conflict review protocol](CUE_CONFLICT_REVIEW.md) for the proposed 400-candidate pool, acceptance/rejection rules and balanced selection of 200 valid images. Selection follows fixed candidate order among accepted images, rather than an undefined visual ranking of the "best" outputs. Extra accepted but unused examples are not rejections.

Track implementation and experiment completion in [Task 1 checklist](TASK1_CHECKLIST.md). A checked implementation item does not establish that final experimental evidence has been generated.

These are revision notes from our implementation discussions, not experimental results or report prose. Keep adding questions, explanations, examples and implementation decisions as we progress. User preference: provide implementation code in chat for manual typing; edit code only when explicitly requested. Documentation updates are requested.

## 1. How should I approach ML analysis?

Start with a question or hypothesis, identify the controlled change, choose a metric that measures the effect, preserve comparable inputs, then examine aggregates and individual failures. Keep image generation, inference, metric calculation and presentation separate. Save enough metadata to reproduce each result. Do not select examples or settings based on desired test outcomes.

The frozen backbone converts images to feature vectors. A trained linear head converts features to class scores. At inference, new images still need feature extraction. A head checkpoint contains learned weights/bias, not the training images' feature vectors. An nn.Linear layer has no additional hidden layer.

## 2. Questions about preprocessing and interventions

- **Does ToTensor scale to [0,1]?** For ordinary uint8 RGB PIL images, yes: values 0..255 become floating-point values 0..1 and layout becomes [C,H,W]. This is not a universal guarantee for every input dtype or image mode.
- **Is this model normalization?** No. Model normalization usually subtracts channel means and divides by channel standard deviations. Apply it exactly once, after interventions; account for wrappers that already normalize inside forward.
- **What order do we use?** RGB conversion -> common 224x224 size -> tensor -> one intervention -> model-specific normalization -> backbone -> decision method. Compare each condition with baseline inputs rather than chaining unrelated interventions.
- **How many color interventions?** Task 1 requires grayscale plus one additional intervention; fixed hue rotation is an allowed choice. Extra channel-removal experiments are optional.
- **Can affine implement translation?** It can translate, but the assignment specifically requires reflection padding followed by a shifted crop. Generic affine fill behavior alone does not establish compliance.
- **What does a 4x4 patch grid mean?** Four cells per image side, 16 patches total. For 224x224 images each cell is 56x56 pixels, not 4x4 pixels. All pixels within a patch move together.
- **Why seed 6304?** The assignment specifies it. A seed initializes a random-number sequence for reproducibility. A train/validation random_state only controls that split, not every other random operation. Seed relevant generators before randomized operations.
- **Why reject identity permutations?** Identity leaves all patches in place. Non-identity requires at least some movement, not that every patch must move.
- **randperm versus permute?** randperm supplies a randomized list of patch IDs. Tensor permute rearranges axes so patches can be assembled correctly; it does not itself randomly shuffle patches.
- **Why generate interventions once?** All models must receive exactly the same transformed images. Reusing a seed alone is insufficient if call order or random-state consumption differs.
- **What do style-transfer encoder/decoder do?** The encoder maps pixels into spatial feature maps; AdaIN matches content features' channel-wise means and standard deviations to style features; the decoder maps features back into pixels. A compatible pretrained encoder/decoder can be reused without training a new style-transfer model. Mean describes average activation and variance describes spread, not a complete texture description. Visual validation is necessary.
- **How does zero-shot CLIP classify?** Encode fixed class prompts, normalize image and text embeddings, compare similarities, apply the learned logit scale for confidence softmax, and select the highest score. No trained STL-10 head is used. Reuse class text features across image conditions.

## 3. Recording results and running experiments

Save model/decision-method identity, backbone settings, head checkpoint when relevant, class order, transformation settings, seed, image IDs, true labels, logits and features. Image IDs associate each row with its source image; storing aligned arrays is sufficient, without a tuple per feature.

Run from the repository root using `python -m assignment_01.task1.scripts.main` and supported CLI arguments. Module names use dots and omit `.py`. Use `pathlib.Path` for filesystem paths. Colab environment indicators are provided by its runtime; Drive mounting belongs in Colab-only setup, not metric calculations.

Dataset reuse depends on the expected directory layout and integrity, not just folder existence. A completed download progress bar does not prove a valid archive; checksum validation detects partial/corrupted files. Download staging and verification before copying to Drive help avoid retaining invalid artifacts.

Changing CLIP's activation implementation changes the feature extractor: regenerate its cached features and retrain its head for a consistent experiment. An unrelated backbone does not need retraining solely because CLIP changed. A one-epoch smoke run tests plumbing; it does not replace full training/model selection.

## 4. Accuracy, precision, recall, F1 and confidence

For one class, treat it as positive and all remaining classes as negative:

- TP: predicted this class and truly this class.
- FP: predicted this class but truly another class.
- FN: truly this class but predicted another class.
- Precision = TP/(TP+FP): how trustworthy are predictions of this class?
- Recall = TP/(TP+FN): how many actual examples of this class were found?
- F1 = 2TP/(2TP+FP+FN): balances precision and recall.
- Support: number of true examples of the class, TP+FN.
- Accuracy = correct predictions / number of images.

Macro metrics average each class's metric equally. Macro-F1 is the average of class F1 scores, not F1 computed from macro precision and macro recall. Weighted averages weight classes by support; micro metrics pool counts. In ordinary single-label classification over all classes, micro precision/recall/F1 equal accuracy.

`average=None` retains per-class arrays. Explicitly listing all classes includes absent classes. `zero_division=0` sets undefined scores to zero; this policy affects macro averages on subsets with absent classes.

Top-k accuracy counts an image correct if its true class occurs among the k highest-scoring classes. Top-1 is ordinary accuracy; top-C is trivially 100% for valid labels. Top-k needs ranked scores, not only saved winning labels. Top-5 is supplementary; the baseline requirement is top-1, macro-F1 and mean maximum confidence.

Softmax converts each image's scores to probabilities. Mean maximum confidence averages the winning probabilities across correct and incorrect predictions; confidence is not correctness or proof of calibration. Zero-shot CLIP confidence uses scaled similarities.

Current code convention: accuracy, top-k and macro metrics use percentages; per-class metrics and mean confidence use fractions. Keep units explicit. Known cleanup items: duplicate k range check, missing integer-type check for k, unused imports, and conversion of labels to integer before validation.

## 5. Axis and dimension questions

**What is an axis?** An index position in an array/tensor. In this context dimension can mean axis. The number of dimensions is the number of index positions needed to select a scalar; the size of a dimension is the number of choices at that position.

**Are columns the number of axes?** No. Shape [2,3] has two axes, with sizes 2 and 3. Shape [2,1000] also has two axes. Count entries in the shape to count axes.

**Is axis 0 always rows?** No. Axis 0 is the first index position. In scores[image,class] it selects images; in image[channel,row,column] it selects channels. Meanings come from the data layout.

**How does this relate to conventional arrays?** scores[image,class] corresponds to indexing a nested array as scores[image][class]. A batch images[image,channel,row,column] has four axes; no geometric four-dimensional visualization is needed.

| Tensor | Shape | Meaning |
| --- | --- | --- |
| logits/probabilities | [N,C] | One class score/probability for each image |
| labels/predictions/confidence | [N] | One true label/predicted label/winning probability per image |
| ranked_classes | [N,C] | Class ID at each rank, not the score at each class column |
| top_classes | [N,k] | First k ranked class IDs for each image |
| labels[:,None] | [N,1] | One true label per image with an added axis |
| matches | [N,k] | Whether each candidate matches the true label |
| matches.any(dim=1) | [N] | Whether any candidate is correct for each image |
| backbone features | [N,D] | D feature coordinates per image, not D class scores |
| image batch | [N,3,H,W] | Image, color channel, pixel row, pixel column |

For scores = [[8,2,1],[1,7,3]]:

- scores[0] selects the first row: [8,2,1].
- scores.max(dim=1).values compares classes within each image: [8,7].
- scores.max(dim=0).values compares images within each class: [8,7,3].

For reductions, vary the selected index while holding the others fixed. By default max removes that axis; keepdim=True retains an axis of size 1. Not every operation removes an axis: softmax and argsort preserve shape.

`labels[:,None]` inserts an axis; broadcasting compares one label with every candidate in its corresponding row. Equality gives Booleans; float converts True/False to 1/0; mean gives a success fraction; item converts a scalar tensor to a Python number.

## 6. Next lesson: paired baseline/intervention comparisons

Source: ATML-PA1.pdf, Task 1 steps 1, 2, 4 and 5. Start with baseline versus grayscale, then reuse for hue. Compare the same model and decision method, checkpoint, class ordering and image IDs.

Accuracy change (percentage points) = transformed accuracy (%) - baseline accuracy (%).
Accuracy drop (percentage points) = baseline accuracy (%) - transformed accuracy (%).
Consistency (%) = 100 * count(baseline prediction == transformed prediction) / N.

Example: 90% baseline and 80% transformed means a 10-percentage-point drop, not a 10% relative decrease. Relative decrease would be 100*(90-80)/90 = 11.11%.

Consistency compares predictions to predictions; accuracy compares predictions to ground truth. A consistently wrong model can have 100% consistency. Equal accuracies can hide changed predictions because gains and losses can cancel.

Pair by image ID. Equal array lengths do not prove matching images. For the first implementation, require identical IDs in identical order and matching labels. If ordering differs, explicitly align by ID in a later helper rather than comparing mismatched rows. Reject duplicates and mismatched sets; do not silently take an intersection.

Manual implementation exercise: add a separate compare_with_baseline(baseline_result, transformed_result) function; retain calcultate_metrics for standalone metrics. Read labels/logits/IDs, validate pairing, derive predictions, compute both accuracies and consistency, and return clearly named fields. No paired-comparison code has been added automatically.

## 7. Remaining Task 1 analysis roadmap

1. Baseline: all three heads and zero-shot CLIP; top-1, macro-F1, confidence.
2. Color: grayscale and chosen additional color condition; absolute performance, accuracy change, consistency.
3. Cue conflicts: at least 200 valid images, at least five unordered pairs, both directions when feasible; visual rejection rule set before evaluation, accepted/rejected counts; shape/texture/other counts, shape bias and coverage. Shape bias = shape/(shape+texture); coverage = (shape+texture)/total. If no shape/texture decisions exist, report shape bias as undefined, not a substantive zero preference.
4. Translation: 0/8/16/32 pixels in four cardinal directions, reflection padding and shifted crop; average across directions and plot accuracy and consistency against displacement.
5. Patch shuffle: 4x4 grid, non-identity permutation per image, seed 6304, identical shuffled images across models; accuracy drop and consistency.
6. Representation analysis: paired cosine stability for grayscale, cue conflicts, translation and patch shuffle; joint baseline/transformed t-SNE or UMAP per backbone with fixed IDs and seed, class colors and condition markers. Do not compare coordinates of separately fitted projections.
7. Inspect informative cue-conflict agreements/disagreements/failures; distinguish prediction stability from feature stability and architecture effects from pretraining/data/capacity confounds. Interpret actual results yourself after running experiments.

## Revision questions

### Cross-review of the user's four shortlisted pairs and a fifth option

Reviewed 54 previews from the downloaded all_pairs_alpha_0_7_v1 pool: first three candidates in both directions of airplane/bird, bird/car, monkey/truck, deer/car, cat/ship and dog/bird; additionally indices 6 and 7 in both directions of monkey/truck, deer/car and bird/cat; and first three in both directions of bird/cat. This is a sample, not a full acceptance audit. Several manufactured-style outputs show color/block artifacts rather than unambiguous class-specific texture. Monkey-content/truck-style is particularly distorted in sampled examples; the reverse includes a promising truck with furry surface (c9_s7_006_1240_2756). Deer/car retains silhouettes but often produces color changes and background transfer. Bird/cat is proposed as a fifth pilot pair with preserved recognizable shapes in some examples, subject to the same visual validity review. Do not equate an attractive or strongly stylized image with an accepted conflict. No metadata decisions or config changes were made by this review.

### Is there published guidance on selecting the best conflict pairs?

A targeted literature search did not establish a validated STL-10/AdaIN ranking of class pairs. Geirhos et al. used curated object and texture images, including surface patches and repeated objects, and iterative Gatys style transfer for cue-conflict evaluation. AdaIN with paintings was used for their separate Stylized-ImageNet training dataset. Their model-based source filtering must not be copied here because the assignment prohibits prediction-based selection. Source: https://arxiv.org/html/1811.12231v3 (Methods and Appendix A.6).

The 2026 preprint On the Reliability of Cue Conflict and Beyond discusses ambiguity from unequal cue informativeness and imperfect cue separation: https://arxiv.org/abs/2603.10834. It supports careful validation, not a proven list of our best class pairs. Color similarity alone is not texture similarity: compare local patterns, fur/feather structure and surface variation. Brown fur shared by deer/dog may be ambiguous even with different shapes; contrasting colors alone may create color bias rather than a strong texture conflict. Visually pilot distinctive source exemplars and alternatives, document choices, retain required metrics and avoid selecting by classifier predictions.

### Are the five current class pairs good experimental choices?

They satisfy the five-unordered-pair requirement and cover all ten classes, but class coverage alone does not guarantee interpretable texture conflicts. Airplane/bird and car/cat provide potentially different surfaces; deer/dog and horse/monkey can share fur-like appearance; ship/truck often share painted manufactured surfaces. These are design concerns, not measured rankings. Previously inspected previews showed ambiguous ship/truck texture transfer and distortions in some car-style outputs. Trial replacing ship/truck with truck/bird or ship/cat using the same visual pilot protocol before changing the final configuration. Classes may appear in multiple pairs; covering each class exactly once is not required by the quoted task. Reverse directions may be harder and must be inspected too. Choose pairs based on visual validity before classifier inference, document pilot changes and acceptance rates, and do not claim the proposed alternatives are proven better.

### Can alpha differ by pair or direction?

Yes, as a documented generation choice set using visual pilots before model evaluation; the quoted assignment does not mandate a common alpha. Inspection of twenty downloaded alpha=0.7 previews (first two per direction) showed both weak transfers and strong distortions, including variation within a direction. Ship/truck share relatively smooth manufactured surfaces, so increasing alpha may increase color/background effects without creating a clearer texture conflict. Pilot identical source combinations at several strengths and freeze one setting per direction, keeping all evaluated models on identical images. Report the settings and recognize that comparisons across pairs now also reflect different strengths. Do not infer that alpha=1 is optimal from images generated only at alpha=0.7.

### Should we choose the rightmost alpha=1 preview?

In the supplied six-panel preview (content, style, alpha=0, 0.5, 0.75, 1), the rightmost output shows the strongest visible stylization and preserves a recognizable airplane silhouette. However, much of the transferred grain resembles the style image's background; recognizable bird plumage is not clear at preview resolution. Stronger whole-image stylization does not establish stronger bird-specific texture. Alpha=1 is a reasonable pilot setting to inspect on multiple exemplars, not automatic evidence that this sample qualifies. Inspect the saved individual output and use a predefined visual rule requiring recognizable content plus discernible intended texture, without classifier predictions. Keep ambiguous samples pending or reject them with a reason. Record style_label as the dataset class bird, not a more specific species inferred from appearance.

### Why does JSON reject PosixPath, and why were alpha variants missing?

Path objects provide filesystem operations; JSON stores basic values such as strings, numbers, lists and dictionaries. Store str(image_path) in metadata while retaining Path objects for filesystem operations. In the reviewed strength sweep, the saving and outputs.append blocks were outside the loop, so they would process only the final alpha. Indent image saving, metadata saving and outputs.append inside the loop; keep the combined preview after it. Explicitly call save_image for individual candidates too. A failed json.dump can leave a partial JSON file; rerunning the corrected write block replaces that file. No model retraining is needed for this serialization error.

### Why does the first airplane/bird stylization look like a color change?

In the supplied preview, airplane shape survives but recognizable bird texture is not evident. This is insufficient visual evidence for an accepted cue conflict; it does not alone prove a coding error. AdaIN matches whole feature-map channel statistics, without identifying the bird or isolating feathers. The blurry bird and substantial background may provide weak object-specific texture cues. The current alpha=0.5 also mixes stylized features equally with original content features.

For a diagnostic preview, keep the same pair and compare alpha=0, 0.5, 0.75 and 1.0. Alpha=0 tests encoder/decoder reconstruction and need not reproduce input pixels exactly; alpha=1 removes the original-content blend but does not guarantee a valid texture conflict. Save distinctly named pilot previews to avoid overwriting candidates. Inspect several texture-rich style exemplars, then freeze a documented generation/visual-rejection protocol before evaluating classifiers. Reject samples whose only visible change is color or whose content becomes unrecognizable; never use classifier predictions for acceptance. Reference implementation: https://github.com/naoto0804/pytorch-AdaIN.

### The generator prints Saved candidate, but where are the images?

The current generator writes to TASK_CONFLICT_DATASET_DIR, which is TASK_DIR/conflict_dataset. It does not write to the separate cue_conflict folder or the repository's scripts directory. In the reported Colab setup this is /content/drive/MyDrive/ATML/assignment_01/task1/conflict_dataset. Candidate conflict_2937_5093 produces conflict_2937_5093.png, preview_conflict_2937_5093.png and conflict_2937_5093.json. The preview is a separate saved image; save_image does not display it in the notebook. Print the configured absolute directory, list files and use PIL with IPython.display.display in a Colab cell to inspect it. A local editor cannot show remote Drive artifacts unless downloaded/synchronized.

### Why is selected_indices.npy missing during conflict generation?

The generator loads previously selected test IDs; loading does not create the file. Training does not necessarily call get_test_subset, which creates this artifact. After correcting both test_ds_full.targets references in make_subset.py to test_ds_full.labels for torchvision STL10, call get_test_subset(transformation_type='original') only if the configured index file is absent, then load the IDs. This prepares baseline images and IDs without running inference. Preserve any existing intended selection by locating/reusing its file rather than replacing it. Longer-term, separate lightweight ID selection from image transformation to avoid materializing 500 images just to obtain their IDs.

### Review of conflict_image_generator.py

The generator separates image generation from saving and creates a PNG plus a content/style/output preview. The weight directory now exists as data/external/model_weights (renamed from mode_weights). Syntax checks passed, but execution remains blocked by config referencing TASK_CONFLICT_DATASET_DIR before assignment; its parent should be TASK_DIR. The installed torchvision STL10 source uses dataset.labels, not dataset.targets, for class selection. download=True in the new script bypasses the staged download helper if data are missing; use prepare_stl10 followed by download=False, or download=False alone when data are already prepared. Add conflict_id and image_path to each JSON record. Avoid adding the conflict_ prefix twice to JSON filenames. This helper now returns tensors and metadata without saving .pt files; the generator currently saves PNGs only. Rerunning the same pair overwrites its filenames, so use distinct run/candidate identifiers when trying multiple alpha values. These are review findings, not automatic Python edits.

### How do I call StyleTransferModel for one candidate?

Instantiate it once with the compatible AdaIN encoder and decoder checkpoint paths, then call `stylizer(content_image, style_image, alpha=0.5)`. Each input is a floating-point RGB tensor [3,224,224] in [0,1], before classifier normalization. The wrapper adds a batch axis, extracts features, applies AdaIN and alpha mixing, decodes, removes the batch axis and clamps the result to [0,1] on CPU. Alpha=0 decodes content features (not necessarily exact input pixels); alpha=1 uses the full AdaIN-transformed features. Inspect quality rather than assuming a numerical alpha guarantees valid texture transfer.

Local checkpoints were found under data/external/mode_weights/vgg_normalised.pth and decoder.pth (the folder is spelled mode_weights). A first smoke example can use original IDs from selected_indices.npy to index the official test dataset, preserving ID meaning. Save the exact output tensor, a PNG for inspection and a JSON record with acceptance initially unset. Do not apply the universal PIL preprocessing again to an already resized tensor. Reuse the same saved output for all classifiers after visual acceptance. The corrected full input shape check was observed in this review.

### Can configured class pairs define a saved cue-conflict dataset?

Yes. The five CLASS_PAIRS in config match its STL10_CLASSES ordering. Each unordered pair creates two content/style directions, giving ten groups. A balanced 200-accepted-image target means 20 accepted examples per direction (40 per unordered pair); generate replacement candidates for visual rejections rather than counting rejected images toward 200. Define visual acceptance rules before inspecting model predictions.

Save each generated image and a manifest containing a unique conflict ID, relative file path, original content/style IDs, both labels, pair/direction, alpha, seed, acceptance status and rejection reason. PNG is suitable for viewing and loading; for exact float preservation, save tensors. Reuse the saved artifact across all methods and load only accepted records for inference. Retain rejected records for reporting. Compare features with the baseline content image later, using original dataset IDs rather than ambiguous subset positions.

Current generation blocker found during review: StyleTransferModel checks image.shape[1] against (3,224,224). A single axis size cannot equal a whole shape tuple. The intended check is tuple(image.shape) != (3,224,224). This correction was provided for manual typing, not applied automatically. Also note that generate_cue_conflicts currently unpacks two fields from dataset[index]; a three-field image/label/ID dataset needs an adapter or adjusted unpacking that preserves original IDs.

### Why does labels.shape[1] fail?

Labels have shape [N], so their shape tuple contains only position 0. Reading position 1 raises IndexError. Logits have shape [N,C], so logits.shape[1] gives the number of classes. Equal label-array shapes do not prove equal class-column meanings: compare the ordered classes metadata too. In the latest review, the paired sorting and consistency logic were correct, but a trailing equals sign in `import torch=` and the labels.shape[1] check prevented execution. Comments were added without changing executable code; fixes were provided for manual typing.

### Review of the first paired-comparison implementation

The saved implementation correctly checks nonempty one-dimensional unique IDs and equal sorted ID sets. However, logits and labels must be loaded from each result before indexing them: assigning `baseline_logits = baseline_logits[...]` without initialization raises UnboundLocalError. Inference records and calculated metric dictionaries are different objects; do not assume raw inference results have a top-level `accuracy_pct` field. Compute accuracy from validated aligned labels/logits, or explicitly calculate standalone metrics first.

After loading, validate [N,C] score shapes, [N] label shapes, row counts, finite scores and valid class IDs. Ensure the same class ordering and model/decision method/checkpoint, then reorder each run using its own sorted indices. Ground-truth labels must agree after alignment for grayscale/hue. Compute predictions using argmax across class axis 1, compare predictions to labels for accuracy, and compare baseline predictions to transformed predictions for consistency. Name accuracy differences with `_pp` (percentage points), rather than implying relative percentage change.

### Does comparing sorted IDs reorder the results?

For one-dimensional IDs, checking uniqueness and comparing `baseline_ids.sort().values` with `transformed_ids.sort().values` validates that the same unique images occur in both runs, regardless of their original order. It allows different orders through validation but does not align the result rows. `sort()` returns sorted values and their original indices; it does not modify the original tensor in place.

For baseline IDs [20,10,30] and transformed IDs [30,20,10], both sorted values are [10,20,30]. baseline sorting indices are [1,0,2]; transformed sorting indices are [2,1,0]. Apply each run's own indices to its labels, logits and any features being compared to put both runs in ascending image-ID order. Never sort IDs alone and then compare untouched predictions. Validate one-dimensional nonempty IDs, matching row counts and matching aligned ground-truth labels too.

- Can 100% consistency coexist with zero accuracy? Why?
- Why do IDs matter if baseline and transformed arrays have the same length?
- For [500,10], what changes when reducing dim=0 versus dim=1?
- Why does macro-F1 differ from computing F1 from macro precision and recall?
- What information is lost when saving only winning class IDs rather than logits?
- Why must shape bias be accompanied by coverage?
- Why can stable predictions coexist with changing feature vectors?
