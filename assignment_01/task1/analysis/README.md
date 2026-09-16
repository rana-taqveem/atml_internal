# Task 1 learning notes and question log

Track implementation and experiment completion in [Task 1 checklist](TASK1_CHECKLIST.md). A checked implementation item does not establish that final experimental evidence has been generated.

These are revision notes from our implementation discussions, not experimental results or report prose. Keep adding questions, explanations, examples and implementation decisions as we progress. User preference: provide implementation code in chat for manual typing; edit code only when explicitly requested. Documentation updates are requested.

## 1. How should I approach ML analysis?

Start with a question or hypothesis, identify the controlled change, choose a metric that measures the effect, preserve comparable inputs, then examine aggregates and individual failures. Keep image generation, inference, metric calculation and presentation separate. Save enough metadata to reproduce each result. Do not select examples or settings based on desired test outcomes.

The frozen backbone converts images to feature vectors. A trained linear head converts features to class scores. At inference, new images still need feature extraction. A head checkpoint contains learned weights/bias, not the training images' feature vectors. An nn.Linear layer has no additional hidden layer.

## 2. Questions about preprocessing and interventions

- **Does ToTensor scale to [0,1]?** For ordinary uint8 RGB PIL images, yes: values 0..255 become floating-point values 0..1 and layout becomes [C,H,W]. This is not a universal guarantee for every input dtype or image mode.
- **Is this model normalization?** No. Model normalization usually subtracts channel means and divides by channel standard deviations. Apply it exactly once, after interventions; account for wrappers that already normalize inside forward.
- **What order do we use?** RGB conversion -> common 224x224 size -> tensor -> one intervention -> model-specific normalization -> backbone -> decision method. Compare each condition with clean inputs rather than chaining unrelated interventions.
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

## 6. Next lesson: paired clean/intervention comparisons

Source: ATML-PA1.pdf, Task 1 steps 1, 2, 4 and 5. Start with clean versus grayscale, then reuse for hue. Compare the same model and decision method, checkpoint, class ordering and image IDs.

Accuracy change (percentage points) = transformed accuracy (%) - clean accuracy (%).
Accuracy drop (percentage points) = clean accuracy (%) - transformed accuracy (%).
Consistency (%) = 100 * count(clean prediction == transformed prediction) / N.

Example: 90% clean and 80% transformed means a 10-percentage-point drop, not a 10% relative decrease. Relative decrease would be 100*(90-80)/90 = 11.11%.

Consistency compares predictions to predictions; accuracy compares predictions to ground truth. A consistently wrong model can have 100% consistency. Equal accuracies can hide changed predictions because gains and losses can cancel.

Pair by image ID. Equal array lengths do not prove matching images. For the first implementation, require identical IDs in identical order and matching labels. If ordering differs, explicitly align by ID in a later helper rather than comparing mismatched rows. Reject duplicates and mismatched sets; do not silently take an intersection.

Manual implementation exercise: add a separate compare_with_clean(clean_result, transformed_result) function; retain calcultate_metrics for standalone metrics. Read labels/logits/IDs, validate pairing, derive predictions, compute both accuracies and consistency, and return clearly named fields. No paired-comparison code has been added automatically.

## 7. Remaining Task 1 analysis roadmap

1. Clean baseline: all three heads and zero-shot CLIP; top-1, macro-F1, confidence.
2. Color: grayscale and chosen additional color condition; absolute performance, accuracy change, consistency.
3. Cue conflicts: at least 200 valid images, at least five unordered pairs, both directions when feasible; visual rejection rule set before evaluation, accepted/rejected counts; shape/texture/other counts, shape bias and coverage. Shape bias = shape/(shape+texture); coverage = (shape+texture)/total. If no shape/texture decisions exist, report shape bias as undefined, not a substantive zero preference.
4. Translation: 0/8/16/32 pixels in four cardinal directions, reflection padding and shifted crop; average across directions and plot accuracy and consistency against displacement.
5. Patch shuffle: 4x4 grid, non-identity permutation per image, seed 6304, identical shuffled images across models; accuracy drop and consistency.
6. Representation analysis: paired cosine stability for grayscale, cue conflicts, translation and patch shuffle; joint clean/transformed t-SNE or UMAP per backbone with fixed IDs and seed, class colors and condition markers. Do not compare coordinates of separately fitted projections.
7. Inspect informative cue-conflict agreements/disagreements/failures; distinguish prediction stability from feature stability and architecture effects from pretraining/data/capacity confounds. Interpret actual results yourself after running experiments.

## Revision questions

### Review of conflict_image_generator.py

The generator separates image generation from saving and creates a PNG plus a content/style/output preview. The weight directory now exists as data/external/model_weights (renamed from mode_weights). Syntax checks passed, but execution remains blocked by config referencing TASK_CONFLICT_DATASET_DIR before assignment; its parent should be TASK_DIR. The installed torchvision STL10 source uses dataset.labels, not dataset.targets, for class selection. download=True in the new script bypasses the staged download helper if data are missing; use prepare_stl10 followed by download=False, or download=False alone when data are already prepared. Add conflict_id and image_path to each JSON record. Avoid adding the conflict_ prefix twice to JSON filenames. This helper now returns tensors and metadata without saving .pt files; the generator currently saves PNGs only. Rerunning the same pair overwrites its filenames, so use distinct run/candidate identifiers when trying multiple alpha values. These are review findings, not automatic Python edits.

### How do I call StyleTransferModel for one candidate?

Instantiate it once with the compatible AdaIN encoder and decoder checkpoint paths, then call `stylizer(content_image, style_image, alpha=0.5)`. Each input is a floating-point RGB tensor [3,224,224] in [0,1], before classifier normalization. The wrapper adds a batch axis, extracts features, applies AdaIN and alpha mixing, decodes, removes the batch axis and clamps the result to [0,1] on CPU. Alpha=0 decodes content features (not necessarily exact input pixels); alpha=1 uses the full AdaIN-transformed features. Inspect quality rather than assuming a numerical alpha guarantees valid texture transfer.

Local checkpoints were found under data/external/mode_weights/vgg_normalised.pth and decoder.pth (the folder is spelled mode_weights). A first smoke example can use original IDs from selected_indices.npy to index the official test dataset, preserving ID meaning. Save the exact output tensor, a PNG for inspection and a JSON record with acceptance initially unset. Do not apply the universal PIL preprocessing again to an already resized tensor. Reuse the same saved output for all classifiers after visual acceptance. The corrected full input shape check was observed in this review.

### Can configured class pairs define a saved cue-conflict dataset?

Yes. The five CLASS_PAIRS in config match its STL10_CLASSES ordering. Each unordered pair creates two content/style directions, giving ten groups. A balanced 200-accepted-image target means 20 accepted examples per direction (40 per unordered pair); generate replacement candidates for visual rejections rather than counting rejected images toward 200. Define visual acceptance rules before inspecting model predictions.

Save each generated image and a manifest containing a unique conflict ID, relative file path, original content/style IDs, both labels, pair/direction, alpha, seed, acceptance status and rejection reason. PNG is suitable for viewing and loading; for exact float preservation, save tensors. Reuse the saved artifact across all methods and load only accepted records for inference. Retain rejected records for reporting. Compare features with the clean content image later, using original dataset IDs rather than ambiguous subset positions.

Current generation blocker found during review: StyleTransferModel checks image.shape[1] against (3,224,224). A single axis size cannot equal a whole shape tuple. The intended check is tuple(image.shape) != (3,224,224). This correction was provided for manual typing, not applied automatically. Also note that generate_cue_conflicts currently unpacks two fields from dataset[index]; a three-field image/label/ID dataset needs an adapter or adjusted unpacking that preserves original IDs.

### Why does labels.shape[1] fail?

Labels have shape [N], so their shape tuple contains only position 0. Reading position 1 raises IndexError. Logits have shape [N,C], so logits.shape[1] gives the number of classes. Equal label-array shapes do not prove equal class-column meanings: compare the ordered classes metadata too. In the latest review, the paired sorting and consistency logic were correct, but a trailing equals sign in `import torch=` and the labels.shape[1] check prevented execution. Comments were added without changing executable code; fixes were provided for manual typing.

### Review of the first paired-comparison implementation

The saved implementation correctly checks nonempty one-dimensional unique IDs and equal sorted ID sets. However, logits and labels must be loaded from each result before indexing them: assigning `clean_logits = clean_logits[...]` without initialization raises UnboundLocalError. Inference records and calculated metric dictionaries are different objects; do not assume raw inference results have a top-level `accuracy_pct` field. Compute accuracy from validated aligned labels/logits, or explicitly calculate standalone metrics first.

After loading, validate [N,C] score shapes, [N] label shapes, row counts, finite scores and valid class IDs. Ensure the same class ordering and model/decision method/checkpoint, then reorder each run using its own sorted indices. Ground-truth labels must agree after alignment for grayscale/hue. Compute predictions using argmax across class axis 1, compare predictions to labels for accuracy, and compare clean predictions to transformed predictions for consistency. Name accuracy differences with `_pp` (percentage points), rather than implying relative percentage change.

### Does comparing sorted IDs reorder the results?

For one-dimensional IDs, checking uniqueness and comparing `clean_ids.sort().values` with `transformed_ids.sort().values` validates that the same unique images occur in both runs, regardless of their original order. It allows different orders through validation but does not align the result rows. `sort()` returns sorted values and their original indices; it does not modify the original tensor in place.

For clean IDs [20,10,30] and transformed IDs [30,20,10], both sorted values are [10,20,30]. Clean sorting indices are [1,0,2]; transformed sorting indices are [2,1,0]. Apply each run's own indices to its labels, logits and any features being compared to put both runs in ascending image-ID order. Never sort IDs alone and then compare untouched predictions. Validate one-dimensional nonempty IDs, matching row counts and matching aligned ground-truth labels too.

- Can 100% consistency coexist with zero accuracy? Why?
- Why do IDs matter if clean and transformed arrays have the same length?
- For [500,10], what changes when reducing dim=0 versus dim=1?
- Why does macro-F1 differ from computing F1 from macro precision and recall?
- What information is lost when saving only winning class IDs rather than logits?
- Why must shape bias be accompanied by coverage?
- Why can stable predictions coexist with changing feature vectors?
