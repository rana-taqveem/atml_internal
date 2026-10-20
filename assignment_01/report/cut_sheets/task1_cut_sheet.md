# Cut sheet: Task 1

Mark a sentence for deletion by putting `x` in the Cut column. Tier A = recommended cut, tier B = optional. Untagged sentences are CORE. Notes marked LINK mean a neighbouring sentence refers back to this one.


## Summary

| Subsection | Words | Tier A cut | Tier B cut | After A | After A+B |
|---|---|---|---|---|---|
| Clean Baseline vs. Visual Interventions | 19 | 0 | 0 | 19 | 19 |
| Model's Reliance on Shape, Texture, and Color | 443 | 149 | 70 | 294 | 224 |
| Spatial and Positional Invariance | 410 | 51 | 63 | 359 | 296 |
| Representation Space Analysis | 496 | 192 | 76 | 304 | 228 |
| Architecture and Observation Relationships | 403 | 71 | 67 | 332 | 265 |
| **Total** | **1771** | **463** | **276** | **1308** | **1032** |

## Clean Baseline vs. Visual Interventions

| # | Para | Words | Tag | Cut | Sentence | Note |
|---|---|---|---|---|---|---|
| 1 | P1 | 19 | UNFINISHED |  | As summarized in Table~\ref{tab:task1_interventions}, all three architectures maintain high clean accuracy, but their sensitivity diverges substantially under visual interventions... | ends in '...': finish it or delete it |

## Model's Reliance on Shape, Texture, and Color

| # | Para | Words | Tag | Cut | Sentence | Note |
|---|---|---|---|---|---|---|
| 2 | P1 | 33 | CORE |  | Model performance under interventions like color changes (greyscaling or hue modifications) showed no substantial drop in overall performance, suggesting that none of the models gave much importance to the color when making predictions. |  |
| 3 | P1 | 20 | REPEATS TABLE B |  | Removing color altogether cost between 1.4 and 2.4 points, and changing color impacted even less, from 0 to 1.6 points. | summarises Table 1 values; worth keeping with a Table ref |
| 4 | P1 | 24 | CORE |  | Although, under the greyscale all four models broke more predictions than they managed to fix, only the zero-shot CLIP change was statistically more reliable. |  |
| 5 | P1 | 19 | DUPLICATE A |  | So we cannot claim that the color interventions meaningfully resulted in degraded performance under the given configuration and dataset. | restates sentence 2 |
| 6 | P2 | 24 | FILLER A |  | Things did get a bit more interesting with the cue conflicts, where the texture of one class is pushed over the shape of another. | lead-in |
| 7 | P2 | 35 | CORE |  | Overall, all of the models showed a strong inclination towards shape bias, with ResNet being the least biased towards shape and appearing to give some importance to texture as well as compared to the rest. |  |
| 8 | P2 | 24 | FILLER B |  | An important factor to highlight here is that the coverage of each model on the cue conflicts gives a different base for these percentages. | lead-in to 11-12 |
| 9 | P2 | 22 | DUPLICATE A |  | As, for ResNet the shape bias is computed from 160 predictions, while for the other models it is computed from about 180. | same point as 11 |
| 10 | P2 | 31 | DEFINITION A |  | Coverage is the share of conflicts on which a model picked one of the two cued classes, so a lower coverage means the model committed to a cue on fewer images. | coverage is defined in the Table 2 caption |
| 11 | P2 | 28 | CORE |  | The models, ViT, the CLIP head, and zero-shot CLIP, predicted a third class on only 17 to 20 of the 200 conflicts, while ResNet did so on 40. |  |
| 12 | P2 | 26 | REPEATS TABLE B |  | Because of this, ResNet's shape bias is measured less precisely than the others, with a 95\% range of 81.4 to 91.9 against ViT's 95.6 to 99.5. | intervals are in Table 2 |
| 13 | P2 | 37 | CORE |  | So, we still cannot conclusively say whether ResNet's lower coverage follows from its greater reliance on texture or from some other weakness, since a third-class prediction only shows that neither cue won, not which cue caused it. |  |
| 14 | P3 | 21 | FILLER A |  | With all of that, it is worth looking at whether texture contributed at all in models' prediction flow and how strongly. | lead-in |
| 15 | P3 | 5 | FILLER A |  | Texture rank gives us that. | lead-in; 16 then names texture rank directly |
| 16 | P3 | 62 | CORE |  | Based on the texture-rank findings, for the three trained heads (ResNet $+14.5$, CLIP head $+12.0$, ViT $+7.0$ points) the texture class rose into the top two of the ranked class scores more often under conflict than on the clean content image, so texture does influence their scores even though shape still dominates the final decision; zero-shot CLIP showed no such reliable movement. |  |
| 17 | P4 | 27 | DEFINITION A |  | Coverage reflects the fraction of cue-conflict images for which the model predicts either the correct shape or texture class, rather than defaulting to an uninformative third category. | second definition of coverage (see 10) |
| 18 | P4 | 5 | UNFINISHED |  | As shown in Table~\ref{tab:task1_shape_bias}, ... | ends in '...': finish it or delete it |

## Spatial and Positional Invariance

| # | Para | Words | Tag | Cut | Sentence | Note |
|---|---|---|---|---|---|---|
| 19 | P1 | 11 | CORE |  | Models show almost no drop in accuracy and performance with translations. |  |
| 20 | P1 | 49 | DETAIL B |  | Out of the 48 translation tests (8, 16, and 32 pixels across four directions and four methods), only 8 resulted in 10 or more predictions flipping to negative, with a maximum of 16 (zero-shot CLIP at 16 pixels left), and 7 of these 8 cases belong to zero-shot CLIP. | fine-grained counts; 19 + 22 carry the point |
| 21 | P1 | 22 | DUPLICATE A |  | Nevertheless, given just 500 evaluation images and only a few flipped predictions, this analysis lacks enough statistical power to detect minor effects. | same caveat as 22 |
| 22 | P1 | 25 | CORE |  | The confidence intervals permit a performance decline of up to approximately 4 points, thus ruling out major degradation from translation while leaving smaller losses possible. |  |
| 23 | P2 | 5 | FILLER A |  | Things change with patch shuffling. | lead-in |
| 24 | P2 | 14 | DUPLICATE B |  | All models suffered a substantial loss in accuracy, and this is consistent across models. | 25-27 say the same with numbers |
| 25 | P2 | 36 | CORE |  | All four drops were statistically reliable, with ResNet and ViT models' performance declining by 6.2 to 7.8 points, where ResNet failed on 39 correct baseline predictions and was unable to correct a single incorrect baseline prediction. |  |
| 26 | P2 | 29 | CORE |  | The CLIP models appeared to be hit the worst, with the CLIP head breaking 75 predictions while fixing only 6, and overall performance decreasing by 12.8 to 13.8 points. |  |
| 27 | P2 | 15 | CORE |  | This difference is reliable, as both CLIP methods lost significantly more than ResNet and ViT. |  |
| 28 | P3 | 24 | FILLER A |  | Between translation and patch shuffling, the variation in model performance can be attributed to the different types of spatial changes introduced by each intervention. | lead-in |
| 29 | P3 | 14 | CORE |  | During the translation interventions, most of the global image and shape structure is preserved. |  |
| 30 | P3 | 31 | CORE |  | Models still have access to enough global representation, evident from the feature similarity between 0.93 and 0.98 when comparing baseline and translated images, and they therefore retain their accuracy and consistency. |  |
| 31 | P3 | 30 | CORE |  | This changes substantially with patch shuffling, which moves the patches around and destroys the global spatial arrangement: representational similarity falls to between 0.63 and 0.75, and all models lose accuracy. |  |
| 32 | P3 | 42 | CORE |  | The correspondence holds when comparing the two interventions, but not when comparing models, since CLIP keeps the highest similarity under shuffling (0.748) while losing the most accuracy ($-13.8$ points), and ViT has the lowest similarity (0.629) with the smallest loss ($-6.2$ points). |  |
| 33 | P3 | 39 | CORE |  | Furthermore, local features retain significant diagnostic value independently: since patch shuffling keeps every pixel and local texture within each block intact, the models maintain an accuracy of 84.0\% to 92.6\% despite the loss of spatial layout---substantially above chance level. |  |
| 34 | P3 | 24 | CORE |  | This demonstrates that local texture information sustains the majority of the classification performance, whereas global structure provides the final 6.2 to 13.8 percentage points. |  |

## Representation Space Analysis

| # | Para | Words | Tag | Cut | Sentence | Note |
|---|---|---|---|---|---|---|
| 35 | P1 | 17 | REPEATS FIGURE A |  | The feature representation similarity for all models remains quite stable across translations when compared with the baseline. | Figure 2(b) shows this |
| 36 | P1 | 14 | REPEATS FIGURE A |  | The similarity drops a little for the color interventions but still stays above 0.74. | Figure 2(b) shows this |
| 37 | P1 | 47 | REPEATS FIGURE B |  | However, a substantial decline is observed for patch shuffling, to 0.63 for ViT, 0.70 for ResNet, and 0.75 for CLIP, and the cue conflicts caused the sharpest drop of all, down to 0.44 for ResNet and 0.48 for ViT, while CLIP still maintained a relatively high 0.78. | Figure 2(b); 39 re-quotes the ResNet value |
| 38 | P2 | 20 | DUPLICATE A |  | In the case of the translation interventions, the features barely move, staying between 0.93 and 0.98, and accuracy remains stable. | repeats 30 |
| 39 | P2 | 34 | CORE |  | The clearest divergence appears with the cue conflicts, where ResNet's representation moves the furthest of all (dropping to 0.442), while its predictions change only slightly, since 139 of 200 still follow the shape class. |  |
| 40 | P2 | 32 | CORE |  | A possible explanation is that cosine similarity measures how far the representation moved, not what it still contains, so a large movement does not necessarily remove the information the classifier relies on. |  |
| 41 | P2 | 10 | CORE |  | In these cases, prediction stability and representation stability come apart. |  |
| 42 | P3 | 38 | CORE |  | Another interesting finding is observed for ViT on the cue conflicts, where the similarity is 0.505 for stable predictions against 0.264 for flipped ones, showing that the images whose predictions flipped had moved further than those that held. |  |
| 43 | P3 | 20 | CORE |  | However, under patch shuffling that gap nearly vanishes (0.014 for ResNet), so feature movement barely explains which predictions broke there. |  |
| 44 | P3 | 14 | CORE |  | Representation change and prediction change therefore agree within a condition, while diverging across conditions. |  |
| 45 | P4 | 34 | DUPLICATE A |  | CLIP managed to keep the most stable representation under every intervention, yet it lost the most accuracy under shuffling ($-13.8$ points), while ViT moved furthest (0.629 under shuffling) and lost the least ($-6.2$ points). | repeats 32 |
| 46 | P5 | 57 | FILLER A |  | [CLIP vs CLIP Zero-Shot.]  While analysing the models' dependency on feature representation, we can take the two CLIP methods, which share the same backbone and therefore the same learned image embeddings, one with a head trained on STL-10 and the other using zero-shot prompts, and compare their accuracies, their agreement, and their sensitivity to shape versus texture. | long lead-in; the paragraph title already says CLIP vs zero-shot |
| 47 | P5 | 40 | DUPLICATE A |  | Looking at accuracy, the two mostly agree, where agreement means both predicted the same class for the same image: they agree closely on the baseline, the translations and the colour changes, and their accuracy and agreement ratios are almost similar. | 'they mostly agree' is implied by 48-49 |
| 48 | P5 | 10 | FILLER A |  | The visible differences come from patch shuffle and cue conflicts. | lead-in |
| 49 | P5 | 43 | CORE |  | Under patch shuffle the accuracy numbers of the two are identical (84.0\%), yet their agreement on the predicted class drops to 88.2\%, meaning they disagreed on 59 of 500 images, with 22 correct only for the head and 22 correct only for zero-shot. |  |
| 50 | P5 | 37 | CORE |  | Under cue conflicts agreement is similarly low at 88.0\%, and the texture-rank analysis shows that the transferred texture reliably raised the texture class in the head's top two ($+12.0$ points) but not in zero-shot's ($+0.5$, not reliable). |  |
| 51 | P5 | 29 | DUPLICATE B |  | So even though both share the same backbone embeddings, the two decision rules read those same representations differently, and the trained head is the more texture-sensitive of the two. | same conclusion as 57 |

## Architecture and Observation Relationships

| # | Para | Words | Tag | Cut | Sentence | Note |
|---|---|---|---|---|---|---|
| 52 | P1 | 10 | FILLER A |  | The results observed during the experimentation differ across the models. | lead-in; LINK: 53 starts 'These models' - fix if you cut 52 |
| 53 | P1 | 11 | CORE |  | These models differ across their architecture, pretraining data, supervision, and capacity. |  |
| 54 | P1 | 10 | FRAGMENT A |  | As in the case of ResNet and ViT against CLIP. | not a full sentence |
| 55 | P1 | 13 | CORE |  | Such cross-model differences are therefore confounded and cannot be attributed to architecture alone. |  |
| 56 | P2 | 27 | CORE |  | Across this experimentation, the one causal claim that can be made about the decision rule comes from comparing CLIP with a trained linear head against zero-shot CLIP. |  |
| 57 | P2 | 28 | CORE |  | Since both share the same frozen backbone, the same pretraining and identical image embeddings, the observed differences must be driven by the decision rule rather than the representation. |  |
| 58 | P2 | 51 | DUPLICATE A |  | This is visible under patch shuffle, where both reached the same accuracy of 84.0\% yet disagreed on 59 of 500 images, and again on the cue conflicts, where the transferred texture reliably raised the texture class into the head's top two ($+12.0$ points) but produced no reliable movement for zero-shot ($+0.5$). | repeats 49-50 almost word for word |
| 59 | P3 | 11 | CORE |  | A partial architectural link can be drawn between ResNet and ViT. |  |
| 60 | P3 | 21 | CORE |  | Both were pretrained on the same dataset (ImageNet-1k) with the same supervision (class labels), so data and supervision are held constant. |  |
| 61 | P3 | 49 | CORE |  | ResNet's reliably lower shape bias (86.9\% vs 97.8\%) and lower coverage (80.0\% vs 91.5\%, driven by 40 third-class predictions concentrated in cases like monkey-to-truck and deer-to-car) indicate a relatively greater reliance on texture than the other three, which is consistent with the convolutional inductive bias reported in the literature. |  |
| 62 | P3 | 30 | CORE |  | However, the two models also differ in capacity (25M vs 87M parameters) and training recipe (the \texttt{IMAGENET1K\_V2} ResNet weights use heavier augmentation), so architecture cannot be isolated as the cause. |  |
| 63 | P3 | 29 | DETAIL B |  | The shape bias built into our conflict set raises the absolute level for every method equally, so it explains the high values overall but not the difference between models. | side point about the conflict set |
| 64 | P4 | 53 | CORE |  | Conversely, CLIP's substantially larger accuracy degradation under patch shuffle ($-12.8$ to $-13.8$ points, compared with $-6.2$ to $-7.8$ for ViT and ResNet) cannot be assigned to architecture, since CLIP differs from ViT in patch size (32 px vs 16 px), supervision (multimodal contrastive) and pretraining scale (400M web pairs) at the same time. |  |
| 65 | P4 | 22 | CORE |  | The fact that both CLIP variants suffer equally does show that the effect comes from the representation rather than the decision rule. |  |
| 66 | P5 | 38 | FUTURE WORK B |  | Definitively isolating these underlying mechanisms would require controlled ablations holding non-architectural factors fixed, such as comparing a supervised ViT-B/16 directly against a CLIP-trained ViT-B/16, or training CNN and Transformer backbones under identical dataset sizes, augmentations, and optimisation recipes. | nice but optional |
