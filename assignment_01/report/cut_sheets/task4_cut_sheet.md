# Cut sheet: Task 4

Mark a sentence for deletion by putting `x` in the Cut column. Tier A = recommended cut, tier B = optional. Untagged sentences are CORE. Notes marked LINK mean a neighbouring sentence refers back to this one.


## Summary

| Subsection | Words | Tier A cut | Tier B cut | After A | After A+B |
|---|---|---|---|---|---|
| Semantic Similarity and Rejection | 189 | 0 | 11 | 189 | 178 |
| What the Novelty Scores Capture | 425 | 111 | 67 | 314 | 247 |
| Did GCSC Improve Closed-Set and Open-Set Performance? | 177 | 0 | 0 | 177 | 177 |
| Did PROSER Improve Rejection, and What Was the CSA Trade-off? | 276 | 0 | 0 | 276 | 276 |
| **Total** | **1067** | **111** | **78** | **956** | **878** |

## Semantic Similarity and Rejection

| # | Para | Words | Tag | Cut | Sentence | Note |
|---|---|---|---|---|---|---|
| 1 | P1 | 14 | CORE |  | The semantic similarity of certain classes made the rejection substantially harder for the model. |  |
| 2 | P1 | 31 | CORE |  | With the vanilla MLS measure of unknownness, the model rejected only \(34.75\%\) near unknowns with an AUROC of \(0.802\), whereas it rejected around \(54.13\%\) far unknowns with an AUROC of \(0.883\). |  |
| 3 | P1 | 11 | DUPLICATE B |  | Therefore, the model frequently accepted unknown classes that resemble CIFAR-10 classes. | restates 1 |
| 4 | P2 | 29 | CORE |  | The most frequently accepted unknown classes were among pickup trucks (\(91\%\) accepted, most often as automobiles), bus (\(82\%\), most often as truck), and wolf (\(77\%\), most often as dog). |  |
| 5 | P2 | 36 | CORE |  | Similarly, the model confused wolves as dogs and leopards as cats, indicating that the visual and semantic similarity among some of the unknown categories with known ones is quite plausible, making rejection difficult for the model. |  |
| 6 | P3 | 20 | CORE |  | Among some of the semantically far unknowns, the wardrobe (\(68\%\)), bowl (\(53\%\)), and sunflower (\(47\%\)) had the highest acceptance rates. |  |
| 7 | P3 | 12 | CORE |  | Additional confident failures are the keyboard/mushroom by bird and chair by airplane. |  |
| 8 | P3 | 36 | CORE |  | The high acceptance rates for these failed semantically far unknowns suggest that the classifier might have responded to incidental similarities in shape, color or some other spurious correlations like background rather than some discriminative class features. |  |

## What the Novelty Scores Capture

| # | Para | Words | Tag | Cut | Sentence | Note |
|---|---|---|---|---|---|---|
| 9 | P1 | 21 | CORE |  | MSP measures relative softmax confidence, where the dominant probability is considered as known-like, and the probabilities spread across classes are unknowns. |  |
| 10 | P1 | 10 | DEFINITION B |  | Given as MSP unknownness is: \[ u_{\mathrm{MSP}}(x) = 1-\max_kp_k(x). \] | MSP formula; given in the assignment |
| 11 | P1 | 10 | CORE |  | As compared to MSP, MLS measures the largest absolute logit. |  |
| 12 | P1 | 15 | DETAIL B |  | The maximum logit represents the strongest raw response of the classifier among the known classes. | extra explanation of MLS; 11 already says what it captures |
| 13 | P1 | 2 | DEFINITION B |  | Given as: | cut together with 14 (MLS formula) |
| 14 | P2 | 3 | DEFINITION B |  | \(u_{\mathrm{MLS}}(x) = -\max_kz_k(x).\) | cut together with 13 (MLS formula) |
| 15 | P3 | 24 | DEFINITION A |  | Here, the negative sign makes the largest value the smallest, representing strength of unknownness: the smaller the \(u_{\mathrm{MLS}}(x)\) the more known the class is. | explains the minus sign |
| 16 | P3 | 24 | CORE |  | Where MSP and MLS uses the max single value to compute the unknownness, Energy uses all the classifier logits to compute unknownness, given as: |  |
| 17 | P4 | 6 | DEFINITION B |  | \[ u_{\mathrm{Energy}}(x) = -\log \sum_{k=1}^{K}e^{z_k(x)}. \] | Energy formula; LINK: 16 ends 'given as:' - fix if you cut 17 |
| 18 | P4 | 5 | DEFINITION A |  | For CIFAR-10: \[ K=10. \] | K=10 on its own line |
| 19 | P4 | 8 | DEFINITION A |  | The expression: \[ \log\sum_ke^{z_k} \] is called logsumexp. | names log-sum-exp on its own line |
| 20 | P4 | 23 | DEFINITION A |  | For large known-class logits, it produces a large total known-class response, which, through the negative, becomes smaller; the small energy means more known. | explains the minus sign |
| 21 | P4 | 29 | CORE |  | Mahalanobis measures distance from the nearest known-class feature cluster by calculating a shared diagonal covariance from the training features For each class: \[ D_c(x) = (f(x)-\mu_c)^T \Sigma^{-1} (f(x)-\mu_c). \] |  |
| 22 | P4 | 17 | DEFINITION A |  | Because the covariance is diagonal, this becomes: \[ D_c(x) = \sum_{j=1}^{512} \frac{ (f_j(x)-\mu_{c,j})^2 }{ \sigma_j^2+\epsilon }. \] | diagonal expansion of the formula |
| 23 | P4 | 7 | DEFINITION B |  | And Mahalanobis novelty score is given as: | cut together with 24 (Mahalanobis score formula) |
| 24 | P5 | 24 | DEFINITION B |  | \(u_{\mathrm{Mah}}(x) = \min_cD_c(x).\) I.e., if the feature f(x) is close to at least one known class cluster, the score will be small; otherwise, large. | cut together with 23; 21 already says what it measures |
| 25 | P6 | 32 | CORE |  | Novelty scores for all four methods show that classifying far unknowns as unknown was much easier than near unknowns; MLS and energy methods produced almost identical results \(0.802/0.883\) and \(0.802/0.883\) near/far AUROC. |  |
| 26 | P6 | 7 | CORE |  | This indicates that the largest logit dominated. |  |
| 27 | P6 | 27 | CORE |  | MSP was most effective for near unknowns, with the highest AUROC of \(0.814\), which indicates that the relative probability has marginally more information for semantically similar unknowns. |  |
| 28 | P7 | 15 | CORE |  | Mahalanobis produced the best far AUROC (\(0.911\)) and the best overall AUROC (\(0.853\)) as well. |  |
| 29 | P7 | 13 | CORE |  | It shows that for semantically far classes, the feature distance was particularly informative. |  |
| 30 | P7 | 38 | CORE |  | However, on the contrary, it depicted the weakest near rejection because the near unknowns lie closer to the known clusters, as in the case of wolves and buses, which can lie close to dog or vehicle feature clusters. |  |
| 31 | P8 | 26 | DUPLICATE A |  | Thus, the MSP confidence score becomes the most appropriate ranking metric for near unknowns, whereas the feature distance from Mahalanobis was the strongest for far unknowns. | same conclusion as 33 |
| 32 | P8 | 8 | DUPLICATE A |  | MLS and Energy added little in this experiment. | same conclusion as 33 |
| 33 | P9 | 31 | CORE |  | Therefore, MSP softmax confidence was most useful for ranking near unknowns, feature distance was strongest for far unknowns, and using all logits through Energy added little beyond MLS in this experiment. |  |

## Did GCSC Improve Closed-Set and Open-Set Performance?

| # | Para | Words | Tag | Cut | Sentence | Note |
|---|---|---|---|---|---|---|
| 34 | P1 | 31 | CORE |  | GCSC shows a slight improvement on closed-set performance, where it gained 0.58 points over the vanilla model, which is about 58 images out of 10,000 test images, not a big number. |  |
| 35 | P1 | 27 | CORE |  | Since we used only one seed for the dataset, this improvement is not something on which we can assert the claim that GCSC improves the closed-set performance. |  |
| 36 | P1 | 20 | CORE |  | On open-set, the ordered ranking score AUROC on near and far shows a tiny improvement of 0.024 and 0.027, respectively. |  |
| 37 | P2 | 41 | CORE |  | While the AUROC gains were roughly the same for both near and far unknowns, at the rejection threshold for near and far unknowns model reports a gain of 2.38 points on near unknown rejections and 4.88 points on far unknown rejections. |  |
| 38 | P2 | 30 | CORE |  | This suggests that the augmentation might have encouraged a kind of representation that is less dependent on superficial fluctuations in appearances and thus improved known-class recognition and detection of unknowns. |  |
| 39 | P2 | 28 | CORE |  | However, the augmentation still does not fully resolve the semantic overlap between classes like bus and truck or wolf and dog, and near-semantic novelty remains the harder problem. |  |

## Did PROSER Improve Rejection, and What Was the CSA Trade-off?

| # | Para | Words | Tag | Cut | Sentence | Note |
|---|---|---|---|---|---|---|
| 40 | P1 | 24 | CORE |  | The PROSER method via MLS reported an AUROC of 0.845, a 0.003 increase over Vanilla (0.842); however, it's still below GCSC (0.868) by 0.023. |  |
| 41 | P1 | 24 | CORE |  | It reports near and far rejection rates as \(32.75\%\) and \(50.00\%\) respectively, on the lower side compared to both Vanilla \(34.75\%/54.13\%\) and GCSC \(37.13\%/59.00\%\). |  |
| 42 | P1 | 15 | CORE |  | The closed-set accuracy of \(94.63\%\) is roughly same as of Vanilla (\(94.54\%\)) and GCSC (\(95.12\%\)). |  |
| 43 | P2 | 33 | CORE |  | PROSER's own placeholder scores were also weaker, especially on near unknowns compared to MLS, with an AUROC of 0.780 on near and 0.886 on far, and rejection rates of \(29.88\%\) and \(48.25\%\), respectively. |  |
| 44 | P2 | 50 | CORE |  | The PROSER was fine-tuned for 50 epochs; however, the highest validation accuracy was achieved very early at the 4th epoch, which may explain why the placeholders were weaker, as at that moment the placeholders' loss was still relatively high; thus, the dummy classifiers had not yet learned stronger class boundaries. |  |
| 45 | P2 | 34 | CORE |  | Later training did reduce the placeholders' loss; however, it destabilized the validation accuracy, which dropped it as low as 53.12\% at epoch 13 and ended at 93.40\% at epoch 50, below the 95.20\% best. |  |
| 46 | P3 | 25 | CORE |  | The models' reported results were not strong enough to clearly support the claim that the classifier and the data placeholders tightened the decision boundaries usefully. |  |
| 47 | P3 | 17 | CORE |  | The interpolation between the known classes appears insufficient to represent the actual near and far unknown distributions. |  |
| 48 | P3 | 28 | CORE |  | The model reported several confident failures, such as bus/truck and fox/cat, suggesting that the near unknown lies close to a single known class rather than between two knowns. |  |
| 49 | P3 | 26 | CORE |  | On the other hand, the failures, such as the keyboard/bird and chair/airplane, show that far unknowns may also occupy directions not represented by interpolating known-class features. |  |
