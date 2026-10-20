# Cut sheet: Task 2

Mark a sentence for deletion by putting `x` in the Cut column. Tier A = recommended cut, tier B = optional. Untagged sentences are CORE. Notes marked LINK mean a neighbouring sentence refers back to this one.


## Summary

| Subsection | Words | Tier A cut | Tier B cut | After A | After A+B |
|---|---|---|---|---|---|
| The Source-Only Domain Gap and Its Failure Modes | 227 | 42 | 72 | 185 | 113 |
| Domain Separability and Target Recognition | 290 | 69 | 18 | 221 | 203 |
| Class-Conditional versus Marginal Alignment | 345 | 148 | 24 | 197 | 173 |
| Effect of Increasing Alignment Strength | 345 | 12 | 55 | 333 | 278 |
| **Total** | **1207** | **271** | **169** | **936** | **767** |

## The Source-Only Domain Gap and Its Failure Modes

| # | Para | Words | Tag | Cut | Sentence | Note |
|---|---|---|---|---|---|---|
| 1 | P1 | 18 | CORE |  | The source-only ERM showed a substantial domain gap of 38.61 points (macro-F1) between the source and target domains. |  |
| 2 | P1 | 21 | REPEATS TABLE B |  | Although the model achieved a good 94.51\% macro-F1 on the source domain, on the target (Sketch domain), it only attained 55.90\%. | both values are in Table 3; 1 already gives the 38.61 gap |
| 3 | P1 | 36 | DETAIL B |  | The model showed a healthy decrease in training loss from 0.477 to 0.049, respectively observed an increase in training accuracy up to 98.79\%, and a reflective increase in source-validation macro-F1, peaking at 94.51\% at epoch 11. | training health; shown by the training-curve figure |
| 4 | P1 | 20 | REPEATS TABLE A |  | The performance remained consistently high across the source domains (Photo at 95.53\%, Art Painting at 92.64\%, and Cartoon at 95.37\%). | per-domain values are in Table 3 |
| 5 | P1 | 36 | CORE |  | This demonstrates that ERM made the model learn class-discriminative source-domain features; however, these features are not directly transferable to the target domain and contain strong domain-specific information, which is clearly visible in the 99.86\% domain-separability score. |  |
| 6 | P2 | 22 | FILLER A |  | The classes that resulted in the most important target failures were found by accounting for the overall error share of each class. | method lead-in; LINK: 7 starts 'As per these numbers' - fix if you cut 6 |
| 7 | P2 | 19 | CORE |  | As per these numbers, Giraffe, Dog, and Horse accounted for 86.7\% of the overall errors on the target domain. |  |
| 8 | P2 | 28 | CORE |  | Additionally, the model also displayed a strong bias towards the person class, and the model confused the majority of the classes with person, producing around 1,116 such errors. |  |
| 9 | P2 | 15 | DETAIL B |  | In contrast, the elephant and guitar classes showed good results, obtaining 88.78\% and 83.06\% accuracy. | the ERM-failures figure shows elephant/guitar |
| 10 | P2 | 12 | CORE |  | This shows that the domain shift did not affect each class uniformly. |  |

## Domain Separability and Target Recognition

| # | Para | Words | Tag | Cut | Sentence | Note |
|---|---|---|---|---|---|---|
| 11 | P1 | 14 | FILLER A |  | After the linear probe via logistic regression, we got the scores reported in Table~\ref{tab:task2_methods}. | lead-in; LINK: it holds the only Table 3 reference in this subsection - move the ref if you cut it |
| 12 | P1 | 25 | CORE |  | Across all methods, we see a clear enough pattern: the higher the domain separability between source and target domains, the lower the target accuracy scores. |  |
| 13 | P1 | 16 | REPEATS TABLE A |  | ERM set the baseline for separability at around 99.86, with the target accuracy at 55.71 points. | ERM values are in Table 3 |
| 14 | P1 | 34 | CORE |  | The treatment methods like DAN and CDAN were able to fractionally reduce this separability gap, and as the domain separability scores reduced with these methods, so did the accuracy on the target domain improve. |  |
| 15 | P1 | 27 | CORE |  | DANN, though it performed miserably because of an unstable optimization flow, shows that it has the highest domain separability score and thus has the lowest target-domain accuracy. |  |
| 16 | P2 | 14 | FILLER A |  | The effect of domain separability is clearly visible in the class-level results as well. | lead-in |
| 17 | P2 | 16 | DUPLICATE A |  | The methods that performed better compared to the others, like DAN or CDAN, showed successful adaptation. | restates 14; LINK: 18 starts 'this improvement' - fix if you cut 17 |
| 18 | P2 | 11 | CORE |  | However, this improvement was not uniform across all of the classes. |  |
| 19 | P2 | 25 | CORE |  | In DAN, the adaptation improved 5 out of 7 classes, with giraffe improving by 27.89 points, house by 55.00 points, and dog by 26.81 points. |  |
| 20 | P2 | 17 | CORE |  | However, for elephants and persons, it exhibited negative transfer, declining in performance by 1.62 and 1.25 points. |  |
| 21 | P2 | 22 | CORE |  | Similarly, CDAN also adapted successfully overall and improved 6 out of 7 classes, except elephant, where its performance decreased by 31.62 points. |  |
| 22 | P2 | 5 | CORE |  | DANN displayed negative transfer overall. |  |
| 23 | P2 | 18 | DETAIL B |  | Despite showing adaptation on dog, giraffe, and house, it reported terrible performance on guitar, person, horse, and elephant. | DANN per-class detail; the per-class figure shows it |
| 24 | P2 | 20 | CORE |  | Its target macro-F1 plummeted by 19.19 points relative to ERM, most likely because of the instability it faced during optimization. |  |
| 25 | P3 | 17 | CORE |  | Overall, reduced domain separability can be associated with better performance on target recognition for DAN and CDAN. |  |
| 26 | P3 | 9 | DUPLICATE A |  | However, for good adaptation, per-class performance matters as well. | restates 18 |

## Class-Conditional versus Marginal Alignment

| # | Para | Words | Tag | Cut | Sentence | Note |
|---|---|---|---|---|---|---|
| 27 | P1 | 17 | CORE |  | The results do not clearly support the claim that class conditioning helps preserve class semantic features better. |  |
| 28 | P1 | 83 | DEFINITION A |  | The treatment methods DAN and DANN use two different approaches for domain alignment, where DAN uses MMD to compare distance between the source and target distributions and uses it as a loss penalty, i.e., the larger the distance between the source and target domain distributions, the greater the loss penalty, forcing the model to learn features that bring the two distributions closer in vector space so that the feature extractor learns about features that have resulted in a reduced gap between the distributions. | 83-word explanation of how MMD works; standard for the reader |
| 29 | P1 | 65 | DUPLICATE A |  | With DAN, the domain separability score reduced down to 89.69 from 99.86 and showed the best target adaptation performance, reaching 66.84\% macro-F1; however, the individual class-wise macro-F1 for DAN is not uniform across the classes, as it performed really well on 5 out of 7 classes (with giraffe $+27.89$, house $+55.00$, and dog $+26.81$ points), and for elephant and person, it saw a marginal decline. | repeats 19-20 and Table 3 (DAN per-class and F1) |
| 30 | P1 | 17 | CORE |  | CDAN also observed an improved domain adaptation and reached a 63.12\% target macro-F1, marginally less than DAN. |  |
| 31 | P1 | 60 | CORE |  | Overall, it performed well on 6 out of 7 classes and also preserved person and guitar recognition, improving them by 1.25 and 2.47 points, but reduced the adaptation on the elephant class by a massive 31.62 points, and also provided just a 6.51-point gain on the giraffe, whereas on the same giraffe class DAN improved the adaptation by 27.89 points. |  |
| 32 | P1 | 28 | CORE |  | Thus, the class conditionality might have partially preserved the class-wise consistency, with 6 classes improving; however, it did not manage to improve the overall aggregate for target adaptation. |  |
| 33 | P2 | 14 | CORE |  | CDAN does outperform the marginal adversarial alignment DANN, where DANN only reported 36.71\% macro-F1. |  |
| 34 | P2 | 24 | DUPLICATE B |  | DANN also diverged with extremely larger classification and alignment losses and the optimizational instability, which is a key confounding factor in the underperforming DANN. | DANN instability is already in 24 and in the alignment-strength subsection |
| 35 | P2 | 37 | CORE |  | Overall results may suggest that conditioning can better preserve semantic structure, and it improved aggregate target adaptation relative to ERM but did not outperform DAN, and stable DAN depicted a good marginal alignment that may outperform CDAN. |  |

## Effect of Increasing Alignment Strength

| # | Para | Words | Tag | Cut | Sentence | Note |
|---|---|---|---|---|---|---|
| 36 | P1 | 18 | FILLER B |  | Overall baseline model performance with DANN, having alignment strength = 1.0, started well in the very first epoch. | lead-in to 37 |
| 37 | P1 | 23 | DETAIL B |  | At the end of epoch 1, the classification loss was 0.61 (well below $\ln(7)=1.946$), training accuracy reached 80.95\%, and validation F1 reached 83.21\%. | epoch-1 values from the training curves |
| 38 | P1 | 14 | DETAIL B |  | Meanwhile, the alignment loss reached 2.78 (exceeding $\ln(2)=0.693$), while domain accuracy settled at 58.18\%. | epoch-1 values from the training curves |
| 39 | P1 | 59 | CORE |  | However, from epoch 2 onwards, the overall optimization became unstable, as the model's classification loss shot up to 15.21 in the second epoch; the training accuracy dropped to 69.80\%, while the validation F1 moved up to 83.43\%, which tells us the model is making some very confident errors as its training accuracy decreases while its validation score remains high. |  |
| 40 | P1 | 24 | CORE |  | Source-validation performance then collapsed outright from the 3rd epoch, when the validation macro-F1 plummeted from 83.43\% down to 35.96\% and then eventually reached 7.83\%. |  |
| 41 | P1 | 12 | DUPLICATE A |  | Both the probabilistic loss and classification performance now show a total failure. | restates 39-40 |
| 42 | P1 | 32 | CORE |  | The domain accuracy stayed between 42.29\% and 58.18\%; however, the alignment loss increased from 2.78 at the end of the first epoch to 97.49 in the second, reaching 2,013,773.19 in 7 epochs. |  |
| 43 | P2 | 39 | CORE |  | As we halve DANN's alignment strength, things do start to get a little better for the source-domain accuracy, as the source macro-F1 gained 7.46 points; however, the reduced alignment strength impacted the target macro-F1, decreasing it by 4.11 points. |  |
| 44 | P2 | 32 | CORE |  | The pattern repeats when we further reduce the alignment strength, which further increases the model accuracy on source classification; however, it progressively damages target recognition, decreasing target macro-F1 by another 12.19 points. |  |
| 45 | P2 | 23 | CORE |  | Interestingly, while we have these variations in target F1 score as we decrease the alignment strength, source and target separability remains around 100\%. |  |
| 46 | P2 | 17 | CORE |  | Therefore, in this scenario, the stronger or weaker adversarial objective failed to produce any measurable domain-invariant features. |  |
| 47 | P3 | 15 | CORE |  | In fact, all three DANN configurations remained well below the ERM target macro-F1 of 55.90\%. |  |
| 48 | P3 | 37 | CORE |  | So, the trade-off between low and high alignment strength had a substantial positive effect on target macro-F1 within the DANN variants; however, stronger alignment destabilized the model's training, and a smaller value further deteriorated the target macro-F1. |  |
