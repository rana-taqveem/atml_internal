# Cut sheet: Task 3

Mark a sentence for deletion by putting `x` in the Cut column. Tier A = recommended cut, tier B = optional. Untagged sentences are CORE. Notes marked LINK mean a neighbouring sentence refers back to this one.


## Summary

| Subsection | Words | Tier A cut | Tier B cut | After A | After A+B |
|---|---|---|---|---|---|
| Source Validation as a Predictor of Sketch Performance | 299 | 47 | 56 | 252 | 196 |
| Source-Domain Invariance and Class-Discriminative Information | 207 | 13 | 0 | 194 | 194 |
| Local Stability and Sketch Performance | 202 | 0 | 21 | 202 | 181 |
| Target-Aware DAN versus Target-Free DAN-DG | 235 | 16 | 0 | 219 | 219 |
| **Total** | **943** | **76** | **77** | **867** | **790** |

## Source Validation as a Predictor of Sketch Performance

| # | Para | Words | Tag | Cut | Sentence | Note |
|---|---|---|---|---|---|---|
| 1 | P1 | 24 | CORE |  | Good performance on the source domain does not guarantee good performance on the target domain, and this is visible via the results as well. |  |
| 2 | P1 | 33 | REPEATS TABLE B |  | A high mean source accuracy / macro-F1 is observed in all 3 methods, where the ERM baseline mean source accuracy / macro-F1 is 94.49 / 94.51; the worst-performing source domain scored 92.64 points. | values are in Table 4; keeps the ERM mean/worst example |
| 3 | P1 | 24 | REPEATS TABLE A |  | Similarly, the mean source accuracy / macro-F1 for DAN-DG (with $\lambda=1.0$) and SAM (with $\rho=0.05$) is 91.82 / 91.56 and 95.16 / 95.12, respectively. | values are in Table 4; 5-6 re-quote the ones the argument needs |
| 4 | P1 | 9 | REPEATS TABLE B |  | Their worst-performing source macro-F1 is 89.35 and 92.34, respectively. | worst-source values are in Table 4 |
| 5 | P1 | 45 | CORE |  | While establishing a possible link between the validation numbers of source domains and the test numbers of the target domains, ERM, as anticipated, performed worst in the target domain, whereas DAN-DG has the lowest mean source accuracy of 91.82 and best target accuracy/macro-F1 of 68.44/72.37. |  |
| 6 | P1 | 21 | CORE |  | SAM has performed the highest mean source accuracy of 95.16, but still its target accuracy is 3.89 points lower than DAN-DG. |  |
| 7 | P1 | 14 | DUPLICATE B |  | This shows there is no linear relation between the source and target domain performances. | 8 states the conclusion that answers the question |
| 8 | P1 | 15 | CORE |  | Therefore, it is difficult to establish that source validation performance reliably predicts performance on Sketch. |  |
| 9 | P2 | 17 | FILLER A |  | The details hidden by the aggregated source metrics come to light when looking at the class-level failures. | lead-in |
| 10 | P2 | 47 | CORE |  | ERM performs very poorly on Sketch for giraffe, dog, horse, and house, with accuracies of 23.51\%, 36.79\%, 45.59\%, and 45.00\%, whereas in the case of DAN-DG, 5 out of 7 classes gained points (house by 48.75 points, horse by 27.45, giraffe by 21.38, and guitar by 12.50). |  |
| 11 | P2 | 10 | CORE |  | However, for the person, the performance declined by 33.75 points. |  |
| 12 | P2 | 22 | CORE |  | Therefore, high aggregate source performance hides substantial target-specific class failures, particularly the degradation of person and the continued difficulty of dog recognition. |  |
| 13 | P2 | 6 | DETAIL A |  | These results are recorded in \texttt{per\_class\_sketch.csv}. | file name; not needed in a report |
| 14 | P3 | 12 | CORE |  | The art domain is recorded as the worst across all three methods. |  |

## Source-Domain Invariance and Class-Discriminative Information

| # | Para | Words | Tag | Cut | Sentence | Note |
|---|---|---|---|---|---|---|
| 15 | P1 | 25 | CORE |  | DAN-DG made the observed source domains slightly less distinguishable, as shown by the source-domain separability score, which decreases to 80.07\% compared with 86.38\% under ERM. |  |
| 16 | P1 | 18 | CORE |  | Its performance on Sketch also got better, and it reported an increase in macro-F1 from 55.90 to 72.37. |  |
| 17 | P1 | 18 | CORE |  | These numbers appear to support the claim; however, the results change drastically when $\lambda_{\mathrm{DG}}$ is increased to 10. |  |
| 18 | P1 | 23 | CORE |  | The separability score fell to 48.84\%, but the macro-F1 on both the Sketch and the source just collapsed to 7.84\% and 16.06\%, respectively. |  |
| 19 | P1 | 28 | CORE |  | Class-wise, 6 out of 7 classes lost substantial points: guitar fell by 82.40 points, person by 71.25, elephant by 58.38, and both giraffe and horse reached 0\% accuracy. |  |
| 20 | P1 | 23 | CORE |  | Moreover, in the case of SAM and ERM, the separability is identical at 86.38\%, yet SAM's Sketch macro-F1 is 67.64 against ERM's 55.90. |  |
| 21 | P1 | 13 | DUPLICATE A |  | Two models with exactly the same domain separability, yet a substantial performance gap. | fragment restating 20 |
| 22 | P1 | 30 | CORE |  | This suggests that excessive source alignment does remove domain-discriminative features, making domains more difficult to distinguish; however, at the same time, it destroys the core semantic structure required for classification. |  |
| 23 | P1 | 29 | CORE |  | Therefore, moderate invariance among the domains partially improves Sketch recognition, while a stronger push results in loss of critical class-discriminative information, resulting in poor performance on the target domain. |  |

## Local Stability and Sketch Performance

| # | Para | Words | Tag | Cut | Sentence | Note |
|---|---|---|---|---|---|---|
| 24 | P1 | 35 | CORE |  | A lower common sharpness proxy under given perturbation partially supports the claim about local stability, but it does not establish that the overall loss surface is flatter and will generalize better on any domain shifts. |  |
| 25 | P1 | 21 | DUPLICATE B |  | The model trained with SAM perturbation did appear to be a little less sensitive to any minor change in its weights. | 26 gives the same point with numbers |
| 26 | P1 | 21 | CORE |  | SAM's loss increased by just 0.141, as compared to ERM and DAN-DG, which reported a change of 0.600 and 0.631, respectively. |  |
| 27 | P1 | 13 | CORE |  | Therefore, the local stability ranking is \[ \text{SAM} < \text{ERM} < \text{DAN-DG}. \] |  |
| 28 | P1 | 35 | CORE |  | However, it does not establish that the local stability ranking enables better performance on the target domain, because on the target domain the performance of each model is \[ \text{DAN-DG} > \text{SAM} > \text{ERM}. \] |  |
| 29 | P1 | 31 | CORE |  | DAN-DG reported the highest sensitivity to a minor perturbation; however, its Sketch macro-F1 is the highest (72.37\%), whereas SAM is much flatter as per the proxy ranking, but it reaches 67.64\%. |  |
| 30 | P1 | 46 | CORE |  | Nevertheless, SAM does improve over ERM by 8.83 accuracy points and 11.74 macro-F1 points, suggesting that local parameter stability may contribute to some extent to better generalization, but it is difficult to confirm that in every setting SAM will result in better performance on target domains. |  |

## Target-Aware DAN versus Target-Free DAN-DG

| # | Para | Words | Tag | Cut | Sentence | Note |
|---|---|---|---|---|---|---|
| 31 | P1 | 9 | CORE |  | Both methods improve substantially over the shared ERM baseline: |  |
| 32 | P2 | 16 | DUPLICATE A |  | Both DAN and DAN-DG displayed comparatively better performance on the target domain as compared to ERM. | restates 31 |
| 33 | P2 | 22 | CORE |  | DAN used unlabeled Sketch data to align the source features and improved over ERM by 12.09 accuracy points and 10.93 macro-F1 points. |  |
| 34 | P2 | 21 | CORE |  | DAN-DG, which never had access to the target data (sketches) during training, improved by 12.73 accuracy points and 16.46 macro-F1 points. |  |
| 35 | P2 | 12 | CORE |  | DAN-DG exceeds target-aware DAN by 0.64 accuracy points and 5.53 macro-F1 points. |  |
| 36 | P3 | 15 | CORE |  | This suggests that access to unlabeled Sketch data did not provide a substantial empirical advantage. |  |
| 37 | P3 | 17 | CORE |  | Instead, it highlights that DAN-DG was able to learn the cross-domain invariances, which resulted in source-target alignment. |  |
| 38 | P3 | 23 | CORE |  | However, this does not establish that unlabeled Sketch data has no value, and several limitations prevent this from being a complete causal attribution. |  |
| 39 | P4 | 18 | CORE |  | The whole experiment is limited to a single seed, 6304, a different seed might have produced different results. |  |
| 40 | P5 | 50 | CORE |  | DAN and DAN-DG work on different distributions; the DAN objective aligns source and target domain features given by \(\mathcal L_{\mathrm{align}}^{\mathrm{DAN}} = \operatorname{MMD}^2 \left( F(X_P\cup X_A\cup X_C), F(X_S) \right).\) whereas DAN-DG tries to learn invariances among the source domains: \[ \mathcal L_{\mathrm{align}}^{\mathrm{DAN-DG}} = \frac{ \operatorname{MMD}^2(P,A) + \operatorname{MMD}^2(P,C) + \operatorname{MMD}^2(A,C) }{3}. \] |  |
| 41 | P5 | 32 | CORE |  | In DAN, the pooled comparison between the source and target can hide the differences between the individual source domains; however, the pairwise terms in DAN-DG ensure that these differences are accounted for. |  |
