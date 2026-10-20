# Space cuts to move Task 4 Q4 onto page 12

Goal: free about 0.75 page so that §4.5 (PROSER) fits on page 12 and a short
conclusion fits before the references. Every item below is a **deletion only**;
no remaining sentence is reworded. Text is quoted exactly as it appears in the
report source (`report/sections/*.tex`) and in `research qna.txt`.

---

## 1. Task 1, §1.5 Representation Space Analysis (RQ3) — first paragraph (~55 words)

**Delete the whole paragraph**

- **From:** `For all models, feature representation similarity remained quite stable across translations.`
- **To (inclusive):** `…Cue conflicts caused the sharpest drop of all, ResNet(0.44), ViT(0.48), but CLIP managed a good performance of 0.78 points.`

**Why:** every value is shown in Figure 2(b), and §1.4 already quotes the translation and patch-shuffle similarities.

---

## 2. Task 1, §1.5 (RQ3) — first sentence of the next paragraph (~16 words)

**Delete one sentence only** (keep the rest of the paragraph, which starts "The clearest divergence appears…")

- **Delete:** `On translation interventions the features barely moved, (stayed 0.93 ~ 0.98), and accuracy remains stable.`

**Why:** §1.4 (RQ2) already says "feature similarity between 0.93 and 0.98 when comparing baseline and translated images, and they therefore retain their accuracy and consistency."

---

## 3. Task 1, §1.5 (RQ3) — one-sentence paragraph before the t-SNE figure (~25 words)

**Delete the whole (one-sentence) paragraph**

- **Delete:** `CLIP lost the most accuracy under shuffling (−13.8 points), while ViT moved furthest (0.629 under shuffling) and lost the least (−6.2 points).`

**Why:** near-duplicate of the §1.4 (RQ2) sentence "The correspondence holds when comparing the two interventions, but not when comparing models, since CLIP keeps the highest similarity under shuffling (0.748) while losing the most accuracy (−13.8 points), and ViT has the lowest similarity (0.629) with the smallest loss (−6.2 points)."

---

## 4. Task 1, §1.6 Architecture and Observation Relationships (Q4) — last paragraph (~38 words)

**Delete the whole paragraph**

- **From:** `Definitively isolating these underlying mechanisms would require controlled ablations holding non-architectural factors fixed,`
- **To (inclusive):** `…or training CNN and Transformer backbones under identical dataset sizes, augmentations, and optimisation recipes.`

**Why:** future work; not needed to answer the question.

---

## 5. Task 2, §2.4 Class-Conditional versus Marginal Alignment (Q3) — second sentence of the first paragraph (~65 words)

**Delete one sentence only** (keep the paragraph's first sentence, "The results do not clearly support the claim that class conditioning helps preserve class semantic features better.")

- **From:** `With DAN, the domain separability score reduced down to 89.69 from 99.86 and showed the best target adaptation performance,`
- **To (inclusive):** `…(with giraffe +27.89, house +55.00, and dog +26.81 points), and for elephant and person, it saw a marginal decline.`

**Why:** §2.3 (Q2) already gives DAN's per-class numbers (giraffe 27.89, house 55.00, dog 26.81; elephant and person negative transfer), and Table 3 gives its separability and macro-F1.
**Note:** the next paragraph starts "CDAN also observed … marginally less than DAN." — it still reads correctly after the cut.
**Bonus:** this also removes the "89.69" that should have been 89.70.

---

## 6. Task 2, §2.5 Effect of Increasing Alignment Strength (Q4) — first paragraph (~12 words)

**Delete one sentence only**

- **Delete:** `Both the probabilistic loss and classification performance now show a total failure.`

**Why:** restates the sentence before it ("Source-validation performance then collapsed outright … eventually reached 7.83%.").

---

## 7. Task 3, §3.5 Target-Aware DAN versus Target-Free DAN-DG (Q4) — the table (~0.15 page)

**Delete Table 7** (the three-row ERM / Task 2 DAN / Task 3 DAN-DG table, `tab:task3_target_access`).

**Why:** its ERM and DAN-DG rows are already in Table 5 (Task 3 Sketch results) and its DAN row is in Table 3 (Task 2).

**Follow-up (reference only, no wording change):** your sentence
`Both methods improve substantially over the shared ERM baseline (Table 7)`
would point to Tables 3 and 5 instead: `(Tables 3 and 5)`.
It also still needs a full stop at the end.

---

## 8. Setup summaries §1.1–§4.1 (written by Claude, ~0.2 page)

Not your text. Claude trims each to one sentence plus the appendix pointer,
keeping the citations. You still need to rewrite these in your own words.

---

## Totals

| # | Task / section | Type | Saves |
|---|---|---|---|
| 1 | Task 1 §1.5 | paragraph | ~55 words |
| 2 | Task 1 §1.5 | sentence | ~16 words |
| 3 | Task 1 §1.5 | paragraph | ~25 words |
| 4 | Task 1 §1.6 | paragraph | ~38 words |
| 5 | Task 2 §2.4 | sentence | ~65 words |
| 6 | Task 2 §2.5 | sentence | ~12 words |
| 7 | Task 3 §3.5 | table | ~0.15 page |
| 8 | §1.1–§4.1 | setup text | ~0.2 page |
| | **Total** | | **~0.75 page** |

## Still to add

**Conclusion (your writing, 3–4 lines):** what a robust representation should
**preserve**, what it should become **invariant to**, and which inputs it should
**reject**. It will go in a final "Conclusion" section before the references.
