# PA2 implementation and evidence audit

Audit date: 10 October 2026 (Asia/Karachi). Assignment due date printed in the manual: 11 October 2026.
Source: `../ATML PA2.pdf`, all 15 pages, especially required evidence and research questions on pages 5, 7, 9, 11, 13, and submission rules on page 14.

This is a technical work checklist, not report text or answers to submit. The manual prohibits generative AI from writing the report's language, interpretation, and analysis. The student must select, inspect and interpret the evidence personally.

## Decision

The assignment is **not complete and cannot yet be certified as 100% compliant**. Tasks 1 and 2 have substantial saved experimental evidence and the three deliberate objective defects are corrected. Task 3 has the cached group-size study, but no completed continuation or normalization forks. Tasks 4 and 5 have implementations but no saved full evaluation results. Task 6 currently contains only partial synthesis.

Two methodological issues need explicit resolution: DPO truncation changes prompt context, and the PPO KL forks do not have equal realized generated-token totals. Passing file/metric checks does not resolve either issue or establish response quality.

## What was checked

- Read the assignment and the Python implementation in `common/`, `scripts/`, and task folders 1-6; inspected all five runner notebooks, configs, cloud guides and saved per-task results.
- `python -m scripts.validate_assets`: passes.
- `python -m scripts.audit_assignment`: 202 saved-evidence/hash checks pass. JSON: `results/audit/evidence_audit.json`. Missing future experiments are recorded separately, not counted as successful checks.
- 87 non-self entries in the current `manifests/sha256.json` match local files. Its self-entry is excluded because a final manifest cannot hash itself. `source_v2_sha256.json` is historical and differs from this release; it is not the current acceptance manifest.
- `python -m unittest discover -s tests -v`: eleven tests pass. Covers DPO sign/reference identity, PPO clipping signs and masked gradients, GAE termination/padding, within-prompt GRPO advantages, additive loss/gradient equivalence, last-final-answer parsing, all 100 fixed verifier expectations, stale/partial cache rejection, safety aggregation/completeness, and judge parse-fallback instrumentation. Model inference is mocked in judge tests.
- Python source compiles. No remaining `NotImplementedError` training scaffolds found.
- Task-specific `.venv`: Transformers 4.57.1, Tokenizers 0.22.1, PEFT 0.17.1, TRL 0.27.2. Local PyTorch is CPU-only, 2.14.0+cpu. GPU training and inference were **not** executed during this audit.
- Task 3 and Task 5 dependency preflights pass without CUDA. Task 4 preflight correctly fails because the final standard GRPO adapter/summary are absent. Adding `--require-cuda` is necessary in the execution session.

## Checkpoints and data

| Asset | Local state | Purpose |
|---|---|---|
| `checkpoints/grpo_midpoint_policy` | Config and weights present; current release hash matches | Start all Task 3 conditions here |
| `checkpoints/rlvr_policy` | Present; hash matches | Frozen Task 5 evaluation; no retraining |
| `checkpoints/rlaif_policy` | Present; hash matches | Frozen Task 5 evaluation; no retraining |
| `checkpoints/ppo_midpoint_policy` and `ppo_midpoint_value` | Present; hashes match | Task 2 start/reproduction |
| `outputs/task1_dpo/standard` | Trained adapter present | Fixed Task 4 DPO policy, subject to the truncation resolution below |
| `outputs/task2_ppo/standard` | Trained adapter present | Fixed Task 4 PPO policy |
| `outputs/task3_grpo/standard` | Missing | Must complete 20-update standard continuation before full Task 4 |

The course checkpoints are LoRA adapters, not standalone full models. Qwen base weights, reward model and judge weights must also be accessible/downloadable in the GPU session; this audit did not load those public model weights.

Data counts: DPO standard train/eval 1500/300; balanced train 1500 (500 per stratum); stratified eval 246 (82 per stratum); RL prompt pools 1200/200; GRPO cache 24 prompts x 8 completions = 192; XSTest 450 prompts; GSM8K train/eval 1500/300; SVAMP transfer 100; controlled diagnostics 20 problems x 5 variants; word-limit prompts 10. Fixed input files were not altered.

`results/audit/tokenization_audit.json` records prompt/response tokenization checks using the local supplied tokenizer. No Task 4 or Task 5 generation prompt exceeds its current input cap (XSTest max 48/256; GSM8K 239/512; SVAMP 114/512).

## Task 1: executed, with a material truncation limitation

| Condition | Training pairs | Updates | Held-out pairs | Generation prompts | Word-limit responses |
|---|---:|---:|---:|---:|---:|
| SFT reference | 0 | 0 | 300 + 246 | 128 | 10 greedy + 50 sampled |
| Standard DPO | 1500, one epoch | 94 | 300 + 246 | 128 | 60 |
| beta 0.03 | 600 | 38 | 300 + 246 | 128 | 60 |
| beta 0.10 | 600 | 38 | 300 + 246 | 128 | 60 |
| beta 0.30 | 600 | 38 | 300 + 246 | 128 | 60 |
| Length-balanced DPO | 1500, one epoch | 94 | 300 + 246 | 128 | 60 |

No nonfinite updates were skipped. Beta forks have identical training ID order. All evaluation conditions use the same saved generation IDs. Per-pair margins, losses, strict-positive accuracy labels and aggregate metrics were independently recomputed, preserving float32 subtraction rounding.

Evidence: `results/task1_dpo/summary.csv`, per-condition `train_log.jsonl`, `train_summary.json`, `eval_metrics.json`, both `eval_pairs_*.jsonl`, `generations.jsonl`, `word_limit.jsonl`; `length_analysis.json`, `length_mechanism.json`, three figure sets and qualitative candidate files. Standard continuation compute: 1534.3 seconds, 8.388 GiB peak allocated VRAM.

The corrected objective subtracts the reference margin. Policy equal to reference produces loss log(2) and strict `m > 0` preference accuracy zero. That zero is a reference-relative tie convention, not an ordinary 0% content-quality accuracy. The three beta losses use different beta scales, so their numerical values are not a common-scale quality score. The per-token-margin analysis is supplementary and must not replace the required summed-response margin.

### DPO context truncation: unresolved

`common/data.py::encode_prompt_response` first shortens the prompt based on each response's length, then takes the last 768 tokens. This is also present in the locally available upstream starter revision. For sufficiently long responses, all prompt tokens disappear; chosen and rejected responses can otherwise retain different amounts of the same prompt.

| File | Encoded responses | Responses retaining no prompt | Pairs with different retained prompt context |
|---|---:|---:|---:|
| Standard training | 3000 | 118 | 264 / 1500 |
| Standard evaluation | 600 | 28 | 53 / 300 |
| Length-balanced training | 3000 | 0 | 97 / 1500 |
| Length-stratified evaluation | 492 | 0 | 16 / 246 |

Saved metrics faithfully reflect this starter convention, but some terms no longer represent two responses conditioned on the same retained prompt. This also differs across the standard and balanced datasets. The audit did not silently change this helper or overwrite completed Task 1 runs.

**Decision (documented methodological judgment, not proof of compliance):** retain the released encoder and the
completed Task 1 runs; report the limitation with the evaluation-side sensitivity analysis in
`results/task1_dpo/truncation_sensitivity.json` (`python -m task1_dpo.summarize`). It classifies every pair as
`same_context`, `context_differs` (both responses keep some prompt, but different prefixes) or `context_free`, for the
training files and both evaluation splits, and reports held-out accuracy/margin per class for every trained condition.
Excluding the 21 context-free standard-evaluation pairs changes held-out accuracy by at most 1.5 percentage points
(standard 0.637 -> 0.645). Scope of that evidence: it shows how much the *reported evaluation numbers* depend on
truncated pairs; it does not show what training without truncation would have produced, and a change smaller than one
standard error does not rule out a systematic bias. The length study is not free of the issue either: the length
files have no context-free pairs, but 97 balanced-training and 16 stratified-evaluation pairs score chosen and rejected
under different prompt prefixes, and the standard model compared there was trained on the affected standard file.

The alternative remains open if the course requires it: resolve whether the course requires this released convention or a corrected shared-prompt truncation policy. A correction needs one documented, common encoding rule across all DPO conditions, objective/data regression checks, retraining all five DPO adapters and re-evaluating every DPO condition. Recomputing only evaluation metrics cannot repair training context loss. If standard DPO is replaced, Task 4 must use the replacement standard adapter and its safety responses must be regenerated. Preserve old results under a separately named legacy run.

### Other Task 1 report work

Generation truncation is high at the required 256-token cap: standard 49.22%, balanced 51.56%, SFT 47.66%. Length comparisons are cap-censored and do not establish unrestricted verbosity. Qualitative candidates are a shortlist, not completed human quality assessment. Required minimum examples: reward/preference versus quality disagreement and a length/instruction-compliance example, inspected and interpreted by the student.

## Task 2: required runs saved; token-budget qualification remains

Standard PPO: 20 updates; five distinct short forks: 8 updates each, with the epsilon=0.20/beta=0.10 condition shared between clipping and KL studies. Eight evaluation conditions each have 64 saved responses under the same prompt IDs, seed, cap and decoding. All start summaries name the supplied midpoint; policy and critic artifacts are present. No nonfinite steps recorded.

Standard compute: 231.3 seconds, 6.252 GiB peak allocated VRAM. Logs include reward and raw reward, token/sequence sampled KL, policy/value losses, exact vocabulary entropy, gradient norms, clip/affected fractions, lengths, critic explained variance and step-change diagnostics. The repaired PPO objective uses the minimum surrogate. GAE terminal behavior and masking pass counterexample checks.

Evidence: `results/task2_ppo/summary.csv`, `summary.json`, `clipping_cached.json`, condition `train_log.jsonl`, `rollouts.jsonl`, `train_summary.json`, `eval_metrics.json`, `heldout_generations.jsonl`, figures and qualitative candidates.

### Identical clipping forks are supported by the logs

All three epsilon forks have the same policy-weight SHA256 and 1869 generated tokens. All active clipped-branch fractions are zero at every update; the 0.05 fork has some ratios outside the interval but on the non-binding sign branch. This supports identical optimization under these settings. Keep this result; do not retune to force separation using held-out outcomes. The cached study still distinguishes the immediate geometric effect: initial clip/affected fractions are approximately 0.15430/0.07613 (0.05), 0.04130/0.01974 (0.20), and 0.02110/0.01010 (0.50); use the saved JSON for exact values.

### Actual generated-token counts differ across KL forks

| KL coefficient | Updates | Actual generated tokens |
|---|---:|---:|
| 0.00 | 8 | 2036 |
| 0.10 | 8 | 1869 |
| 0.20 | 8 | 1969 |

Rollout texts (`rollouts.jsonl`) are identical across the three KL forks at updates 1, 3, 4 and 5; the beta=0.20 response already differs at update 2 (all three hit the same 512-token cap there, so counts match), and all forks diverge from update 6. Prompts, update count, seed and maximum generation allowance match, but actual generated tokens differ because EOS is policy dependent. The manual requests equal generated-token and update budgets. Do not describe these as exactly equal realized-token runs. If the course intends equal allocation/caps, state both allocation and realized totals; if literal equality is required, a documented controlled protocol and reruns are needed. Token totals cannot be retroactively equalized by changing the saved summaries. The same issue must be checked for Task 3 normalization forks before final acceptance.

The sampled-response KL estimator can be slightly negative; do not clamp it to zero or call it an exact nonnegative distribution KL. Reward with the missing-EOS penalty and raw reward are saved separately. The released critic is explicitly weak; report its observed diagnostics. Required human qualitative work remains: at least one reward/quality agreement and one disagreement.

## Task 3: implementation ready for GPU smoke test, execution incomplete

Completed: cached group-size analysis. For K=2,4,8, the same 192 completions yield 96/48/24 groups. Informative rates are 0.93750/0.95833/0.95833. The output contains within-group reward std, centered-signal and normalized-advantage variances, regrouping instability and three difficulty bins. Bins are tertiles of each prompt's mean cached learned reward; these are reward-defined difficulty proxies, not independent correctness labels. Binary-threshold results are supplementary, not a replacement for the learned-reward study.

Missing: 20-update standard continuation and standard/midpoint/SFT held-out evaluations; canonical and Dr.-GRPO 8-update forks and their evaluations; `normalization_study.json`, full summary and training figures. Group advantages now use within-prompt mean/std; non-terminating max-length completions are masked from loss. The Dr.-GRPO comparison intentionally changes only the sequence denominator and keeps std-scaled advantages, as in the supplied helper.

During this audit, explicit surrogate token weights were added to training logs and length-conditioned statistics. They isolate the 1/T versus 1/max-length denominator. Full parameter-gradient norms divided by |advantage| still contain KL, token content and model-Jacobian effects; they are not a pure normalization measurement. Inspect gradient norms and truncation/loss-masked fractions in the two-update GPU smoke test before the full run. CPU tests do not certify GPU memory, model-loading or numerical behavior.

## Task 4: waits for standard Task 3

Required workload: 450 prompts x four fixed standard policies = 1800 greedy responses, then fixed categorical judge scoring. Use final standard DPO, standard PPO and standard GRPO only. Supplied midpoints and ablation forks are not substitutes. Generate common prompts at the same 256-token cap. No Task 4 labels may be used to select/retune earlier policies.

The evaluation implements safe answer/over-refusal conditioned on SAFE; unsafe compliance/justified refusal conditioned on UNSAFE; ambiguity overall and by class; category-label distributions, lengths, policy disagreements, and audit confusion/accuracy/kappa. New completeness checks reject a missing policy, duplicate/reordered/missing IDs, invalid labels and mismatched benchmark classes.

Manual work: default `make_audit_sheet --mode all` uses 60 unique IDs (30 SAFE/30 UNSAFE) across four policies = 240 blinded response rows. This supports comparison on the same prompts. The optional `rotate` mode produces only 60 response rows and is less suitable for within-prompt policy disagreement; use `all` for the main audit. Labels must be assigned by the student before viewing AI labels. Partial annotation is now explicitly marked incomplete. Judge confidence is used for auditing only. Required qualitative coverage: harmful compliance, justified refusal, over-refusal, with policy versus judge disagreements identified by the student.

## Task 5: independent frozen-policy evaluation, ready for GPU smoke test

Use SFT, supplied RLVR and supplied RLAIF; do not train new Task 5 policies. Required totals: GSM8K 300 x 3 = 900 generations; SVAMP 100 x 3 = 300 generations; 100 controlled responses from the fixed 20 problems. Greedy 512-token cap. Judge RLVR and RLAIF against SFT on all 400 problems (800 primary comparisons), plus diagnostic controlled pairs, reversed-order checks and the five-way round robin. The code caches judge decisions.

Implemented outputs include exact accuracy, format compliance, mean/std length, truncation and failure types, pairwise win/tie/loss, verifier-judge agreement/confusion, transfer drops, four clean-anchor controlled pair rates, S_reason and S_outcome, pointwise verifier scores and round-robin win rates for all five variants. Clean-correct is the shared anchor, not an additional perturbation pair. The exact verifier uses the last designated `#### <number>`; the gold-number distractor is not sufficient for reward.

Audit fixes: incomplete/stale smoke-generation reuse now fails; policy/data/cap fingerprints are saved; fixed judge output parsing is unchanged, with raw outputs and parse-fallback ties exposed in `judge_cache.details.json`; parse-ambiguous and unknown-cache counts are recorded separately. Diagnostic inference timing now covers round-robin calls too. SFT pairwise win rate against itself is explicitly 0.5 by convention, with no judge inference. Timing reused generations remains unavailable rather than an estimate of original cost.

Student work: inspect reasoning-only, filler and wrong-final/gold-distractor examples; identify actual judge disagreements; assess failure types and evidence for transfer. Outcome rewards and pairwise win rates are different signals and must not be treated as calibrated utilities on one scale.

## Research-question evidence map (student writes the answers)

| Question | Evidence to inspect | Pending |
|---|---|---|
| T1 Q1: beta versus fitting/drift/reward, monotonicity | Task 1 short-beta rows; pair metrics; generation metrics | Address truncation; own interpretation |
| T1 Q2: dataset length correlations versus learned behavior | `length_analysis.json`, `length_mechanism.json`, strata, word-limit files | Address context/cap limitations; own interpretation |
| T1 Q3: stronger signal but worse response | Full saved generations and word-limit candidates | Student quality checks and examples |
| T2 Q1: epsilon and constrained updates/stability | Cached clip/affected fractions; fork logs and summary stability statistics | Explain observed non-binding clipping from evidence |
| T2 Q2: first changes as KL weakens | Update-aligned KL-fork reward/KL/entropy/length logs and rollouts | Token-budget resolution; student trajectory assessment |
| T2 Q3: reward reliability away from reference | Saved reward gains/drops, responses, drift diagnostics | Student quality assessment; avoid unsupported overoptimization claims |
| T3 Q1: group size and difficulty | `group_size_study.json` overall and three bins | Student interpretation |
| T3 Q2: normalization, length, gradient allocation, quality | Fork logs, token weights, `normalization_study.json`, held-out generations | GPU execution and student comparison |
| T3 Q3: instability/sample inefficiency without critic | Standard logs: group std, zero-std, truncation, gradients, reward | GPU execution and student analysis |
| T4 Q1: reward metrics versus safety | Standard policy metrics from Tasks 1-3 plus safety rates | Task 4 execution and student analysis |
| T4 Q2: sensitive wording versus harmful intent | Category distributions and paired prompt responses | Task 4 execution and student examples |
| T4 Q3: judge errors and impact on policy comparison | Completed blinded audit, confusion, disagreements, audit rates | Student labels and analysis |
| T5 Q1: useful distinctions beyond binary outcome | Controlled reasoning pair, S_reason, verifier-tie cases | Task 5 execution and student examples |
| T5 Q2: style/filler bias and sensitivity interpretation | Filler pair rates, raw judge output/order checks | Task 5 execution and student inspection |
| T5 Q3: mechanism vulnerability by category | All controlled pair rates, wrong-final and distractor cases | Task 5 execution and student interpretation |
| T5 Q4: transfer and general reasoning claim | GSM/SVAMP accuracy and pairwise drops, failure types | Task 5 execution and student analysis |
| T6: all five cross-task themes | Drift/reward/compute, safety and feedback tables plus earlier evidence | All tasks complete; student synthesis |

## Remaining execution order and final acceptance

See `docs/remaining_execution.md` for exact commands. Task 5 can run independently of Task 3. Task 4 requires final standard adapters from Tasks 1-3; resolve any Task 1 replacement before its final safety generation. Then complete the blinded audit, regenerate safety metrics and Task 6 tables, and re-run evidence audits.

Before submission: the main report must contain every task's required evidence within eight NeurIPS-style pages; references excluded; appendix cannot replace required main evidence. Abstract must end with an accessible public repository link. Include commands, fixed IDs, hyperparameters, small saved logs/results, attribution and code commit. Dataset/checkpoint/model caches remain ignored. Repository visibility was not checked remotely; a name containing `internal` does not establish public visibility. No commit or push was performed during this audit. New audit fixes must reach the branch used by the cloud runner before execution.

Outstanding acceptance items: DPO context truncation (current decision: retain runs, report limitation + `truncation_sensitivity.json`; revisit only if the course requires a corrected encoder); actual-token-budget reporting (state equal updates/prompts/seeds/caps plus realized totals); GPU smoke tests and full Tasks 3-5; all blinded manual labels; student-selected qualitative evidence; student-written report and synthesis; publication/commit/submission checks. The existing partial Task 6 output is not evidence that Tasks 3-5 are complete.
