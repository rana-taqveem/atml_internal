# ATML PA2 - LLM Post-Training

<!-- FINAL_STUDENT_SETUP -->

## Quick start

Audit and remaining work: [assignment audit](docs/assignment_audit.md),
[remaining execution commands](docs/remaining_execution.md).
Run `python -m scripts.audit_assignment` to check saved evidence, and
`python -m scripts.preflight_task --task 3 --require-cuda` (or task 4/5) before GPU work.
See the audit's unresolved DPO truncation and realized-token-budget issues before
claiming full assignment compliance. New evaluation-cache checks reject partial
smoke results and generations from different policies/protocols.

```bash
git clone https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
```

The fixed datasets, cached diagnostics, and supplied
continuation checkpoints are downloaded from:

https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets

Pinned release revision:

`0b350481fb03f5525a35bcdec4131bd4fe487f98`

---
# ATML PA2 - LLM Post-Training

This is the **student starter repository** for ATML PA2. The released code is intentionally incomplete: Tasks 1-3 provide model/data loading, objective helpers, checkpoint restoration, and experiment entry points, but **you must implement the training loops and ablation orchestration yourself**. Each of Tasks 1-3 also contains one deliberate algorithmic defect in its core objective code; identifying and correcting these defects is part of validating your implementation.

Task 4 supplies the fixed AI safety judge and response-generation utilities, but you must write the evaluation/aggregation code. Task 5 supplies the exact RLVR verifier, the fixed pairwise AI judge used for RLAIF evaluation, and data/model loaders; you must implement the requested evaluation and analysis.

## 1. Clone and install

```bash
git clone https://github.com/COURSE_ORG/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
```

## 2. Download the course assets

The large course-created checkpoints and fixed data are distributed as a GitHub Release asset rather than normal Git files. After cloning, run:

```bash
python -m scripts.download_assets
python -m scripts.validate_assets
```

If your instructor provides a direct asset URL separately, use:

```bash
python -m scripts.download_assets --url '<ASSET_URL>'
```

Public base/reward/judge models are downloaded from Hugging Face at runtime and are **not** included in the course asset archive.

The installer also materializes the fixed 100-example Task 5 transfer set from the official SVAMP challenge-set source if it is not already present. The tiny Task 1 word-limit prompt set is tracked directly in this repository.

## 3. Environment check

```bash
python -m scripts.check_environment
```

Run commands from the repository root. The reference environment used to prepare the release pins Transformers 4.57.1, TRL 0.27.2, PEFT 0.17.1, and Tokenizers 0.22.1.

## 4. Supplied course checkpoints

After `download_assets`, these directories should exist:

```text
checkpoints/ppo_midpoint_policy/
checkpoints/ppo_midpoint_value/
checkpoints/grpo_midpoint_policy/
checkpoints/rlvr_policy/
checkpoints/rlaif_policy/
```

PPO and GRPO begin from the supplied continuation checkpoints. RLVR and RLAIF are supplied frozen evaluation policies; students do not retrain them.

The PPO value checkpoint is intentionally released as the exact staff midpoint state, including its imperfect held-out value calibration. Treat critic behavior as an analysis variable rather than assuming a perfect baseline, and start every PPO fork from the identical supplied policy/value state. The default continuation generation cap is 512 tokens for feasibility; frozen evaluation uses the larger cap specified in `configs/ppo.yaml`.

## 5. Task entry points

### Task 1 - DPO

Objective fix: `task1_dpo/dpo.py` used `beta * (policy_margin + ref_margin)`; the DPO logit is
`beta * (policy_margin - ref_margin)` (loss = log 2 when policy == reference). Preference accuracy
now uses the manual's margin `m > 0` instead of the raw policy margin.

```bash
# Step 1: standard DPO (1 epoch, 1500 pairs, beta=0.10) + SFT reference baseline
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter none --name sft
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
# Step 2: beta forks (first 600 training pairs each, beta in {0.03, 0.10, 0.30}), train + evaluate
python -m task1_dpo.ablate_beta --config configs/dpo.yaml --skip-existing
# Step 3: length-balanced DPO (1500 pairs, 500 per stratum) + per-stratum / word-limit comparison
python -m task1_dpo.analyze_length --config configs/dpo.yaml --skip-existing
# Tables, figures, qualitative-example shortlist (CPU only)
python -m task1_dpo.summarize --config configs/dpo.yaml
```

Outputs: adapters in `outputs/task1_dpo/<run>/`; per-run logs and metrics in `results/task1_dpo/<run>/`
(`train_log.jsonl`, `train_summary.json` incl. training prompt-ID order, wall-clock and peak VRAM,
`eval_metrics.json`, `eval_pairs_*.jsonl`, `generations.jsonl`, `word_limit.jsonl`); aggregates in
`results/task1_dpo/summary.csv`, `length_analysis.json`, `figures/`, `qualitative_candidates_*.md`.

Evaluation protocol (same for every condition, see `eval:` in `configs/dpo.yaml`): held-out DPO loss/accuracy on all
300 standard-eval pairs and all 246 length-stratified pairs; reward-model score, sampled-response KL (token mean via
`common.metrics.sampled_kl`, plus per-sequence sum), entropy and length on one seeded sample (T=0.7, top-p 0.9,
256 new tokens) for each of the first 128 held-out prompts that fit in 512 prompt tokens; word-limit compliance on
the 10 common prompts with one greedy and 5 seeded sampled responses each.

### Task 2 - PPO

Objective fix: `task2_ppo/ppo.py` combined the two surrogate terms with `torch.maximum`; the PPO
clipped surrogate is `min(rho*A, clip(rho, 1-eps, 1+eps)*A)` (with max, clipping never restrains an update).
`clip_diagnostics` adds the affected-token fraction (tokens where the clipped branch is active).

```bash
# Step 1: 20-update continuation from the supplied midpoint policy + critic, then held-out evaluation
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter checkpoints/ppo_midpoint_policy --name midpoint
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter none --name sft
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
# Step 2: cached-batch clipping study + 8-update forks for eps in {0.05, 0.20, 0.50} (kl_beta 0.10)
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml --skip-existing
# Step 3: 8-update forks for kl_beta in {0, 0.10, 0.20} (eps 0.20; the 0.10 fork is shared with step 2)
python -m task2_ppo.ablate_kl --config configs/ppo.yaml --skip-existing
python -m task2_ppo.summarize --config configs/ppo.yaml
```

Implementation choices (same for every condition): one rollout per update (`prompts_per_update: 1`) from a
seeded permutation of the train pool (prompts <= 256 tokens), sampling seed `seed + update`, so forks see the same
prompts; dropout disabled so the importance ratio is exactly 1 before an update; learned reward minus
`missing_eos_penalty` when no EOS; sampled-token KL shaping with `kl_beta`; GAE (gamma 1, lambda 0.95) with
advantages whitened over the batch; `ppo_epochs: 2` policy + critic steps per rollout. Held-out evaluation: first
64 eval-pool prompts, one seeded sample each at the 768-token evaluation cap. Logs: `results/task2_ppo/<run>/`.

### Task 3 - GRPO

Objective fix: `task3_grpo/grpo.py::group_relative_advantages` normalised rewards with the mean/std of the
whole batch, ignoring `group_ids`; the GRPO advantage uses the mean/std *within each prompt's group*.

```bash
# Step 1: 20-update continuation (K=4, truncated completions masked) + held-out evaluation
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter checkpoints/grpo_midpoint_policy --name midpoint
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter none --name sft
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
# Step 2: equal-generation group-size study on the cached K=8 completions (CPU only)
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
# Step 3: 8-update canonical GRPO vs Dr. GRPO forks + length-conditioned gradient statistics
python -m task3_grpo.compare_normalization --config configs/grpo.yaml --skip-existing
python -m task3_grpo.summarize --config configs/grpo.yaml
```

Implementation choices: rollout prompts/seeds as in Task 2 (shared `common.policy_eval.prompt_schedule`); truncated
completions keep their reward in the group mean/std but contribute no loss tokens; the Dr. GRPO condition changes
only the sequence normaliser (1/T_k -> 1/max_completion_length) and keeps the std-scaled advantage, so the comparison
isolates length normalisation; each completion is backpropagated separately (identical total gradient) to log its
own gradient norm vs. its length. Group-size study: each prompt's 8 cached completions split into disjoint K-groups
(192 generations for every K), difficulty bins = tertiles of a prompt's mean cached reward, plus random-partition
averages and a binarised-reward variant. Held-out evaluation is the Task 2 protocol (`common.policy_eval`).

### Task 4 - Safety calibration

Policies (fixed): untouched SFT, `outputs/task1_dpo/standard`, `outputs/task2_ppo/standard`,
`outputs/task3_grpo/standard`. One greedy response per XSTest prompt (256-token cap), fixed prompt order.

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml      # generated_<policy>.jsonl
python -m task4_safety.judge_responses --config configs/feedback.yaml         # judged_<policy>.jsonl (cached)
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml        # blinded manual_audit_sheet.csv
# ... fill manual_label in the sheet by hand (before looking at AI labels) ...
python -m task4_safety.evaluate_safety --config configs/feedback.yaml         # safety_metrics.json, category_labels.csv
```

Rates follow the course spec (`manifests/source_v2_task4_safety_judge_spec.json`): safe-answer = P(SAFE_ANSWER | SAFE),
over-refusal = P(OVER_REFUSAL | SAFE), unsafe-compliance = P(UNSAFE_COMPLIANCE | UNSAFE), justified-refusal =
P(JUSTIFIED_REFUSAL | UNSAFE); also ambiguous rate, labels inconsistent with the benchmark class, per-category label
distributions, length, and manual-vs-AI agreement (accuracy, Cohen's kappa, confusion matrix, disagreement list).
The audit sheet hides policy and AI label; `manual_audit_key.csv` maps rows back. `--mode all` (default, 60 IDs x 4
policies) or `--mode rotate` (60 rows, one policy per ID).

### Task 5 - RLVR vs RLAIF

```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm        # 300 GSM8K eval problems
python -m task5_feedback.score_perturbations --config configs/feedback.yaml                # 20 problems x 5 variants
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer   # 100 SVAMP problems
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

Protocol: greedy decoding, `math_max_new_tokens` cap, exact verifier = supplied `exact_reward` (last `#### <n>`),
format compliance = designated final parsed. AI pairwise result = supplied `PairwiseAIJudge` comparing each policy's
response (A) with SFT's (B) on the same problem (win 1 / tie 0.5 / loss 0; the judge balances A/B orientation by hash).
Verifier-judge agreement: verifier label A/B/TIE from correctness vs judge label. Diagnostics: four controlled pairs
against `clean_correct` (reasoning / outcome / filler / gold-distractor), each judged in both candidate orders
(primary = course orientation; order consistency reported), plus a 5-way round-robin win rate per variant;
S_reason = better rate on the reasoning pair, S_outcome = better rate pooled over the two outcome-changing pairs.
Judge and verifier inference cost is logged.

### Task 6 - Synthesis (no training, CPU)

```bash
python -m task6_synthesis.summarize     # results/task6_synthesis/: drift/reward/compute, safety, feedback-source tables
```

### Running on Kaggle

Each task folder has a runner notebook (`task1_dpo/atml-pa2-task1.ipynb` ... `task5_feedback/atml-pa2-task5.ipynb`)
that clones this branch, installs, downloads assets, runs a smoke test and then the commands above, and zips
`outputs/` + `results/` for download. Order: Task 1, 2, 3, 5 (independent), then Task 4 (needs the three
`standard` adapters; attach the task1-3 zips as a Kaggle Dataset). Task 6 runs locally on the downloaded results.

### Attribution

All training/evaluation loops, studies and analysis scripts in `task*/` and `common/policy_eval.py` were written
for this assignment on top of the course starter code (`common/`, objective helpers, judges, verifier, loaders).
No external implementation was copied; conventions follow the cited papers (DPO: Rafailov et al. 2023; PPO/GAE:
Schulman et al. 2017/2016; GRPO: Shao et al. 2024; Dr. GRPO: Liu et al. 2025). Libraries: PyTorch, Transformers,
PEFT, bitsandbytes, pandas, matplotlib. Coding assistance from an LLM (Claude) was used; all code was reviewed.

## 6. Reproducibility rules

- Do not alter course-provided data, cached rollouts, or supplied checkpoints.
- Start every short fork from the **same supplied midpoint checkpoint**.
- Keep prompt IDs, generated-token/update budgets, seed, and evaluation procedure matched across ablations.
- Commit your code, configs, small JSON/CSV logs, and figures. Do not commit downloaded checkpoints, raw course assets, or model caches.
- Record peak VRAM and wall-clock time for the standard PPO and GRPO continuations.

See the assignment manual for the required experiments, metrics, and report questions.
