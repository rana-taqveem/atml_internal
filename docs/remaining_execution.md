# Remaining PA2 execution

Commands run from the PA2 repository root in the pinned GPU environment. Use the updated source from this audit. No GPU runs were launched locally. Before the final Task 4 run, resolve the DPO truncation decision in `docs/assignment_audit.md`; replacing standard DPO later requires regenerating its safety responses.

## Environment and evidence checks

```bash
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
python -m scripts.check_environment
python -m unittest discover -s tests -v
python -m scripts.audit_assignment
```

Keep CPU-only audits separate from successful GPU inference/training. `audit_assignment` can pass consistency checks while still listing unexecuted tasks. The current asset bundle already contains the 100-row transfer file; if a different installation lacks it, run `python -m scripts.prepare_transfer_eval` once, then validate assets. Do not change its subset/order.

## Task 3

```bash
python -m scripts.preflight_task --task 3 --require-cuda
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name smoke --output outputs/task3_grpo/smoke --updates 2
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/smoke --name smoke --n-prompts 8
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter checkpoints/grpo_midpoint_policy --name midpoint
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter none --name sft
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
python -m task3_grpo.compare_normalization --config configs/grpo.yaml --skip-existing
python -m task3_grpo.summarize --config configs/grpo.yaml
python -m scripts.audit_assignment
```

Inspect smoke logs for finite gradients/losses, informative groups, nonzero loss-token counts and VRAM before the standard run. Keep smoke outputs named `smoke`. Accept 20 standard updates, 8 per fork, 64 held-out prompts per full condition. Verify actual token totals for the two forks and resolve the budget issue before claiming exact equality. Preserve `outputs/task3_grpo/standard` and its training summary for Task 4.

## Task 5 (independent of Task 3)

```bash
python -m scripts.preflight_task --task 5 --require-cuda
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm --limit 8
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

The full GSM command intentionally omits `--skip-existing` after the eight-problem smoke run. A full run must produce 300 generations per policy, not eight. Existing judge cache calls can be reused. Transfer: 100 per policy. Diagnostics: 20 problems x 5 variants; four controlled pairs per problem and the round robin. Inspect `judge_parse_ambiguous_count` and any unknown parse-status counts; a parse-fallback tie is not confirmed judge indecision.

## Task 4 (after standard Tasks 1-3)

Copy all three standard adapters **and their corresponding `results/<task>/standard/train_summary.json` files** from the execution bundles. The preflight checks the documented training budgets. Use the original experiment bundles, not adapters selected using safety results.

```bash
python -m scripts.preflight_task --task 4 --require-cuda
python -m task4_safety.generate_responses --config configs/feedback.yaml --skip-existing
python -m task4_safety.judge_responses --config configs/feedback.yaml
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml --mode all
python -m task4_safety.evaluate_safety --config configs/feedback.yaml
```

Required: 450 generated and 450 judged rows for each of SFT/DPO/PPO/GRPO. New generation fingerprints prevent reuse under a different adapter, dataset or generation cap. If a cache lacks provenance, preserve it separately and regenerate without `--skip-existing`.

The student then fills all 240 blinded response rows in `results/task4_safety/manual_audit_sheet.csv` before opening AI label files or the audit key. Do not ask an AI to supply the manual labels. Afterward:

```bash
python -m task4_safety.evaluate_safety --config configs/feedback.yaml
```

Confirm `manual_audit.status` is `complete`, `n_labeled == n_rows`, and there are 60 unique audited prompt IDs, balanced 30/30. Include ambiguous cases in the agreement breakdown rather than discarding them.

## Task 6 and final evidence collection

```bash
python -m task6_synthesis.summarize
python -m scripts.audit_assignment --require-complete
python -m scripts.audit_tokenization
python -m unittest discover -s tests -v
```

Inspect the missing-file section of the evidence JSON and the actual counts in full Task 4/5 outputs. A regenerated synthesis table still needs the student's own interpretation. Keep each reported value linked to a saved file and reproduction command; include required qualitative excerpts selected by the student. Follow the submission checks in `assignment_audit.md`. These commands do not create or submit the report and do not publish Git changes.
