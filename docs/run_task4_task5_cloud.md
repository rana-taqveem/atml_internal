# Running Tasks 4 and 5 on Kaggle / Colab

Same setup cell as before (`git clone -b pa2 ...` or `git pull`, install, `download_assets`, `validate_assets`).

## Task 5 - RLVR vs RLAIF (independent of Tasks 1-4; ~1-1.5 h on a T4)
Uses only the supplied `checkpoints/rlvr_policy` and `checkpoints/rlaif_policy`.
```bash
!python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm --limit 8      # smoke test (~3 min)
!rm -rf results/task5_feedback
!python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm 2>&1 | tee -a task5.log
!python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer 2>&1 | tee -a task5.log
!python -m task5_feedback.score_perturbations --config configs/feedback.yaml 2>&1 | tee -a task5.log
!python -m task5_feedback.compare_feedback --config configs/feedback.yaml 2>&1 | tee -a task5.log
!zip -qr task5_outputs.zip results/task5_feedback task5.log
```

## Task 4 - Safety calibration (needs the standard Task 1-3 adapters; ~1.5-2 h on a T4)
The four fixed policies are SFT, `outputs/task1_dpo/standard`, `outputs/task2_ppo/standard`,
`outputs/task3_grpo/standard`. Bring them into the session first, e.g. upload `task1_outputs.zip`,
`task2_outputs.zip`, `task3_outputs.zip` as a Kaggle Dataset, then:
```bash
!for z in /kaggle/input/<your-dataset>/task*_outputs.zip; do unzip -oq "$z"; done
!ls outputs/task1_dpo/standard outputs/task2_ppo/standard outputs/task3_grpo/standard
```
Run:
```bash
!python -m task4_safety.generate_responses --config configs/feedback.yaml 2>&1 | tee -a task4.log
!python -m task4_safety.judge_responses --config configs/feedback.yaml 2>&1 | tee -a task4.log
!python -m task4_safety.make_audit_sheet --config configs/feedback.yaml 2>&1 | tee -a task4.log
!python -m task4_safety.evaluate_safety --config configs/feedback.yaml 2>&1 | tee -a task4.log
!zip -qr task4_outputs.zip results/task4_safety task4.log
```
`judge_responses` caches every label in `judge_cache.jsonl`, so rerunning after a disconnect resumes.

### Manual audit (done by you, locally, no GPU)
1. Unzip `task4_outputs.zip` locally. Open **only** `results/task4_safety/manual_audit_sheet.csv`
   (blinded: prompt + response, no policy, no AI label). Do not open `judged_*.jsonl` or
   `manual_audit_key.csv` before you finish.
2. Fill `manual_label` with SAFE_ANSWER / JUSTIFIED_REFUSAL / UNSAFE_COMPLIANCE / OVER_REFUSAL / AMBIGUOUS
   (same definitions as the judge prompt in `task4_safety/judge_responses.py`).
3. Rerun `python -m task4_safety.evaluate_safety --config configs/feedback.yaml` (CPU) for agreement,
   kappa, confusion matrix and disagreement list.

Sheet size: default `--mode all` = 60 audit IDs x 4 policies = 240 rows; `--mode rotate` = 60 rows
(each ID paired with one policy, 15 per policy). Regenerate with `make_audit_sheet --mode rotate` after
deleting the sheet if you choose the smaller audit.
