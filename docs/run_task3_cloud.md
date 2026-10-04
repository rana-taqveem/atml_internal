# Running Task 3 (GRPO) on Kaggle / Colab

GPU: one T4. Same setup cell as Task 1/2 (`git clone -b pa2 ...` or `git pull`, install, `download_assets`,
`validate_assets`). Rough T4 estimate: standard continuation 10-20 min, two 8-update forks ~5-10 min each,
five held-out evaluations ~5-8 min each. The group-size study needs no GPU (seconds). About 1-1.5 h in total.

## Smoke test first (~5 min)
```bash
!python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name smoke --output outputs/task3_grpo/smoke --updates 2
!rm -rf outputs/task3_grpo/smoke results/task3_grpo/smoke
```

## Full run
```bash
!python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard 2>&1 | tee -a task3.log
!python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter checkpoints/grpo_midpoint_policy --name midpoint 2>&1 | tee -a task3.log
!python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter none --name sft 2>&1 | tee -a task3.log
!python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard 2>&1 | tee -a task3.log
!python -m task3_grpo.analyze_group_size --config configs/grpo.yaml 2>&1 | tee -a task3.log
!python -m task3_grpo.compare_normalization --config configs/grpo.yaml --skip-existing 2>&1 | tee -a task3.log
!python -m task3_grpo.summarize --config configs/grpo.yaml 2>&1 | tee -a task3.log
!zip -qr task3_outputs.zip outputs/task3_grpo results/task3_grpo task3.log
```
`outputs/task3_grpo/standard` is the GRPO policy used in Task 4 - keep the zip.
