# Running Task 2 (PPO) on Kaggle / Colab

GPU: one T4. Setup cell is the same as Task 1 (`git clone -b pa2 ...`, install, `download_assets`,
`validate_assets`). Rough T4 estimate: standard continuation 10-20 min, cached clipping study ~10 min,
five 8-update forks ~5-10 min each, seven held-out evaluations ~5-8 min each. About 1.5-2.5 h in total.

## Smoke test first (~5 min)
```bash
!python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name smoke --output outputs/task2_ppo/smoke --updates 2
!python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/smoke --name smoke --n-prompts 8
!rm -rf outputs/task2_ppo/smoke* results/task2_ppo/smoke
```

## Full run
```bash
!python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard 2>&1 | tee -a task2.log
!python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter checkpoints/ppo_midpoint_policy --name midpoint 2>&1 | tee -a task2.log
!python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter none --name sft 2>&1 | tee -a task2.log
!python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard 2>&1 | tee -a task2.log
!python -m task2_ppo.analyze_clipping --config configs/ppo.yaml --skip-existing 2>&1 | tee -a task2.log
!python -m task2_ppo.ablate_kl --config configs/ppo.yaml --skip-existing 2>&1 | tee -a task2.log
!python -m task2_ppo.summarize --config configs/ppo.yaml 2>&1 | tee -a task2.log
!zip -qr task2_outputs.zip outputs/task2_ppo results/task2_ppo task2.log
```
`outputs/task2_ppo/standard` is the PPO policy used in Task 4 - keep the zip.
