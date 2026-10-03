# Running Task 1 (DPO) on Kaggle / Colab

GPU: one T4 (16 GB) is enough. On Kaggle choose **GPU T4 x2** (the code uses one GPU); avoid P100.
Expected time on a T4 (rough): standard train ~45-70 min, each beta fork ~20-30 min, length-balanced train
~45-70 min, each evaluation ~10-20 min. Total roughly 4-6 h, so split it across sessions (cells 4-7 are
independent and resumable with `--skip-existing`).

## Cell 1 - code + environment
```bash
!git clone -b pa2 https://github.com/rana-taqveem/atml_internal.git pa2
%cd pa2
!pip install -q -r requirements.txt
!python -m scripts.download_assets
!python -m scripts.validate_assets
!python -m scripts.check_environment
```

## Cell 2 (optional) - restore earlier results
If a previous session produced `task1_outputs.zip`, upload it and run `!unzip -o task1_outputs.zip`.

## Cell 3 - Step 1: standard DPO + SFT baseline
```bash
!python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
!python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter none --name sft
!python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
```

## Cell 4 - Step 2: beta study
```bash
!python -m task1_dpo.ablate_beta --config configs/dpo.yaml --skip-existing
```

## Cell 5 - Step 3: length study
```bash
!python -m task1_dpo.analyze_length --config configs/dpo.yaml --skip-existing
```

## Cell 6 - tables, figures, qualitative shortlist
```bash
!python -m task1_dpo.summarize --config configs/dpo.yaml
```

## Cell 7 - save everything (run at the end of EVERY session)
```bash
!zip -qr task1_outputs.zip outputs/task1_dpo results/task1_dpo
```
Download `task1_outputs.zip` (Kaggle: Output panel; Colab: `from google.colab import files; files.download('task1_outputs.zip')`).
`outputs/task1_dpo/standard` is needed again in Task 4. Commit `results/task1_dpo/` to Git; keep `outputs/` out of Git.
