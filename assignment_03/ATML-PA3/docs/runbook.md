# PA3 runbook (Kaggle 30 h/week + Colab)

## One-time setup
1. Hugging Face: create a **write** token → add as secret `HF_TOKEN` on Kaggle (*Add-ons → Secrets*,
   tick it for the notebook) and in Colab (key icon, enable notebook access).
2. GitHub: fine-grained token with *Contents: read* on `rana-taqveem/atml_internal` → secret `GITHUB_TOKEN`
   (not needed once the repo is public).
3. Upload a notebook: Kaggle → *Create → New Notebook → File → Import Notebook* →
   `assignment_03/ATML-PA3/task1_finetune/atml-pa3-task1.ipynb`. Settings: Accelerator **GPU T4 ×2**,
   Internet **on**.

## Launch checklist (every Kaggle session)
1. Notebook *Settings*: Accelerator **GPU T4 x2**, Internet **On** (needs a phone-verified account).
2. *Add-ons → Secrets*: `HF_TOKEN` and `GITHUB_TOKEN` both present **and ticked** for this notebook.
3. Config cell: `MODE` as intended (`"smoke"` once, then `"full"`).
4. Click **Save Version → Save & Run All (Commit) → Save**. Do *not* use the interactive Run button
   for long jobs: an interactive session stops when the browser closes or goes idle.
5. You can close the tab and switch off the laptop. Within ~2 minutes the version log should show
   `PREFLIGHT OK`; if it shows `PREFLIGHT FAILED`, the run stops immediately and the message says what to fix.
6. Check progress any time: Kaggle → *Your Work* → the notebook → *Versions* → *Logs*
   (runner prints a status line per job every ~10 min), or the files appearing in the HF store repo.

## Plan
| Session | Platform | Notebook / MODE | What happens | Est. |
|---|---|---|---|---|
| 1 | Kaggle T4×2 | task1, `smoke` | 20 updates × 5 variants, 16-example evals; check per-update time and memory in the log | ~1 h |
| 2 | Kaggle T4×2 | task1, `full` | data → Full FT ‖ L2-SP, then LoRA-16 ‖ OPLoRA-16 | ~11 h |
| 3 | Kaggle T4×2 | task1, `full` (same notebook) | LoRA-4 + baseline/finished evals | ~6–9 h |
| 4 | Kaggle T4×2 | task1, `full` | remaining evals, diagnostics, qualitative, summary | ~3–5 h |
| in parallel | Colab T4 | task3-colab, `LIMIT=2` then `0` | live generation, resumes per question | 2–5 h |
| next week | Kaggle T4 (or Colab) | task2, `smoke` then `full` | features → training → evaluation | ~5–6 h |

Use **Save Version → Save & Run All (Commit)** on Kaggle so it runs headless; re-run the same
notebook after each session until it prints `REMAINING JOBS: none`. After each session the job log
is in the notebook output and in `logs/` inside `task1_results_logs.zip`.

## After the runs
Results are mirrored in the private HF store (`results/...`). To pull them locally and commit:

```bash
HF_TOKEN=... python -c "from common.store import Store; s=Store(); [s.pull(p) for p in ['results/task1','results/task2','results/task3/live']]"
```

## If something fails
* A job shows `failed(3)`: the training time guard stopped it cleanly; re-run the notebook.
* Out of memory on Full FT: lower `CE_CHUNK` in `task1_finetune/train.py` (default 1024).
* lm-eval GSM8K too slow: lower `eval.batch_size_gen` only if memory is the issue; generation length
  is fixed by the protocol (1,024 tokens).
