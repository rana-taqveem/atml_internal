"""Generate the Kaggle/Colab runner notebooks from one shared setup cell.

    python tools/make_notebooks.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SETUP = r'''# --- Setup: secrets, clone, install (identical in every task notebook) ---
import os, subprocess, sys

def secret(name):
    try:
        from kaggle_secrets import UserSecretsClient          # Kaggle
        return UserSecretsClient().get_secret(name)
    except Exception:
        pass
    try:
        from google.colab import userdata                       # Colab
        return userdata.get(name)
    except Exception:
        return os.environ.get(name)

os.environ["HF_TOKEN"] = secret("HF_TOKEN")
if STORE_REPO:
    os.environ["PA3_STORE_REPO"] = STORE_REPO
gh = secret("GITHUB_TOKEN")
url = REPO.replace("https://", f"https://{gh}@") if gh else REPO
# Kaggle: /kaggle/working is capped at 20 GB and saved with the notebook, so the repo and all
# checkpoints live in /tmp; only result zips are written to /kaggle/working.
KAGGLE = os.path.exists("/kaggle/working")
WORK = "/kaggle/working" if KAGGLE else "/content"
CLONE = "/tmp/pa3" if KAGGLE else "/content/pa3"
if not os.path.exists(CLONE):
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, url, CLONE], check=True)
else:
    subprocess.run(["git", "-C", CLONE, "pull", "--ff-only"], check=True)
os.chdir(f"{CLONE}/assignment_03/ATML-PA3")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)
!nvidia-smi --query-gpu=name,memory.total --format=csv
'''

COLLECT = r'''# --- Collect: small results into a zip next to the notebook output (also mirrored in the HF store) ---
import shutil
shutil.make_archive(f"{WORK}/{ZIP_NAME}", "zip", ".", RESULTS_DIR)
if os.path.exists("logs"):
    shutil.make_archive(f"{WORK}/{ZIP_NAME}_logs", "zip", ".", "logs")
print("wrote", f"{WORK}/{ZIP_NAME}.zip")
'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def notebook(cells, colab=False):
    meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}
    if colab:
        meta["accelerator"] = "GPU"
        meta["colab"] = {"provenance": []}
    else:
        meta["kaggle"] = {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True,
                          "isGpuEnabled": True}
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}


def config_cell(extra):
    return code(f'''# --- Run configuration ---
REPO = "https://github.com/rana-taqveem/atml_internal.git"
BRANCH = "main"
STORE_REPO = ""          # "" -> <your-hf-username>/atml-pa3-store (private, created automatically)
{extra.strip()}''')


SECRETS_NOTE = """
**Secrets** (Kaggle: *Add-ons → Secrets*; Colab: key icon): `HF_TOKEN` (write access; used for the
private checkpoint store and model downloads) and `GITHUB_TOKEN` (read access to the private repo;
not needed once the repo is public).
"""

TASK1 = [
    md("""
# ATML PA3 · Task 1 — LLM fine-tuning (Kaggle, GPU T4 ×2)

Runs every Task 1 job in dependency order on both GPUs (one job per GPU) and **resumes where the
previous session stopped**: finished jobs are skipped, training resumes from its latest recovery
checkpoint in the private HF store. Re-run (*Save & Run All*) until it prints
`REMAINING JOBS: none`.

1. First run `MODE = "smoke"` (~1 h): 20 updates of every variant + 16-example evaluations.
2. Then `MODE = "full"`. Expect 3–4 sessions in total (5 trainings ≈ 3–4 h each, two at a time;
   7 evaluations ≈ 1 h each; diagnostics, qualitative generations, summary).

Settings: *Accelerator* GPU T4 ×2, *Internet* on, *Persistence* off. Use **Save Version → Save &
Run All (Commit)** so it runs in the background (12 h limit; `HOURS` keeps a safety margin).
""" + SECRETS_NOTE),
    config_cell('''
MODE = "smoke"           # "smoke" first, then "full"
HOURS = 11.3             # session budget; training jobs stop cleanly and checkpoint before it
GPUS = "0,1"
ONLY = ""                # optional comma-separated job names, e.g. "train_full,train_l2sp"
'''),
    code(SETUP),
    code('''# --- Run all pending Task 1 jobs ---
args = ["--gpus", GPUS, "--hours", str(HOURS)]
if MODE == "smoke":
    args += ["--smoke"]
if ONLY:
    args += ["--only", ONLY]
!python -m task1_finetune.run_all {" ".join(args)}
'''),
    code('''# --- Peek at results so far ---
import glob, json
for f in sorted(glob.glob("results/task1/train/*/summary.json")):
    s = json.load(open(f)); print(f.split("/")[-2], f"{s['train_hours']:.2f} h", f"peak {s.get('peak_allocated_gb', 0):.1f} GB")
for f in sorted(glob.glob("results/task1/eval/*/scores.json")):
    s = json.load(open(f)); print(f.split("/")[-2], {k: s[k] for k in ("gsm8k", "hellaswag", "boolq", "arc_easy") if k in s})
if os.path.exists("results/task1/summary/benchmarks.csv"):
    import pandas as pd; display(pd.read_csv("results/task1/summary/benchmarks.csv").round(2))
'''),
    code("ZIP_NAME, RESULTS_DIR = 'task1_results', 'results/task1'\n" + COLLECT),
]


TASK2 = [
    md("""
# ATML PA3 · Task 2 — SigLIP–Qwen VLM (Kaggle GPU T4, or Colab T4)

One run through the three curriculum stages, then the final four-condition evaluation.
Jobs: `features` (download ~11k selected COCO images, cache SigLIP features, ~30 min) →
`train` (4,500 updates, resumable every 250 updates) → `evaluate` → `summarize`.
Data selection (IDs, pairs, mismatched mapping) is already committed in `results/task2/data/`.
Re-run until it prints `REMAINING JOBS: none`. Run `MODE = "smoke"` first (~30 min).
""" + SECRETS_NOTE),
    config_cell('''
MODE = "smoke"           # "smoke" first, then "full"
HOURS = 11.3
GPUS = "0"               # Task 2 is a single sequential run; one GPU is enough
'''),
    code(SETUP),
    code('''# --- Run all pending Task 2 jobs ---
args = ["--gpus", GPUS, "--hours", str(HOURS)] + (["--smoke"] if MODE == "smoke" else [])
!python -m task2_vlm.run_all {" ".join(args)}
'''),
    code('''# --- Peek ---
import glob
for f in glob.glob("results/task2/summary*/final_metrics.csv"):
    import pandas as pd; display(pd.read_csv(f).round(3))
'''),
    code("ZIP_NAME, RESULTS_DIR = 'task2_results', 'results/task2'\n" + COLLECT),
]

TASK3 = [
    md("""
# ATML PA3 · Task 3 — live generation (Colab T4)

Only Step 4 (parallel vs sequential vs single long response, 50 MATH questions, Qwen2.5-3B-Instruct,
batch size 1) needs a GPU; the offline Steps 1-3 and 5-6 are CPU-only and their results are already
in `results/task3/`. Every finished question is appended to `results/task3/live/generations.jsonl`
and pushed to the HF store, so if Colab disconnects just run all cells again.

Expected time on a T4: roughly 2-5 h for 50 questions (≤ 8,192 new tokens per question).
""" + SECRETS_NOTE),
    config_cell('''
LIMIT = 0                # 0 = all 50 questions; e.g. 2 for a quick smoke test
RUN_OFFLINE = False      # True re-runs the CPU-only offline analysis (~30-60 min)
'''),
    code(SETUP),
    code('''# --- Data: supplied rollouts (Google Drive) + fixed splits ---
!python -m task3_ttc.data
if RUN_OFFLINE:
    !python -m task3_ttc.grading
    !python -m task3_ttc.offline
    !python -m task3_ttc.signals
    !python -m task3_ttc.adaptive
'''),
    code('''# --- Step 4: live generation (resumable) ---
!python -m task3_ttc.live {"--limit " + str(LIMIT) if LIMIT else ""}
'''),
    code("!python -m task3_ttc.summarize\nZIP_NAME, RESULTS_DIR = 'task3_results', 'results/task3'\n" + COLLECT),
]


def main():
    for folder, name, cells, colab in [("task1_finetune", "atml-pa3-task1.ipynb", TASK1, False),
                                       ("task2_vlm", "atml-pa3-task2.ipynb", TASK2, False),
                                       ("task3_ttc", "atml-pa3-task3-colab.ipynb", TASK3, True)]:
        out = ROOT / folder / name
        out.write_text(json.dumps(notebook(cells, colab), indent=1), encoding="utf-8")
        print("wrote", out)


if __name__ == "__main__":
    main()
