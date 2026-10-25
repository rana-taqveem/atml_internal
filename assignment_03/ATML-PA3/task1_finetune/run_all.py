"""Task 1 end-to-end driver (used by the Kaggle notebook). Safe to re-run until complete.

    python -m task1_finetune.run_all --gpus 0,1 --hours 11.3            # full run
    python -m task1_finetune.run_all --smoke --gpus 0,1 --hours 2       # smoke test
"""
from __future__ import annotations

import argparse

from common.runner import Job, run_jobs
from common.store import Store
from common.utils import load_config

VARIANTS = ["full", "l2sp", "lora16", "oplora16", "lora4"]   # launch order
# measured on Kaggle T4 (smoke: ~7.2 s/update; OPLoRA ~9 s/update) + checkpoint uploads
TRAIN_HOURS = {"full": 2.8, "l2sp": 2.9, "lora16": 2.7, "oplora16": 3.3, "lora4": 2.6}


def jobs(smoke: bool, config: str) -> list[Job]:
    cfg = load_config(config)
    sfx = "_smoke" if smoke else ""
    c = ["--config", config]
    js = [Job("data", ["-m", "task1_finetune.data", *c], "data/task1/train_tokens.npz",
              gpu=False, hours=0.2)]
    for v in VARIANTS:
        cmd = ["-m", "task1_finetune.train", *c, "--variant", v]
        if smoke:
            cmd += ["--smoke-updates", "20"]
        js.append(Job(f"train_{v}", cmd, f"results/task1/train/{v}{sfx}/summary.json",
                      deps=["data"], hours=0.15 if smoke else TRAIN_HOURS[v],
                      budget_flag=None if smoke else "--max-hours"))
    for cond, spec in cfg["eval"]["conditions"].items():
        cmd = ["-m", "task1_finetune.evaluate", *c, "--condition", cond]
        if smoke:
            cmd += ["--limit", "16"]
        deps = [f"train_{spec['variant']}"] if spec["variant"] else ["data"]
        js.append(Job(f"eval_{cond}", cmd, f"results/task1/eval/{cond}{sfx}/scores.json",
                      deps=deps, hours=0.2 if smoke else 0.8))
    all_train = [f"train_{v}" for v in VARIANTS]
    if not smoke:
        js.append(Job("qualitative", ["-m", "task1_finetune.qualitative", *c],
                      "results/task1/qualitative/annotation_sheet_42.csv",
                      deps=all_train + ["eval_lora4", "eval_lora16", "eval_oplora16"], hours=0.6))
        js.append(Job("diagnostics", ["-m", "task1_finetune.diagnostics", *c],
                      "results/task1/diagnostics/weight_space.csv", deps=all_train, hours=0.4))
        js.append(Job("summarize", ["-m", "task1_finetune.summarize", *c],
                      "results/task1/summary/benchmarks.csv",
                      deps=[j.name for j in js], gpu=False, hours=0.1))
    return js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--hours", type=float, default=11.3)
    ap.add_argument("--only", default=None, help="comma-separated job names")
    ap.add_argument("--config", default=None, help="override (tests)")
    a = ap.parse_args()
    config = a.config or ("configs/task1_smoke.yaml" if a.smoke else "configs/task1.yaml")
    status = run_jobs(jobs(a.smoke, config), a.gpus.split(","), a.hours,
                      only=a.only.split(",") if a.only else None, store=Store())
    remaining = [k for k, v in status.items() if v != "done"]
    print("REMAINING JOBS:", remaining or "none - Task 1 complete")


if __name__ == "__main__":
    main()
