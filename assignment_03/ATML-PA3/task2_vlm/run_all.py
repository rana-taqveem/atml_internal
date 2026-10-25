"""Task 2 end-to-end driver (Kaggle/Colab notebook). Safe to re-run until complete.

    python -m task2_vlm.run_all --hours 11.3            # full run
    python -m task2_vlm.run_all --smoke --hours 2       # smoke test
"""
from __future__ import annotations

import argparse

from common.runner import Job, run_jobs


def jobs(smoke: bool, config: str) -> list[Job]:
    c = ["--config", config]
    sfx = "_smoke" if smoke else ""
    return [
        Job("features", ["-m", "task2_vlm.features", *c], "results/task2/data/feature_index.json", hours=0.6),
        Job("train", ["-m", "task2_vlm.train", *c], f"results/task2/train{sfx}/summary.json",
            deps=["features"], hours=0.3 if smoke else 4.0, budget_flag=None if smoke else "--max-hours"),
        Job("evaluate", ["-m", "task2_vlm.evaluate", *c], f"results/task2/eval{sfx}/metrics.csv",
            deps=["train"], hours=0.4),
        Job("summarize", ["-m", "task2_vlm.summarize", *c], f"results/task2/summary{sfx}/final_metrics.csv",
            deps=["evaluate"], gpu=False, hours=0.1),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--hours", type=float, default=11.3)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    config = a.config or ("configs/task2_smoke.yaml" if a.smoke else "configs/task2.yaml")
    status = run_jobs(jobs(a.smoke, config), a.gpus.split(","), a.hours)
    remaining = [k for k, v in status.items() if v != "done"]
    print("REMAINING JOBS:", remaining or "none - Task 2 complete")


if __name__ == "__main__":
    main()
