"""Tiny dependency-aware job runner for Kaggle/Colab sessions.

Each job is a shell command with dependencies, an estimated duration and a "done" path in
the run store. Re-running the same notebook resumes: finished jobs are skipped, training
jobs resume from their latest recovery checkpoint. GPU jobs are placed on free GPUs
(one job per GPU, CUDA_VISIBLE_DEVICES); a job is not started if its estimate exceeds the
remaining session budget. Training jobs receive the remaining budget as --max-hours.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

from .store import Store
from .utils import ROOT, ensure_dir


@dataclass
class Job:
    name: str
    cmd: list[str]
    done: str                       # store path whose existence marks completion
    deps: list[str] = field(default_factory=list)
    hours: float = 1.0              # rough estimate on a T4
    gpu: bool = True
    budget_flag: str | None = None  # e.g. "--max-hours": pass remaining time to the job


def run_jobs(jobs: list[Job], gpus: list[str], hours: float, margin: float = 0.4,
             only: list[str] | None = None, store: Store | None = None) -> dict:
    store = store or Store()
    t_end = time.time() + hours * 3600
    logdir = ensure_dir("logs")
    status = {j.name: ("done" if store.exists(j.done) else "todo") for j in jobs}
    if only:
        for j in jobs:
            if j.name not in only and status[j.name] == "todo":
                status[j.name] = "skipped"
    by_name = {j.name: j for j in jobs}
    running: dict[str, tuple[subprocess.Popen, str | None, object]] = {}
    free = list(gpus)
    print("[runner] status:", {k: v for k, v in status.items()}, flush=True)

    def remaining():
        return (t_end - time.time()) / 3600

    while True:
        # reap finished jobs
        for name, (proc, gpu, fh) in list(running.items()):
            if proc.poll() is None:
                continue
            fh.close()
            ok = proc.returncode == 0 and store.exists(by_name[name].done)
            status[name] = "done" if ok else f"failed({proc.returncode})"
            print(f"[runner] {name} -> {status[name]} ({remaining():.2f} h left)", flush=True)
            if gpu is not None:
                free.append(gpu)
            del running[name]
        # launch ready jobs
        launched = False
        for j in jobs:
            if status[j.name] != "todo":
                continue
            if any(status.get(d) != "done" for d in j.deps):
                if any(str(status.get(d, "")).startswith("failed") for d in j.deps):
                    status[j.name] = "blocked"
                continue
            if j.gpu and not free:
                continue
            budget = remaining() - margin
            if j.budget_flag is None and j.hours > budget:
                continue
            if budget < 0.25:
                continue
            gpu = free.pop(0) if j.gpu else None
            env = dict(os.environ, PYTHONUNBUFFERED="1",
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu
            cmd = [sys.executable, *j.cmd]
            if j.budget_flag:
                cmd += [j.budget_flag, f"{budget:.2f}"]
            fh = open(logdir / f"{j.name}.log", "a", encoding="utf-8")
            print(f"[runner] start {j.name} on gpu={gpu}: {' '.join(cmd[1:])}", flush=True)
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
            running[j.name] = (proc, gpu, fh)
            status[j.name] = "running"
            launched = True
        if not running and not launched:
            break
        time.sleep(int(os.environ.get("PA3_POLL", 30)))
        _tail(running, logdir)
    print("[runner] final status:", status, flush=True)
    return status


_last_print = {}


def _tail(running, logdir, every=600):
    """Print the last log line of each running job every ~10 minutes."""
    now = time.time()
    for name in running:
        if now - _last_print.get(name, 0) < every:
            continue
        _last_print[name] = now
        try:
            lines = (logdir / f"{name}.log").read_text(encoding="utf-8", errors="ignore").splitlines()
            last = next((l for l in reversed(lines) if l.strip()), "")
            print(f"[runner] {name}: {last[-200:]}", flush=True)
        except FileNotFoundError:
            pass
