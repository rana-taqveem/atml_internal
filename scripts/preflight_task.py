"""Fail before expensive inference when a task's dependencies are incomplete."""
from __future__ import annotations

import argparse
import importlib.metadata
import json

from common.data import load_yaml, repo_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, choices=[3, 4, 5], required=True)
    ap.add_argument("--require-cuda", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml("configs/feedback.yaml")
    errors = []
    for package, expected in {"transformers": "4.57.1", "tokenizers": "0.22.1", "peft": "0.17.1", "trl": "0.27.2"}.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        if actual != expected:
            errors.append(f"{package}: expected {expected}, found {actual}")
    if args.require_cuda:
        import torch
        if not torch.cuda.is_available():
            errors.append("CUDA is unavailable in this Python environment")
    keys = {3: ["grpo_midpoint_policy"], 4: [], 5: ["rlvr_policy", "rlaif_policy"]}[args.task]
    adapters = [(k, cfg["paths"][k]) for k in keys]
    if args.task == 4:
        adapters += [(name, cfg["policies"][name]) for name in ["dpo", "ppo", "grpo"]]
        for name, task, budget_field, expected in [("dpo", "task1_dpo", "n_pairs", 1500),
                                                   ("ppo", "task2_ppo", "updates", 20),
                                                   ("grpo", "task3_grpo", "updates", 20)]:
            path = repo_path(f"results/{task}/standard/train_summary.json")
            if not path.is_file():
                errors.append(f"{name}: missing standard training summary {path}")
            else:
                summary = json.loads(path.read_text(encoding="utf-8"))
                if summary.get(budget_field) != expected or (name == "dpo" and summary.get("epochs") != 1):
                    errors.append(f"{name}: checkpoint is not the required standard training budget")
    for name, path in adapters:
        for filename in ["adapter_config.json", "adapter_model.safetensors"]:
            if not (repo_path(path) / filename).is_file():
                errors.append(f"{name}: missing {path}/{filename}")
        print(name, path)
    data_keys = {3: ["rl_prompt_train", "rl_prompt_eval", "grpo_k_cache"],
                 4: ["xstest"], 5: ["gsm_eval", "math_transfer_eval", "task5_diagnostics"]}[args.task]
    for key in data_keys:
        if not repo_path(cfg["paths"][key]).is_file():
            errors.append(f"missing course asset: {cfg['paths'][key]}")
    if errors:
        raise SystemExit("Task preflight failed:\n- " + "\n- ".join(errors))
    print(f"Task {args.task} dependencies ready. Run scripts.validate_assets and scripts.audit_assignment for release/evidence validation.")


if __name__ == "__main__":
    main()
