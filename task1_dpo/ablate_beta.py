from __future__ import annotations

import argparse

from common.data import load_yaml, repo_path
from task1_dpo.evaluate import evaluate_policy
from task1_dpo.train import run_training


def beta_run_name(beta: float) -> str:
    return f"beta_{beta:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--betas", type=float, nargs="+", help="override the configured beta grid")
    ap.add_argument("--skip-existing", action="store_true", help="reuse adapters/metrics that already exist")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    betas = args.betas or [float(b) for b in cfg["betas"]]
    n_examples = int(cfg["short_ablation_examples"])
    print("Required beta values:", betas)
    print("Short-run examples per condition:", n_examples)

    # Every fork: fresh LoRA on the original policy, same first-N training pairs, same seed/order,
    # same optimizer and LoRA config. Only beta changes.
    for beta in betas:
        name = beta_run_name(beta)
        output = cfg["beta_output_template"].format(beta=f"{beta:g}")
        adapter_ready = (repo_path(output) / "adapter_config.json").exists()
        if args.skip_existing and adapter_ready:
            print(f"[{name}] reusing existing adapter at {output}")
        else:
            run_training(args.config, name, output_path=output, beta=beta, max_examples=n_examples)

        if args.skip_eval:
            continue
        if args.skip_existing and (repo_path(cfg["results_dir"]) / name / "eval_metrics.json").exists():
            print(f"[{name}] reusing existing eval metrics")
            continue
        evaluate_policy(args.config, output, name, beta=beta)


if __name__ == "__main__":
    main()
