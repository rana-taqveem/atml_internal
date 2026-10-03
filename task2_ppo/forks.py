"""Matched short PPO forks: same midpoint policy/critic, prompts, seeds and update budget."""
from __future__ import annotations

from common.data import load_yaml, repo_path
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate_policy


def fork_name(eps: float, kl_beta: float) -> str:
    return f"fork_eps{eps:g}_kl{kl_beta:g}"


def run_fork(config_path: str, eps: float, kl_beta: float, skip_existing: bool = False, evaluate: bool = True):
    """Train (fork_updates updates from the supplied midpoint) and evaluate one fork.

    The clipping and KL studies share the (eps=0.20, kl_beta=0.10) fork, so it is trained once.
    """
    cfg = load_yaml(config_path)
    name = fork_name(eps, kl_beta)
    output = cfg["fork_output_template"].format(name=name)
    if skip_existing and (repo_path(output) / "adapter_config.json").exists():
        print(f"[{name}] reusing existing adapter at {output}")
    else:
        run_ppo(config_path, output=output, updates=int(cfg["fork_updates"]), clip_epsilon=eps,
                kl_beta=kl_beta, run_name=name)
    if evaluate:
        if skip_existing and (repo_path(cfg["results_dir"]) / name / "eval_metrics.json").exists():
            print(f"[{name}] reusing existing eval metrics")
        else:
            evaluate_policy(config_path, output, name)
    return name
