from __future__ import annotations

import argparse

from common.data import load_yaml
from task2_ppo.forks import run_fork


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--skip-existing", action="store_true", help="reuse adapters/metrics that already exist")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("KL beta conditions:", cfg["kl_values"])
    print("Fork update budget:", cfg["fork_updates"])
    # eps fixed at the reference value; only the KL-shaping coefficient changes.
    for beta in cfg["kl_values"]:
        run_fork(args.config, float(cfg["clip_epsilon"]), float(beta), args.skip_existing, not args.skip_eval)


if __name__ == "__main__":
    main()
