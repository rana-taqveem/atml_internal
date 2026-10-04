from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl
from common.models import load_policy, load_reward_model, load_tokenizer
from common.policy_eval import evaluate_adapter


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def evaluate_policy(config_path: str, adapter: str | None, name: str, n_prompts: int | None = None):
    """Same held-out protocol as Task 2 (prompts, seed, decoding, 768-token cap, reward model, KL)."""
    return evaluate_adapter(load_yaml(config_path), adapter, name, n_prompts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter dir, or 'none' for the untouched base policy")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--n-prompts", type=int)
    args = ap.parse_args()
    adapter = None if args.adapter.lower() == "none" else args.adapter
    evaluate_policy(args.config, adapter, args.name, args.n_prompts)


if __name__ == "__main__":
    main()
