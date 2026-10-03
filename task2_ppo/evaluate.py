from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json, wall_timer
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer
from common.policy_eval import evaluate_heldout, heldout_prompts


def load_evaluation_bundle(config_path: str, adapter: str | None):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def evaluate_policy(config_path: str, adapter: str | None, name: str, n_prompts: int | None = None):
    """Common held-out protocol: fixed eval prompts, seeded sampling, eval generation cap (768)."""
    bundle = load_evaluation_bundle(config_path, adapter)
    cfg = bundle["cfg"]
    ecfg = cfg.get("eval", {})
    elapsed = wall_timer()
    max_prompt = int(cfg["max_prompt_length"])
    prompts = heldout_prompts(cfg, bundle["tokenizer"], int(n_prompts or ecfg.get("prompts", 64)), max_prompt)
    records, metrics = evaluate_heldout(
        bundle["policy"], bundle["tokenizer"], bundle["reward"], cfg, prompts,
        has_adapter=adapter is not None,
        max_new_tokens=int(cfg["eval_max_response_length"]),
        max_prompt_length=max_prompt,
        batch_size=int(ecfg.get("batch_size", 8)),
        seed=int(cfg["seed"]),
        missing_eos_penalty=float(cfg["missing_eos_penalty"]),
        reward_max_length=int(cfg["reward_max_length"]),
    )
    out_dir = repo_path(cfg["results_dir"]) / name
    write_jsonl(out_dir / "heldout_generations.jsonl", records)
    metrics.update({"name": name, "adapter": adapter, "eval_wall_clock_s": round(elapsed(), 1)})
    save_json(out_dir / "eval_metrics.json", metrics)
    print(f"[{name}] held-out reward={metrics['reward_mean']:+.3f}±{metrics['reward_se']:.3f} "
          f"kl={metrics['kl_token_mean']:.4f} ent={metrics['entropy_token_mean']:.3f} "
          f"len={metrics['response_tokens']['mean']:.1f} trunc={metrics['truncation_rate']:.2f}", flush=True)
    del bundle
    clear_gpu()
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter dir, or 'none' for the untouched base policy")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--n-prompts", type=int)
    args = ap.parse_args()
    adapter = None if args.adapter.lower() == "none" else args.adapter
    evaluate_policy(args.config, adapter, args.name, args.n_prompts)


if __name__ == "__main__":
    main()
