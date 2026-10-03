from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import save_json, set_seed
from common.metrics import masked_mean
from common.models import clear_gpu, disable_dropout, load_policy, load_tokenizer, trainable_parameters
from common.policy_eval import token_logprobs_and_entropy
from task2_ppo.forks import run_fork
from task2_ppo.ppo import clip_diagnostics, compute_gae, normalize_advantages, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def reconstruct_batch(cfg, tokenizer, rows):
    """Token ids for each cached (prompt, response) plus padded old/ref/value tensors.

    The cache stores response text only; response = tokenizer(text) + EOS (if it terminated)
    reproduces the cached per-token arrays exactly (length is checked).
    """
    pool = {r["prompt_id"]: r for path in (cfg["paths"]["rl_prompt_train"], cfg["paths"]["rl_prompt_eval"])
            for r in read_jsonl(path)}
    seqs = []
    for row in rows:
        prompt_ids = tokenizer.apply_chat_template(prompt_messages(pool[row["prompt_id"]]), tokenize=True,
                                                   add_generation_prompt=True)
        resp_ids = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
        if row.get("terminated_with_eos", True):
            resp_ids = resp_ids + [tokenizer.eos_token_id]
        if len(resp_ids) != len(row["old_logprobs"]):
            raise ValueError(f"source_index {row['source_index']}: re-tokenized length {len(resp_ids)} "
                             f"!= cached {len(row['old_logprobs'])}")
        seqs.append((prompt_ids, resp_ids))

    n, t_max = len(rows), max(len(r["old_logprobs"]) for r in rows)
    pad = lambda key: torch.stack([torch.nn.functional.pad(r[key].float(), (0, t_max - len(r[key]))) for r in rows])
    mask = torch.stack([torch.nn.functional.pad(torch.ones(len(r["old_logprobs"])), (0, t_max - len(r["old_logprobs"])))
                        for r in rows])
    reward = torch.tensor([float(r.get("effective_terminal_reward", r.get("raw_terminal_reward"))) for r in rows])
    return seqs, {"old_logp": pad("old_logprobs"), "ref_logp": pad("ref_logprobs"), "values": pad("values"),
                  "mask": mask, "task_reward": reward, "n": n, "t_max": t_max}


def per_row_logprobs(policy, seqs, t_max, device, grad: bool = False):
    """New-policy log-probs of the cached tokens, one sequence at a time, padded to t_max."""
    out = []
    for prompt_ids, resp_ids in seqs:
        ids = torch.tensor([prompt_ids + resp_ids], device=device)
        attn = torch.ones_like(ids)
        resp = torch.tensor([resp_ids], device=device)
        with torch.set_grad_enabled(grad):
            lp, _ = token_logprobs_and_entropy(policy, ids, attn, len(prompt_ids), resp, with_entropy=False)
        out.append(torch.nn.functional.pad(lp[0], (0, t_max - len(resp_ids))))
    return torch.stack(out)


def ratio_summary(new_logp, old_logp, mask) -> dict:
    lr = (new_logp - old_logp)[mask.bool()].detach().cpu()
    r = lr.exp()
    return {"mean_abs_log_ratio": float(lr.abs().mean()), "ratio_p01": float(torch.quantile(r, 0.01)),
            "ratio_p99": float(torch.quantile(r, 0.99)), "ratio_min": float(r.min()), "ratio_max": float(r.max()),
            "step_kl_k3": float((r - 1 - lr).mean())}


def cached_batch_study(cfg, probe_steps: int):
    """Clipped surrogate / clip & affected fractions on the fixed cached batch for every eps.

    (a) ratios of the supplied midpoint policy against the cached old log-probs (no update);
    (b) `probe_steps` clipped-surrogate gradient steps on this same batch, restarted from the midpoint
        for each eps, re-measuring the diagnostics before every step (immediate geometric effect).
    """
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    seqs, b = reconstruct_batch(cfg, tokenizer, rows)
    policy = disable_dropout(load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True))
    device = next(policy.parameters()).device
    old, ref, values, mask = (b[k].to(device) for k in ("old_logp", "ref_logp", "values", "mask"))

    rewards = shaped_rewards(b["task_reward"].to(device), old, ref, mask, float(cfg["kl_beta"]))
    adv, _ = compute_gae(rewards, values * mask, mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
    if cfg.get("normalize_advantages", True):
        adv = normalize_advantages(adv, mask)

    init_state = {n: p.detach().clone() for n, p in policy.named_parameters() if p.requires_grad}
    result = {"n_rollouts": b["n"], "n_tokens": int(mask.sum()), "kl_beta": float(cfg["kl_beta"]),
              "probe_steps": probe_steps, "policy_learning_rate": float(cfg["policy_learning_rate"]),
              "positive_advantage_fraction": float(masked_mean((adv > 0).float(), mask)), "eps": {}}

    new0 = per_row_logprobs(policy, seqs, b["t_max"], device)
    result["midpoint_vs_cached_old"] = ratio_summary(new0, old, mask)
    for eps in cfg["clip_values"]:
        eps = float(eps)
        entry = {"at_midpoint": clip_diagnostics(torch.exp(new0 - old), adv, mask, eps), "probe": []}
        with torch.no_grad():
            for n, p in policy.named_parameters():
                if n in init_state:
                    p.copy_(init_state[n])
        opt = AdamW(trainable_parameters(policy), lr=float(cfg["policy_learning_rate"]))
        n_tok = mask.sum()
        for step in range(probe_steps + 1):
            new = per_row_logprobs(policy, seqs, b["t_max"], device)
            ratio = torch.exp(new - old)
            entry["probe"].append({"step": step, **clip_diagnostics(ratio, adv, mask, eps),
                                   **ratio_summary(new, old, mask)})
            if step == probe_steps:
                break
            opt.zero_grad(set_to_none=True)
            for i, (prompt_ids, resp_ids) in enumerate(seqs):  # batch-level masked mean, one row at a time
                lp = per_row_logprobs(policy, [(prompt_ids, resp_ids)], b["t_max"], device, grad=True)[0]
                r = torch.exp(lp - old[i])
                obj = torch.minimum(r * adv[i], r.clamp(1 - eps, 1 + eps) * adv[i])
                (-(obj * mask[i]).sum() / n_tok).backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float(cfg["max_grad_norm"]))
            opt.step()
        result["eps"][f"{eps:g}"] = entry
        last = entry["probe"][-1]
        print(f"[cached eps={eps:g}] midpoint clip={entry['at_midpoint']['clip_fraction']:.4f} "
              f"affected={entry['at_midpoint']['affected_fraction']:.4f} | after {probe_steps} steps "
              f"clip={last['clip_fraction']:.4f} affected={last['affected_fraction']:.4f} "
              f"surr={last['clipped_surrogate']:+.4f} step_kl={last['step_kl_k3']:.5f}", flush=True)
    del policy, opt
    clear_gpu()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--skip-existing", action="store_true", help="reuse fork adapters/metrics that already exist")
    ap.add_argument("--skip-cached", action="store_true")
    ap.add_argument("--skip-forks", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    print("Cached PPO rollouts:", len(rows))
    print("Required epsilon values:", cfg["clip_values"])
    print("Cache keys:", sorted(rows[0].keys()))

    if not args.skip_cached:
        study = cached_batch_study(cfg, int(cfg.get("clip_probe_steps", 4)))
        save_json(repo_path(cfg["results_dir"]) / "clipping_cached.json", study)
    if not args.skip_forks:
        # kl_beta fixed at the reference value; only eps changes.
        for eps in cfg["clip_values"]:
            run_fork(args.config, float(eps), float(cfg["kl_beta"]), args.skip_existing)


if __name__ == "__main__":
    main()
