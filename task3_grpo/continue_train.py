from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import masked_mean, safe_corr, sampled_kl
from common.models import clear_gpu, disable_dropout, load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters, upcast_trainable
from common.policy_eval import prompt_schedule, token_logprobs_and_entropy
from task3_grpo.grpo import (
    group_relative_advantages,
    group_reward_stats,
    grpo_policy_loss,
    mask_truncated_sequences,
    per_sequence_loss_terms,
)


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    disable_dropout(policy)
    upcast_trainable(policy)
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


@torch.no_grad()
def collect_groups(bundle, rows, cfg, seed: int):
    """K completions per prompt, learned reward, group-relative advantages, old/ref log-probs."""
    policy, tokenizer = bundle["policy"], bundle["tokenizer"]
    k = int(cfg["num_generations"])
    messages = [prompt_messages(r) for r in rows for _ in range(k)]
    group_ids = torch.arange(len(rows)).repeat_interleave(k)
    set_seed(seed)
    gen = batch_generate(policy, tokenizer, messages, max_prompt_length=int(cfg["max_prompt_length"]),
                         max_new_tokens=int(cfg["max_completion_length"]),
                         temperature=float(cfg["generation"]["temperature"]),
                         top_p=float(cfg["generation"]["top_p"]), do_sample=True)
    gen = {key: (v.clone() if torch.is_tensor(v) else v) for key, v in gen.items()}  # leave inference_mode
    args = (gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"])
    old_logp, entropy = token_logprobs_and_entropy(policy, *args)
    with reference_mode(policy):
        ref_logp, _ = token_logprobs_and_entropy(policy, *args, with_entropy=False)
    rewards = score_reward_pairs(bundle["reward_model"], bundle["reward_tokenizer"], messages, gen["responses"],
                                 max_length=int(cfg.get("reward_max_length", 1280))).float().cpu()
    adv = group_relative_advantages(rewards, group_ids)
    stds, informative = group_reward_stats(rewards, group_ids)

    mask = gen["response_mask"]
    # Truncated completions keep their reward in the group statistics but contribute no loss tokens.
    loss_mask = mask_truncated_sequences(mask, gen["truncated"]) if cfg.get("mask_truncated_completions") else mask
    return {
        "args": args, "mask": mask, "loss_mask": loss_mask, "old_logp": old_logp, "ref_logp": ref_logp,
        "advantages": adv.to(mask.device), "rewards": rewards, "group_ids": group_ids,
        "stats": {
            "reward": float(rewards.mean()),
            "group_reward_std": float(stds.mean()),
            "uninformative_group_fraction": float(1 - informative.float().mean()),
            "kl_token_mean": float(sampled_kl(old_logp, ref_logp, mask)),
            "entropy": float(masked_mean(entropy, mask)),
            "response_tokens": float(mask.sum(-1).mean()),
            "truncated_fraction": float(np.mean(gen["truncated"])),
            "loss_masked_fraction": float((loss_mask.sum(-1) == 0).float().mean()),
            "advantage_abs_mean": float(adv.abs().mean()),
        },
        "records": [{"prompt_id": rows[int(g)]["prompt_id"], "group": int(g), "response": y, "response_tokens": n,
                     "truncated": t, "reward": float(r), "advantage": float(a)}
                    for g, y, n, t, r, a in zip(group_ids, gen["responses"], gen["response_lengths"],
                                                gen["truncated"], rewards, adv)],
    }


def grpo_update(bundle, batch, cfg, loss_type: str):
    """One clipped GRPO step; each completion is backpropagated separately to log its gradient norm."""
    policy, opt = bundle["policy"], bundle["optimizer"]
    params = trainable_parameters(policy)
    eps, beta, max_len = float(cfg["clip_epsilon"]), float(cfg["kl_beta"]), int(cfg["max_completion_length"])
    lmask = batch["loss_mask"]
    per_epoch = []
    for _ in range(int(cfg["policy_epochs"])):
        opt.zero_grad(set_to_none=True)
        new_rows, seq_grad_norms = [], []
        for i in range(lmask.shape[0]):
            sl = slice(i, i + 1)
            new_i, _ = token_logprobs_and_entropy(policy, *(a[sl] if torch.is_tensor(a) else a for a in batch["args"]),
                                                  with_entropy=False)
            new_rows.append(new_i.detach())
            if lmask[i].sum() == 0:
                seq_grad_norms.append(0.0)
                continue
            before = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
            # This completion's additive share of the batch loss (batch-level denominators).
            term = per_sequence_loss_terms(new_i, batch["old_logp"][sl], batch["advantages"][sl], lmask[sl],
                                           batch["ref_logp"][sl], eps, beta, loss_type, max_len,
                                           n_sequences=lmask.shape[0], total_tokens=lmask.sum())[0]
            term.backward()
            seq_grad_norms.append(float(torch.sqrt(sum(((p.grad - b) ** 2).sum() for p, b in zip(params, before)))))
        new_logp = torch.cat(new_rows)
        with torch.no_grad():
            loss, diag = grpo_policy_loss(new_logp, batch["old_logp"], batch["advantages"], lmask, batch["ref_logp"],
                                          eps, beta, loss_type, max_len)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite GRPO loss {float(loss)} (advantages finite: "
                               f"{bool(torch.isfinite(batch['advantages']).all())}); stopping instead of skipping updates")
        gn = torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"]))
        if torch.isfinite(gn):
            opt.step()
        per_epoch.append({"policy_loss": float(loss), "policy_term": float(diag["policy_term"]),
                          "kl_k3": float(diag["sampled_kl"]), "clip_fraction": float(diag["clip_fraction"]),
                          "grad_norm": float(gn), "nonfinite": not bool(torch.isfinite(gn)),
                          "seq_grad_norms": seq_grad_norms})
    lengths = batch["mask"].sum(-1).tolist()
    last = per_epoch[-1]
    kept = [j for j in range(len(lengths)) if lmask[j].sum() > 0]
    return {
        "policy_loss": float(np.mean([e["policy_loss"] for e in per_epoch])),
        "policy_term": float(np.mean([e["policy_term"] for e in per_epoch])),
        "kl_k3": float(np.mean([e["kl_k3"] for e in per_epoch])),
        "clip_fraction": float(np.mean([e["clip_fraction"] for e in per_epoch])),
        "grad_norm": float(np.mean([e["grad_norm"] for e in per_epoch])),
        "nonfinite_steps": int(sum(e["nonfinite"] for e in per_epoch)),
        # Length-conditioned gradient statistics (last epoch, completions that carry loss tokens).
        "seq_grad_norms": last["seq_grad_norms"],
        "seq_lengths": [int(x) for x in lengths],
        "seq_advantages": [float(a) for a in batch["advantages"].tolist()],
        "corr_seq_grad_norm_vs_length": safe_corr([last["seq_grad_norms"][j] for j in kept], [lengths[j] for j in kept]),
    }


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard"):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    n_updates = int(cfg["updates"])
    results_dir = repo_path(cfg["results_dir"]) / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path, rollout_path = results_dir / "train_log.jsonl", results_dir / "rollouts.jsonl"
    for p in (log_path, rollout_path):
        if p.exists():
            p.unlink()
    schedule = prompt_schedule(cfg, bundle["tokenizer"], bundle["prompt_rows"], n_updates)
    print(f"[{run_name}] updates={n_updates} loss_type={loss_type} K={cfg['num_generations']} "
          f"prompts/update={cfg['prompts_per_update']} eps={cfg['clip_epsilon']} kl_beta={cfg['kl_beta']} "
          f"max_completion={cfg['max_completion_length']} mask_truncated={cfg.get('mask_truncated_completions')}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    elapsed = wall_timer()
    tokens_generated = 0
    for u, rows in enumerate(schedule, start=1):
        batch = collect_groups(bundle, rows, cfg, seed=int(cfg["seed"]) + u)
        tokens_generated += int(batch["mask"].sum())
        stats = grpo_update(bundle, batch, cfg, loss_type)
        record = {"update": u, "prompt_ids": [r["prompt_id"] for r in rows], **batch["stats"], **stats,
                  "tokens_generated": tokens_generated, "elapsed_s": round(elapsed(), 2)}
        append_jsonl(log_path, record)
        for rec in batch["records"]:
            append_jsonl(rollout_path, {"update": u, **rec})
        print(f"[{run_name}] update {u}/{n_updates} reward={record['reward']:+.3f} "
              f"grp_std={record['group_reward_std']:.3f} uninf={record['uninformative_group_fraction']:.2f} "
              f"kl={record['kl_token_mean']:.4f} ent={record['entropy']:.3f} len={record['response_tokens']:.0f} "
              f"trunc={record['truncated_fraction']:.2f} loss={record['policy_loss']:+.4f} gn={record['grad_norm']:.3f} "
              f"t={record['elapsed_s']:.0f}s", flush=True)
        del batch

    wall_s = elapsed()
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    bundle["policy"].save_pretrained(str(out))
    bundle["tokenizer"].save_pretrained(str(out))
    save_json(results_dir / "train_summary.json", {
        "run_name": run_name, "adapter_path": str(out), "start_policy": cfg["paths"]["grpo_midpoint_policy"],
        "loss_type": loss_type, "updates": n_updates, "num_generations": int(cfg["num_generations"]),
        "prompts_per_update": int(cfg["prompts_per_update"]), "policy_epochs": int(cfg["policy_epochs"]),
        "learning_rate": float(cfg["learning_rate"]), "clip_epsilon": float(cfg["clip_epsilon"]),
        "kl_beta": float(cfg["kl_beta"]), "max_prompt_length": int(cfg["max_prompt_length"]),
        "max_completion_length": int(cfg["max_completion_length"]),
        "mask_truncated_completions": bool(cfg.get("mask_truncated_completions")),
        "generation": cfg["generation"], "seed": int(cfg["seed"]), "tokens_generated": tokens_generated,
        "wall_clock_s": round(wall_s, 1), "peak_vram_gib": None if peak is None else round(peak, 3),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "prompt_ids_by_update": [[r["prompt_id"] for r in rows] for rows in schedule],
    })
    print(f"[{run_name}] done in {wall_s / 60:.1f} min, peak VRAM {peak if peak is None else round(peak, 2)} GiB -> {out}",
          flush=True)
    del bundle
    clear_gpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
