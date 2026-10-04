from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import masked_mean, sampled_kl
from common.models import (
    clear_gpu,
    disable_dropout,
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    upcast_trainable,
    value_parameter_groups,
)
from common.policy_eval import prompt_schedule, token_logprobs_and_entropy
from task2_ppo.ppo import clip_diagnostics, compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    disable_dropout(policy)
    disable_dropout(value_model)
    upcast_trainable(policy)
    upcast_trainable(value_model)  # fp16 critic head + AdamW -> NaN on the first step otherwise
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def response_values(value_model, sequences, attention_mask, prompt_width, steps):
    """V(s_t) for response step t: the critic's output at the position that has seen prompt + y_<t."""
    values = token_values(value_model, sequences, attention_mask)
    return values[:, prompt_width - 1:-1][:, :steps].float()


def explained_variance(values, returns, mask) -> float:
    v, r = values[mask.bool()], returns[mask.bool()]
    var_r = r.var(unbiased=False)
    return float(1 - (r - v).var(unbiased=False) / var_r) if var_r > 0 else float("nan")


def build_ppo_batch(*, sequences, attention_mask, prompt_width, response_ids, mask, old_logp, ref_logp, values,
                    task_reward, cfg, kl_beta: float):
    """KL-shaped token rewards -> GAE advantages/returns for one rollout batch."""
    rewards = shaped_rewards(task_reward, old_logp, ref_logp, mask, kl_beta)
    advantages, returns = compute_gae(rewards, values * mask, mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
    raw_adv_mean = float(masked_mean(advantages, mask))
    if cfg.get("normalize_advantages", True):
        advantages = normalize_advantages(advantages, mask)
    return {
        "sequences": sequences, "attention_mask": attention_mask, "prompt_width": prompt_width,
        "response_ids": response_ids, "mask": mask, "old_logp": old_logp, "ref_logp": ref_logp,
        "values": values, "returns": returns, "advantages": advantages, "raw_adv_mean": raw_adv_mean,
    }


@torch.no_grad()
def collect_rollout(bundle, rows, cfg, kl_beta: float, seed: int):
    policy, value_model, tokenizer = bundle["policy"], bundle["value_model"], bundle["tokenizer"]
    messages = [prompt_messages(r) for r in rows]
    set_seed(seed)
    gen = batch_generate(policy, tokenizer, messages, max_prompt_length=int(cfg["max_prompt_length"]),
                         max_new_tokens=int(cfg["max_response_length"]),
                         temperature=float(cfg["generation"]["temperature"]),
                         top_p=float(cfg["generation"]["top_p"]), do_sample=True)
    # batch_generate runs under inference_mode; clone so the tensors can enter autograd in ppo_update.
    gen = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in gen.items()}
    args = (gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"])
    mask = gen["response_mask"]
    old_logp, entropy = token_logprobs_and_entropy(policy, *args)
    with reference_mode(policy):
        ref_logp, _ = token_logprobs_and_entropy(policy, *args, with_entropy=False)
    values = response_values(value_model, gen["sequences"], gen["attention_mask"], gen["prompt_width"], mask.shape[1])

    raw = score_reward_pairs(bundle["reward_model"], bundle["reward_tokenizer"], messages, gen["responses"],
                             max_length=int(cfg["reward_max_length"])).to(mask.device)
    no_eos = torch.tensor([not t for t in gen["terminated_with_eos"]], device=mask.device, dtype=raw.dtype)
    task_reward = raw - float(cfg["missing_eos_penalty"]) * no_eos

    batch = build_ppo_batch(sequences=gen["sequences"], attention_mask=gen["attention_mask"],
                            prompt_width=gen["prompt_width"], response_ids=gen["response_ids"], mask=mask,
                            old_logp=old_logp, ref_logp=ref_logp, values=values, task_reward=task_reward,
                            cfg=cfg, kl_beta=kl_beta)
    batch["stats"] = {
        "reward_raw": float(raw.mean()),
        "reward": float(task_reward.mean()),
        "kl_token_mean": float(sampled_kl(old_logp, ref_logp, mask)),
        "kl_seq_mean": float(((old_logp - ref_logp) * mask).sum(-1).mean()),
        "entropy": float(masked_mean(entropy, mask)),
        "response_tokens": float(mask.sum(-1).mean()),
        "eos_rate": float(np.mean(gen["terminated_with_eos"])),
        "value_mean": float(masked_mean(values, mask)),
        "return_mean": float(masked_mean(batch["returns"], mask)),
        "advantage_mean_prenorm": batch["raw_adv_mean"],
        "critic_explained_variance": explained_variance(values, batch["returns"], mask),
        "value_at_first_token": float(values[:, 0].mean()),
    }
    batch["records"] = [{"prompt_id": r["prompt_id"], "prompt": m[-1]["content"], "response": y,
                         "response_tokens": n, "terminated_with_eos": t, "reward_raw": float(rr),
                         "reward": float(tr)}
                        for r, m, y, n, t, rr, tr in zip(rows, messages, gen["responses"], gen["response_lengths"],
                                                         gen["terminated_with_eos"], raw, task_reward)]
    return batch


def check_finite(what: str, value, batch):
    """Stop the run on the first non-finite loss instead of silently skipping updates."""
    if not torch.isfinite(value):
        raise RuntimeError(
            f"non-finite {what} ({float(value)}); finite values/returns/advantages: "
            f"{bool(torch.isfinite(batch['values']).all())}/{bool(torch.isfinite(batch['returns']).all())}/"
            f"{bool(torch.isfinite(batch['advantages']).all())}")


def ppo_update(bundle, batch, cfg, eps: float):
    """`ppo_epochs` passes over one rollout batch: clipped policy step + critic regression step."""
    policy, value_model = bundle["policy"], bundle["value_model"]
    popt, vopt = bundle["policy_optimizer"], bundle["value_optimizer"]
    max_norm = float(cfg["max_grad_norm"])
    mask, adv = batch["mask"], batch["advantages"]
    args = (batch["sequences"], batch["attention_mask"], batch["prompt_width"], batch["response_ids"])
    per_epoch = []
    for _ in range(int(cfg["ppo_epochs"])):
        new_logp, _ = token_logprobs_and_entropy(policy, *args, with_entropy=False)
        loss, ratio, clip_frac = ppo_policy_loss(new_logp, batch["old_logp"], adv, mask, eps)
        check_finite("policy loss", loss, batch)
        diag = clip_diagnostics(ratio, adv, mask, eps)
        popt.zero_grad(set_to_none=True)
        loss.backward()
        p_gn = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_norm)
        if torch.isfinite(p_gn):
            popt.step()

        values = response_values(value_model, batch["sequences"], batch["attention_mask"],
                                 batch["prompt_width"], mask.shape[1])
        v_loss = value_mse_loss(values, batch["returns"], mask)
        check_finite("value loss", v_loss, batch)
        vopt.zero_grad(set_to_none=True)
        (float(cfg["value_coef"]) * v_loss).backward()
        v_gn = torch.nn.utils.clip_grad_norm_(trainable_parameters(value_model), max_norm)
        if torch.isfinite(v_gn):
            vopt.step()

        valid = ratio[mask.bool()]
        per_epoch.append({"policy_loss": float(loss.detach()), "value_loss": float(v_loss.detach()),
                          "policy_grad_norm": float(p_gn), "value_grad_norm": float(v_gn),
                          "clip_fraction": float(clip_frac), "affected_fraction": diag["affected_fraction"],
                          "ratio_max": float(valid.max()), "ratio_min": float(valid.min()),
                          "nonfinite_step": not (torch.isfinite(p_gn) and torch.isfinite(v_gn))})

    # Size of the policy change produced by this update, measured on the update's own tokens.
    with torch.no_grad():
        post_logp, _ = token_logprobs_and_entropy(policy, *args, with_entropy=False)
        log_r = post_logp - batch["old_logp"]
        step_kl = float(masked_mean(-log_r, mask))                         # k1: E_old[log old - log new]
        step_kl_k3 = float(masked_mean(torch.exp(log_r) - 1 - log_r, mask))  # k3, always >= 0

    out = {k: float(np.mean([e[k] for e in per_epoch])) for k in per_epoch[0] if k != "nonfinite_step"}
    out.update({"last_epoch_clip_fraction": per_epoch[-1]["clip_fraction"],
                "last_epoch_affected_fraction": per_epoch[-1]["affected_fraction"],
                "nonfinite_steps": int(sum(e["nonfinite_step"] for e in per_epoch)),
                "step_kl_old_new": step_kl, "step_kl_old_new_k3": step_kl_k3})
    return out


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    n_updates, eps, beta = int(cfg["updates"]), float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    results_dir = repo_path(cfg["results_dir"]) / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path, rollout_path = results_dir / "train_log.jsonl", results_dir / "rollouts.jsonl"
    for p in (log_path, rollout_path):
        if p.exists():
            p.unlink()

    schedule = prompt_schedule(cfg, bundle["tokenizer"], bundle["prompt_rows"], n_updates)
    print(f"[{run_name}] updates={n_updates} eps={eps} kl_beta={beta} prompts/update={cfg['prompts_per_update']} "
          f"ppo_epochs={cfg['ppo_epochs']} max_response={cfg['max_response_length']}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    elapsed = wall_timer()
    tokens_generated = 0
    for u, rows in enumerate(schedule, start=1):
        # Same sampling seed per update index in every fork -> matched randomness across conditions.
        batch = collect_rollout(bundle, rows, cfg, beta, seed=int(cfg["seed"]) + u)
        tokens_generated += int(batch["mask"].sum())
        stats = ppo_update(bundle, batch, cfg, eps)
        record = {"update": u, "prompt_ids": [r["prompt_id"] for r in rows], **batch["stats"], **stats,
                  "tokens_generated": tokens_generated, "elapsed_s": round(elapsed(), 2)}
        append_jsonl(log_path, record)
        for rec in batch["records"]:
            append_jsonl(rollout_path, {"update": u, **rec})
        print(f"[{run_name}] update {u}/{n_updates} reward={record['reward']:+.3f} kl={record['kl_token_mean']:.4f} "
              f"ent={record['entropy']:.3f} len={record['response_tokens']:.0f} pl={record['policy_loss']:+.4f} "
              f"vl={record['value_loss']:.3f} clip={record['clip_fraction']:.3f} gn={record['policy_grad_norm']:.2f} "
              f"t={record['elapsed_s']:.0f}s", flush=True)
        del batch

    wall_s = elapsed()
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    bundle["policy"].save_pretrained(str(out))
    bundle["tokenizer"].save_pretrained(str(out))
    bundle["value_model"].save_pretrained(str(out) + "_value")
    save_json(results_dir / "train_summary.json", {
        "run_name": run_name, "adapter_path": str(out), "value_adapter_path": str(out) + "_value",
        "start_policy": cfg["paths"]["ppo_midpoint_policy"], "start_value": cfg["paths"]["ppo_midpoint_value"],
        "updates": n_updates, "clip_epsilon": eps, "kl_beta": beta,
        "prompts_per_update": int(cfg["prompts_per_update"]), "ppo_epochs": int(cfg["ppo_epochs"]),
        "policy_learning_rate": float(cfg["policy_learning_rate"]),
        "value_lora_learning_rate": float(cfg["value_lora_learning_rate"]),
        "value_head_learning_rate": float(cfg["value_head_learning_rate"]),
        "gamma": float(cfg["gamma"]), "gae_lambda": float(cfg["gae_lambda"]), "value_coef": float(cfg["value_coef"]),
        "missing_eos_penalty": float(cfg["missing_eos_penalty"]),
        "normalize_advantages": bool(cfg.get("normalize_advantages", True)),
        "max_prompt_length": int(cfg["max_prompt_length"]), "max_response_length": int(cfg["max_response_length"]),
        "generation": cfg["generation"], "seed": int(cfg["seed"]),
        "tokens_generated": tokens_generated, "wall_clock_s": round(wall_s, 1),
        "peak_vram_gib": None if peak is None else round(peak, 3),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "prompt_ids_by_update": [[r["prompt_id"] for r in rows] for rows in schedule],
    })
    print(f"[{run_name}] done in {wall_s / 60:.1f} min, peak VRAM {peak if peak is None else round(peak, 2)} GiB "
          f"-> {out}", flush=True)
    del bundle
    clear_gpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
