"""Held-out protocol shared by the online-RL tasks (PPO, GRPO).

Same prompts, seed, decoding settings, generation cap, reward model and KL estimator for every
condition, so conditions differ only in the policy adapter.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from common.data import prompt_messages, read_jsonl
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import set_seed
from common.metrics import sampled_kl
from common.models import reference_mode


def length_stats(values) -> dict:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {}
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {"mean": float(a.mean()), "std": float(a.std()), "median": float(med),
            "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1), "n": int(a.size)}


def fitting_prompts(tokenizer, rows, max_prompt_length: int, n: int | None = None):
    """Rows (in file order) whose rendered chat prompt fits `max_prompt_length` tokens.

    `batch_generate` truncates on the right, which would cut the assistant header, so over-long
    prompts are skipped deterministically instead.
    """
    out = []
    for row in rows:
        ids = tokenizer.apply_chat_template(prompt_messages(row), tokenize=True, add_generation_prompt=True)
        if len(ids) <= max_prompt_length:
            out.append(row)
            if n is not None and len(out) == n:
                break
    return out


def token_logprobs_and_entropy(model, sequences, attention_mask, prompt_width, response_ids, chunk: int = 1,
                               with_entropy: bool = True):
    """Sampled-token log-probs and exact per-token entropy over response positions.

    Processes `chunk` sequences at a time so full-vocabulary float logits stay small.
    """
    logps, ents = [], []
    for i in range(0, sequences.shape[0], chunk):
        out = model(input_ids=sequences[i:i + chunk], attention_mask=attention_mask[i:i + chunk],
                    use_cache=False, return_dict=True)
        logits = out.logits[:, prompt_width - 1:-1, :][:, :response_ids.shape[1], :].float()
        logp_all = F.log_softmax(logits, dim=-1)
        logps.append(torch.gather(logp_all, -1, response_ids[i:i + chunk].unsqueeze(-1)).squeeze(-1))
        if with_entropy:
            ents.append(-(logp_all.exp() * logp_all).sum(-1))
        del out, logits, logp_all
    return torch.cat(logps), (torch.cat(ents) if with_entropy else None)


@torch.no_grad()
def evaluate_heldout(model, tokenizer, reward, cfg, prompts, *, has_adapter: bool, max_new_tokens: int,
                     max_prompt_length: int, batch_size: int, seed: int, missing_eos_penalty: float = 0.0,
                     reward_max_length: int = 1024):
    """One seeded sample per prompt; returns (per-response records, aggregate metrics)."""
    rm_model, rm_tok = reward
    gen_cfg = cfg["generation"]
    set_seed(seed)
    records, flat_pol, flat_ref, flat_ent, flat_mask = [], [], [], [], []
    for start in range(0, len(prompts), batch_size):
        rows = prompts[start:start + batch_size]
        messages = [prompt_messages(r) for r in rows]
        gen = batch_generate(model, tokenizer, messages, max_prompt_length=max_prompt_length,
                             max_new_tokens=max_new_tokens, temperature=float(gen_cfg["temperature"]),
                             top_p=float(gen_cfg["top_p"]), do_sample=bool(gen_cfg["do_sample"]))
        args = (gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"])
        pol_logp, ent = token_logprobs_and_entropy(model, *args)
        if has_adapter:
            with reference_mode(model):
                ref_logp, _ = token_logprobs_and_entropy(model, *args, with_entropy=False)
        else:
            ref_logp = pol_logp
        mask = gen["response_mask"]
        raw = score_reward_pairs(rm_model, rm_tok, messages, gen["responses"], max_length=reward_max_length)
        for i, row in enumerate(rows):
            n_tok = max(int(mask[i].sum()), 1)
            diff = ((pol_logp[i] - ref_logp[i]) * mask[i]).sum()
            terminated = gen["terminated_with_eos"][i]
            records.append({
                "prompt_id": row["prompt_id"],
                "prompt": messages[i][-1]["content"],
                "response": gen["responses"][i],
                "response_tokens": gen["response_lengths"][i],
                "terminated_with_eos": terminated,
                "truncated": gen["truncated"][i],
                "reward_raw": float(raw[i]),
                "reward": float(raw[i]) - (0.0 if terminated else missing_eos_penalty),
                "kl_seq": float(diff),
                "kl_token_mean": float(diff / n_tok),
                "entropy_token_mean": float((ent[i] * mask[i]).sum() / n_tok),
            })
        flat_pol.append(pol_logp.flatten().cpu())
        flat_ref.append(ref_logp.flatten().cpu())
        flat_ent.append(ent.flatten().cpu())
        flat_mask.append(mask.flatten().cpu())

    pol, ref, ent, msk = (torch.cat(x) for x in (flat_pol, flat_ref, flat_ent, flat_mask))
    rewards = np.array([r["reward"] for r in records])
    metrics = {
        "n_prompts": len(records),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_se": float(rewards.std() / np.sqrt(len(rewards))),
        "reward_raw_mean": float(np.mean([r["reward_raw"] for r in records])),
        # Course helper (token-weighted mean over all valid response tokens) + per-sequence sum.
        "kl_token_mean": float(sampled_kl(pol, ref, msk)),
        "kl_seq_mean": float(np.mean([r["kl_seq"] for r in records])),
        "entropy_token_mean": float((ent * msk).sum() / msk.sum().clamp_min(1)),
        "response_tokens": length_stats([r["response_tokens"] for r in records]),
        "eos_rate": float(np.mean([r["terminated_with_eos"] for r in records])),
        "truncation_rate": float(np.mean([r["truncated"] for r in records])),
        "decoding": {**gen_cfg, "max_new_tokens": max_new_tokens, "max_prompt_tokens": max_prompt_length,
                     "seed": seed, "missing_eos_penalty": missing_eos_penalty},
    }
    return records, metrics


def heldout_prompts(cfg, tokenizer, n: int, max_prompt_length: int):
    return fitting_prompts(tokenizer, read_jsonl(cfg["paths"]["rl_prompt_eval"]), max_prompt_length, n)
