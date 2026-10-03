from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from common.data import load_yaml, prompt_messages, prompt_messages_from_preference, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import load_json, save_json, set_seed, wall_timer
from common.metrics import parse_word_limit, sampled_kl, sample_entropy, word_count
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.train import make_collate, pair_logprobs, to_device


def load_evaluation_bundle(config_path: str, adapter: str | None):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def length_stats(values) -> dict:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {}
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {"mean": float(a.mean()), "std": float(a.std(ddof=0)), "median": float(med),
            "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1), "n": int(a.size)}


@torch.no_grad()
def evaluate_pairs(model, tokenizer, rows, beta: float, max_length: int, batch_size: int, has_adapter: bool):
    """Held-out DPO margin m, loss at `beta`, and preference accuracy (m > 0) for every pair."""
    collate = make_collate(tokenizer, max_length)
    device = next(model.parameters()).device
    records = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        batch = to_device(collate(chunk), device)
        n = len(chunk)
        pol_c, pol_r, mask = pair_logprobs(model, batch, n)
        if has_adapter:
            with reference_mode(model):
                ref_c, ref_r, _ = pair_logprobs(model, batch, n)
        else:
            ref_c, ref_r = pol_c, pol_r  # the untouched base policy *is* the reference
        margin = (pol_c - pol_r) - (ref_c - ref_r)
        loss = -F.logsigmoid(beta * margin)
        tokens = mask.sum(-1)
        for i, row in enumerate(chunk):
            records.append({
                "prompt_id": row["prompt_id"],
                "length_stratum": row.get("length_stratum"),
                "chosen_tokens": int(tokens[i]),
                "rejected_tokens": int(tokens[n + i]),
                "policy_chosen_logp": float(pol_c[i]),
                "policy_rejected_logp": float(pol_r[i]),
                "ref_chosen_logp": float(ref_c[i]),
                "ref_rejected_logp": float(ref_r[i]),
                "margin": float(margin[i]),
                "loss": float(loss[i]),
                "correct": bool(margin[i] > 0),
            })
    return records


def summarize_pairs(records) -> dict:
    out = {
        "n_pairs": len(records),
        "dpo_loss": float(np.mean([r["loss"] for r in records])),
        "preference_accuracy": float(np.mean([r["correct"] for r in records])),
        "margin_mean": float(np.mean([r["margin"] for r in records])),
        "margin_std": float(np.std([r["margin"] for r in records])),
        "chosen_logratio_mean": float(np.mean([r["policy_chosen_logp"] - r["ref_chosen_logp"] for r in records])),
        "rejected_logratio_mean": float(np.mean([r["policy_rejected_logp"] - r["ref_rejected_logp"] for r in records])),
    }
    strata = sorted({r["length_stratum"] for r in records if r["length_stratum"]})
    if strata:
        out["by_stratum"] = {s: summarize_pairs([dict(r, length_stratum=None) for r in records if r["length_stratum"] == s])
                             for s in strata}
    return out


def select_generation_prompts(tokenizer, rows, n: int, max_prompt_length: int):
    """First `n` held-out prompts whose rendered chat prompt fits the generation context.

    `batch_generate` truncates on the right, which would cut the assistant header off long prompts,
    so over-long prompts are skipped (deterministically) rather than truncated.
    """
    picked = []
    for row in rows:
        messages = prompt_messages_from_preference(row)
        n_tok = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        if n_tok <= max_prompt_length:
            picked.append((row["prompt_id"], messages))
        if len(picked) == n:
            break
    return picked


@torch.no_grad()
def generate_and_score(model, tokenizer, reward, prompts, cfg, has_adapter: bool, *, seed: int,
                       max_new_tokens: int, max_prompt_length: int, batch_size: int, do_sample: bool):
    """Generate one response per prompt, then score it with the RM and the policy/reference log-probs."""
    rm_model, rm_tok = reward
    gen_cfg = cfg["generation"]
    set_seed(seed)
    records = []
    all_pol, all_ref, all_mask = [], [], []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        messages = [m for _, m in chunk]
        gen = batch_generate(model, tokenizer, messages, max_prompt_length=max_prompt_length,
                             max_new_tokens=max_new_tokens, temperature=float(gen_cfg["temperature"]),
                             top_p=float(gen_cfg["top_p"]), do_sample=do_sample)
        pol_logp, _ = response_token_logprobs(model, gen["sequences"], gen["attention_mask"],
                                              gen["prompt_width"], gen["response_ids"])
        if has_adapter:
            with reference_mode(model):
                ref_logp, _ = response_token_logprobs(model, gen["sequences"], gen["attention_mask"],
                                                      gen["prompt_width"], gen["response_ids"])
        else:
            ref_logp = pol_logp
        mask = gen["response_mask"]
        diff = (pol_logp - ref_logp) * mask
        rewards = score_reward_pairs(rm_model, rm_tok, messages, gen["responses"])

        all_pol.append(pol_logp.flatten().cpu())
        all_ref.append(ref_logp.flatten().cpu())
        all_mask.append(mask.flatten().cpu())
        for i, (pid, msg) in enumerate(chunk):
            n_tok = int(mask[i].sum())
            records.append({
                "prompt_id": pid,
                "prompt": msg[-1]["content"],
                "response": gen["responses"][i],
                "response_tokens": gen["response_lengths"][i],
                "response_words": word_count(gen["responses"][i]),
                "truncated": gen["truncated"][i],
                "reward": float(rewards[i]),
                "kl_seq": float(diff[i].sum()),
                "kl_token_mean": float(diff[i].sum() / max(n_tok, 1)),
            })
    flat_pol, flat_ref, flat_mask = torch.cat(all_pol), torch.cat(all_ref), torch.cat(all_mask)
    agg = {
        # Course helper: masked mean over all valid response tokens of the evaluation set.
        "kl_token_mean": float(sampled_kl(flat_pol, flat_ref, flat_mask)),
        # Same estimator summed per response, then averaged over responses.
        "kl_seq_mean": float(np.mean([r["kl_seq"] for r in records])),
        "entropy_token_mean": float(sample_entropy(flat_pol, flat_mask)),
    }
    return records, agg


def summarize_generations(records, agg) -> dict:
    rewards = [r["reward"] for r in records]
    out = {
        "n_prompts": len(records),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "response_tokens": length_stats([r["response_tokens"] for r in records]),
        "response_words": length_stats([r["response_words"] for r in records]),
        "truncation_rate": float(np.mean([r["truncated"] for r in records])),
    }
    out.update(agg)
    return out


def evaluate_word_limits(model, tokenizer, reward, cfg, has_adapter, *, seed, n_samples, max_new_tokens, batch_size):
    """Greedy (deterministic) + `n_samples` sampled responses per word-limit prompt."""
    rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
    prompts = [(r["prompt_id"], prompt_messages(r)) for r in rows]
    runs = {"greedy": (prompts, False)}
    if n_samples > 0:
        runs["sampled"] = ([p for p in prompts for _ in range(n_samples)], True)

    records, summary = [], {}
    for mode, (plist, do_sample) in runs.items():
        recs, _ = generate_and_score(model, tokenizer, reward, plist, cfg, has_adapter, seed=seed,
                                     max_new_tokens=max_new_tokens, max_prompt_length=256,
                                     batch_size=batch_size, do_sample=do_sample)
        for r in recs:
            limit = parse_word_limit(r["prompt"])
            r.update({"decoding": mode, "word_limit": limit,
                      "compliant": None if limit is None else bool(r["response_words"] <= limit),
                      "words_over_limit": None if limit is None else max(0, r["response_words"] - limit),
                      "words_to_limit_ratio": None if limit is None else r["response_words"] / limit})
        scored = [r for r in recs if r["word_limit"] is not None]
        summary[mode] = {
            "n_responses": len(scored),
            "compliance_rate": float(np.mean([r["compliant"] for r in scored])),
            "words": length_stats([r["response_words"] for r in scored]),
            "words_to_limit_ratio_mean": float(np.mean([r["words_to_limit_ratio"] for r in scored])),
            "response_tokens": length_stats([r["response_tokens"] for r in scored]),
            "reward_mean": float(np.mean([r["reward"] for r in scored])),
        }
        records.extend(recs)
    return records, summary


def evaluate_policy(config_path: str, adapter: str | None, name: str, beta: float | None = None,
                    n_gen_prompts: int | None = None, skip_generation: bool = False):
    cfg = load_yaml(config_path)
    ecfg = cfg.get("eval", {})
    out_dir = repo_path(cfg["results_dir"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    has_adapter = adapter is not None
    if beta is None:
        summary_path = out_dir / "train_summary.json"
        beta = float(load_json(summary_path)["beta"]) if summary_path.exists() else float(cfg["beta"])

    elapsed = wall_timer()
    bundle = load_evaluation_bundle(config_path, adapter)
    model, tokenizer, reward = bundle["policy"], bundle["tokenizer"], bundle["reward"]
    seed = int(cfg["seed"])
    metrics = {"name": name, "adapter": adapter, "beta_for_loss": beta, "seed": seed}

    # 1) Held-out preference pairs: standard eval split and length-stratified split.
    pair_bs = int(ecfg.get("pair_batch_size", 4))
    max_len = int(cfg["max_sequence_length"])
    std_pairs = evaluate_pairs(model, tokenizer, bundle["rows"], beta, max_len, pair_bs, has_adapter)
    strat_rows = read_jsonl(cfg["paths"]["dpo_length_eval"])
    strat_pairs = evaluate_pairs(model, tokenizer, strat_rows, beta, max_len, pair_bs, has_adapter)
    write_jsonl(out_dir / "eval_pairs_standard.jsonl", std_pairs)
    write_jsonl(out_dir / "eval_pairs_length_stratified.jsonl", strat_pairs)
    metrics["pairs_standard"] = summarize_pairs(std_pairs)
    metrics["pairs_length_stratified"] = summarize_pairs(strat_pairs)
    print(f"[{name}] held-out acc={metrics['pairs_standard']['preference_accuracy']:.3f} "
          f"loss={metrics['pairs_standard']['dpo_loss']:.4f}", flush=True)

    if not skip_generation:
        gen_bs = int(ecfg.get("generation_batch_size", 8))
        max_new = int(cfg["max_generation_tokens"])
        max_prompt = int(ecfg.get("max_prompt_tokens", 512))
        n_gen = int(n_gen_prompts or ecfg.get("generation_prompts", 128))

        # 2) Sampled generations on held-out prompts: reward, KL, entropy, length.
        prompts = select_generation_prompts(tokenizer, bundle["rows"], n_gen, max_prompt)
        gens, agg = generate_and_score(model, tokenizer, reward, prompts, cfg, has_adapter, seed=seed,
                                       max_new_tokens=max_new, max_prompt_length=max_prompt,
                                       batch_size=gen_bs, do_sample=bool(cfg["generation"]["do_sample"]))
        write_jsonl(out_dir / "generations.jsonl", gens)
        metrics["generation"] = summarize_generations(gens, agg)
        metrics["generation"]["decoding"] = {**cfg["generation"], "max_new_tokens": max_new,
                                             "max_prompt_tokens": max_prompt, "seed": seed}
        g = metrics["generation"]
        print(f"[{name}] reward={g['reward_mean']:.3f} kl_tok={g['kl_token_mean']:.4f} "
              f"len={g['response_tokens']['mean']:.1f}", flush=True)

        # 3) Explicit word-limit compliance on the common prompt set.
        wl, wl_summary = evaluate_word_limits(model, tokenizer, reward, cfg, has_adapter, seed=seed,
                                              n_samples=int(ecfg.get("word_limit_samples", 5)),
                                              max_new_tokens=max_new, batch_size=gen_bs)
        write_jsonl(out_dir / "word_limit.jsonl", wl)
        metrics["word_limit"] = wl_summary
        print(f"[{name}] word-limit compliance greedy={wl_summary['greedy']['compliance_rate']:.2f}", flush=True)

    metrics["eval_wall_clock_s"] = round(elapsed(), 1)
    save_json(out_dir / "eval_metrics.json", metrics)
    del model, reward, bundle
    clear_gpu()
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter directory, or 'none' for the untouched SFT/reference policy")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float, help="beta used for the held-out DPO loss (default: the run's training beta)")
    ap.add_argument("--n-gen-prompts", type=int)
    ap.add_argument("--skip-generation", action="store_true")
    args = ap.parse_args()
    adapter = None if args.adapter.lower() == "none" else args.adapter
    evaluate_policy(args.config, adapter, args.name, args.beta, args.n_gen_prompts, args.skip_generation)


if __name__ == "__main__":
    main()
