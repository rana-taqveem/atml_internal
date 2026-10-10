from __future__ import annotations

import argparse
import time

import numpy as np

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.evidence import generation_fingerprint, validate_cached_generation
from common.logging_utils import save_json, set_seed
from common.models import clear_gpu, load_policy, load_tokenizer
from common.policy_eval import length_stats
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def row_id(row) -> str:
    return str(row.get("prompt_id", row.get("source_index")))


def failure_type(rec) -> str:
    if rec["correct"]:
        return "correct"
    if rec["pred"] is None:
        return "truncated_no_final" if rec["truncated"] else "missing_final_format"
    return "wrong_final"


def generate_policy(cfg, tokenizer, rows, name: str, batch_size: int):
    """Deterministic (greedy) responses, common cap; exact verifier fields per response."""
    model = load_frozen_policy(cfg, name)
    set_seed(int(cfg["seed"]))
    out, t0 = [], time.perf_counter()
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        gen = batch_generate(model, tokenizer, [prompt_messages(r) for r in chunk], max_prompt_length=512,
                             max_new_tokens=int(cfg["math_max_new_tokens"]), temperature=0.0, top_p=1.0,
                             do_sample=False)
        for r, y, n, trunc in zip(chunk, gen["responses"], gen["response_lengths"], gen["truncated"]):
            pred = extract_designated_final(y)
            rec = {"id": row_id(r), "policy": name, "response": y, "response_tokens": n, "truncated": trunc,
                   "pred": pred, "gold": str(r["gold_final"]), "format_ok": pred is not None,
                   "correct": bool(exact_reward(y, str(r["gold_final"])))}
            rec["failure_type"] = failure_type(rec)
            out.append(rec)
        print(f"[{name}] {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    gen_s = time.perf_counter() - t0
    del model
    clear_gpu()
    return out, gen_s


def verifier_label(rec_a, rec_b) -> str:
    """Verifier preference between two responses: A, B, or TIE (both right / both wrong)."""
    if rec_a["correct"] == rec_b["correct"]:
        return "TIE"
    return "A" if rec_a["correct"] else "B"


def judge_against_sft(judge, rows, gens, policy: str):
    """Pairwise judge: policy response (A) vs SFT response (B) for every problem."""
    by_id = {name: {g["id"]: g for g in recs} for name, recs in gens.items()}
    out, judge_s, n_calls = [], 0.0, 0
    for r in rows:
        pid = row_id(r)
        a, b = by_id[policy][pid], by_id["sft"][pid]
        cached = judge._key(r["question"], a["response"], b["response"]) in judge.cache
        t0 = time.perf_counter()
        label = judge.compare(r["question"], a["response"], b["response"])
        detail = judge.details.get(judge._key(r["question"], a["response"], b["response"]), {})
        if not cached:
            judge_s += time.perf_counter() - t0
            n_calls += 1
        out.append({"id": pid, "policy": policy, "judge": label,
                    "judge_parse_ambiguous": detail.get("parse_ambiguous"),
                    "score": {"A": 1.0, "TIE": 0.5, "B": 0.0}[label],
                    "verifier": verifier_label(a, b), "policy_correct": a["correct"], "sft_correct": b["correct"]})
    return out, judge_s, n_calls


def agreement(records) -> dict:
    """Verifier-vs-judge agreement on (policy vs SFT) pairs."""
    labels = ["A", "TIE", "B"]
    conf = {v: {j: 0 for j in labels} for v in labels}
    for r in records:
        conf[r["verifier"]][r["judge"]] += 1
    decisive = [r for r in records if r["verifier"] != "TIE"]
    ties = [r for r in records if r["verifier"] == "TIE"]
    return {
        "exact_label_agreement": float(np.mean([r["verifier"] == r["judge"] for r in records])),
        "n_verifier_decisive": len(decisive),
        "judge_agrees_on_decisive": float(np.mean([r["judge"] == r["verifier"] for r in decisive])) if decisive else float("nan"),
        "judge_prefers_wrong_on_decisive": float(np.mean([r["judge"] not in (r["verifier"], "TIE") for r in decisive])) if decisive else float("nan"),
        "n_verifier_ties": len(ties),
        "judge_tie_rate_on_verifier_ties": float(np.mean([r["judge"] == "TIE" for r in ties])) if ties else float("nan"),
        "confusion_verifier_rows_judge_cols": conf,
    }


def summarize_policy(recs, gen_s) -> dict:
    return {
        "n": len(recs),
        "exact_accuracy": float(np.mean([r["correct"] for r in recs])),
        "format_compliance": float(np.mean([r["format_ok"] for r in recs])),
        "truncation_rate": float(np.mean([r["truncated"] for r in recs])),
        "response_tokens": length_stats([r["response_tokens"] for r in recs]),
        "failure_types": {t: int(sum(r["failure_type"] == t for r in recs))
                          for t in ["correct", "wrong_final", "missing_final_format", "truncated_no_final"]},
        "generation_seconds": round(gen_s, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--limit", type=int, help="first N problems only (smoke tests)")
    ap.add_argument("--skip-existing", action="store_true", help="reuse saved generations")
    args = ap.parse_args()
    cfg, rows, tokenizer = load_math_evaluation(args.config, args.dataset)
    if args.limit:
        rows = rows[: args.limit]
    print("Rows:", len(rows))
    print("Policies:", list(policy_specs(cfg)))
    out_dir = repo_path(cfg["results_dir"]) / "task5_feedback" / args.dataset
    bs = int(cfg.get("math_batch_size", 8))

    gens, gen_time = {}, {}
    for name in policy_specs(cfg):
        path = out_dir / f"generations_{name}.jsonl"
        fingerprint = generation_fingerprint(cfg, policy_specs(cfg)[name], dataset_path(cfg, args.dataset),
                                             max_prompt_length=512, max_new_tokens=int(cfg["math_max_new_tokens"]))
        if args.skip_existing and path.exists():
            gens[name], gen_time[name] = read_jsonl(path), float("nan")
            validate_cached_generation(gens[name], [row_id(r) for r in rows], "id", fingerprint, context=str(path))
            print(f"[{name}] reusing {path}")
        else:
            gens[name], gen_time[name] = generate_policy(cfg, tokenizer, rows, name, bs)
            for rec in gens[name]:
                rec["generation_fingerprint"] = fingerprint
            write_jsonl(path, gens[name])

    judge = PairwiseAIJudge(cfg, repo_path(cfg["results_dir"]) / "task5_feedback" / "judge_cache.json")
    metrics = {"dataset": args.dataset, "n_problems": len(rows), "decoding": "greedy",
               "max_new_tokens": int(cfg["math_max_new_tokens"]), "judge_model": cfg["ai_judge_model"],
               "policies": {}}
    for name, recs in gens.items():
        metrics["policies"][name] = summarize_policy(recs, gen_time[name])
    metrics["policies"]["sft"].update({"ai_pairwise_win_rate_vs_sft": 0.5,
                                        "pairwise_baseline_note": "self-comparison tie by definition; no judge calls"})
    for name in ["rlvr", "rlaif"]:
        jrecs, judge_s, n_calls = judge_against_sft(judge, rows, gens, name)
        write_jsonl(out_dir / f"judge_{name}_vs_sft.jsonl", jrecs)
        counts = {k: int(sum(r["judge"] == k for r in jrecs)) for k in ["A", "TIE", "B"]}
        metrics["policies"][name].update({
            "ai_pairwise_win_rate_vs_sft": float(np.mean([r["score"] for r in jrecs])),
            "judge_counts_policy_win_tie_loss": [counts["A"], counts["TIE"], counts["B"]],
            "judge_parse_ambiguous_count": sum(r["judge_parse_ambiguous"] is True for r in jrecs),
            "judge_parse_status_unknown_count": sum(r["judge_parse_ambiguous"] is None for r in jrecs),
            "verifier_judge_agreement": agreement(jrecs),
            "judge_seconds_per_new_call": (judge_s / n_calls) if n_calls else None,
        })
    del judge
    clear_gpu()
    save_json(out_dir / "metrics.json", metrics)

    print(f"\n{'policy':<7}{'exact acc':>10}{'format':>8}{'tokens':>9}{'trunc':>7}{'win vs SFT':>12}{'W/T/L':>12}{'judge=verifier (decisive)':>27}")
    for name, m in metrics["policies"].items():
        w = m.get("ai_pairwise_win_rate_vs_sft", float("nan"))
        wtl = "/".join(map(str, m.get("judge_counts_policy_win_tie_loss", ["-"] * 3)))
        agr = m.get("verifier_judge_agreement", {}).get("judge_agrees_on_decisive", float("nan"))
        print(f"{name:<7}{m['exact_accuracy']:>10.3f}{m['format_compliance']:>8.3f}{m['response_tokens']['mean']:>9.1f}"
              f"{m['truncation_rate']:>7.2f}{w:>12.3f}{wtl:>12}{agr:>27.3f}")
    print(f"Saved {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
