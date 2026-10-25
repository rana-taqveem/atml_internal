"""Task 1 evaluation of one condition with lm-evaluation-harness (zero-shot, FP16).

GSM8K: gsm8k_cot, num_fewshot=0, non-thinking chat template, greedy, 1,024-token cap,
stock stop rules/filters (flexible-extract is the primary metric). HellaSwag/BoolQ/ARC-Easy:
candidate likelihood with stock completion prompts, no chat template.

Adapter conditions are merged into a fresh FP32 backbone copy (OPLoRA: projected factors),
cast to FP16, and a merged-vs-unmerged logit check is recorded.

    python -m task1_finetune.evaluate --condition lora16 [--limit 20]
"""
from __future__ import annotations

import argparse
import gc
import json
import time

import torch

from common.store import Store
from common.utils import (ensure_dir, env_info, gpu_memory, load_config, read_json, rel,
                          write_json)
from task1_finetune import adapters as ad
from task1_finetune.data import format_prompt, get_tokenizer


def eval_dir(cond, smoke=False):
    return f"results/task1/eval/{cond}{'_smoke' if smoke else ''}"


def weights_dir(cfg, variant, stage):
    return f"outputs/task1/{variant}{cfg.get('run_suffix', '')}/weights_{stage}"


@torch.no_grad()
def merge_check(cfg, variant, raw, eff, merged_dir, tok, device):
    """Max |logit| difference between the unmerged adapter model (raw factors, OPLoRA
    projection applied on the fly) and the merged model, on 4 prompts.
    fp32: in-memory FP32 merge (checks the merge itself);
    fp16: the saved FP16 merged model used for evaluation vs unmerged FP16 (GPU only)."""
    from transformers import AutoModelForCausalLM

    prompts = ["What is 12 * 7?", "Name three primary colours.",
               "Solve for x: 2x + 3 = 11.", "Write one sentence about the sea."]
    batch = tok([format_prompt(tok, p) for p in prompts], return_tensors="pt", padding=True)
    batch = {k: v.to(device) for k, v in batch.items()}
    vcfg = cfg["variants"][variant]
    out = {}
    for dtype, name in [(torch.float32, "fp32"), (torch.float16, "fp16")]:
        if device == "cpu" and dtype == torch.float16:
            continue
        unm = AutoModelForCausalLM.from_pretrained(cfg["model_id"], dtype=dtype).to(device).eval()
        ad.attach_adapters(unm, cfg["lora"]["targets"], vcfg["r"], vcfg["alpha"], 0.0,
                           vcfg.get("k"), cfg["seed"])
        ad.load_adapter_state(unm, raw)
        if dtype == torch.float32:
            mer = AutoModelForCausalLM.from_pretrained(cfg["model_id"], dtype=dtype)
            ad.merge_effective(mer, eff)
            mer = mer.to(device).eval()
        else:
            mer = AutoModelForCausalLM.from_pretrained(merged_dir, dtype=dtype).to(device).eval()
        mask = batch["attention_mask"].bool()
        a = unm(**batch).logits.float()[mask]
        b = mer(**batch).logits.float()[mask]
        out[name] = {"max_abs_diff": float((a - b).abs().max()),
                     "mean_abs_diff": float((a - b).abs().mean()),
                     "argmax_agreement": float((a.argmax(-1) == b.argmax(-1)).float().mean())}
        del unm, mer
        gc.collect()
    return out


def materialize(cfg, cond, store, device):
    """Return a loadable HF directory for the condition (plus merge-check info)."""
    c = cfg["eval"]["conditions"][cond]
    if c["variant"] is None:
        return cfg["model_id"], {}
    variant, stage = c["variant"], c["stage"]
    wdir = weights_dir(cfg, variant, stage)
    if not rel(wdir).exists():
        assert store.pull(wdir), f"missing {wdir} (train {variant} first)"
    if cfg["variants"][variant]["kind"] == "full":
        return str(rel(wdir)), {}
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    tok = get_tokenizer(cfg["model_id"])
    eff = load_file(str(rel(wdir) / "adapter_effective.safetensors"))
    mdir = ensure_dir(f"outputs/task1/merged/{cond}")
    if not (mdir / "config.json").exists():
        model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], dtype=torch.float32)
        n = ad.merge_effective(model, eff)
        model.to(torch.float16).save_pretrained(mdir, safe_serialization=True)
        tok.save_pretrained(mdir)
        del model
    info = {"merged_modules": len({k.rsplit('.', 1)[0] for k in eff}),
            "merge_check": merge_check(cfg, variant, torch.load(rel(wdir) / "adapter_raw.pt"),
                                       eff, str(mdir), tok, device)}
    return str(mdir), info


def _gsm_rows(samples, tok, cap):
    """Collapse lm-eval samples (one entry per filter) into one row per question."""
    rows = {}
    for s in samples:
        r = rows.setdefault(s["doc_id"], {
            "doc_id": s["doc_id"], "question": s["doc"]["question"], "target": s["target"],
            "response": s["resps"][0][0] if s["resps"] else ""})
        f = s.get("filter", "none")
        r[f"extracted_{f}"] = s["filtered_resps"][0] if s["filtered_resps"] else None
        r[f"correct_{f}"] = float(s.get("exact_match", 0.0))
    for r in rows.values():
        n = len(tok(r["response"], add_special_tokens=False)["input_ids"])
        r["n_tokens"] = n
        r["truncated"] = n >= cap
        r["extractable_flexible"] = r.get("extracted_flexible-extract") not in (None, "", "[invalid]")
    return sorted(rows.values(), key=lambda r: r["doc_id"])


def _ll_rows(samples):
    rows = []
    for s in samples:
        rows.append({"doc_id": s["doc_id"], "target": s["target"],
                     **{m: float(s[m]) for m in ("acc", "acc_norm") if m in s},
                     "loglik": [float(x[0]) for x in s["filtered_resps"]]})
    return sorted(rows, key=lambda r: r["doc_id"])


def run_lm_eval(model_dir, ecfg, limit, gen: bool, batch_size):
    import lm_eval

    margs = {"pretrained": model_dir, "dtype": "float16", "max_length": ecfg["max_length"]}
    kw = dict(model="hf", num_fewshot=0, batch_size=batch_size, limit=limit,
              log_samples=True, bootstrap_iters=0, device="cuda" if torch.cuda.is_available() else "cpu")
    if gen:
        margs["enable_thinking"] = False
        return lm_eval.simple_evaluate(model_args=margs, tasks=[ecfg["gsm8k_task"]],
                                       apply_chat_template=True,
                                       gen_kwargs={"max_gen_toks": ecfg["gsm8k_max_gen_toks"],
                                                   "do_sample": False}, **kw)
    return lm_eval.simple_evaluate(model_args=margs, tasks=ecfg["likelihood_tasks"], **kw)


def evaluate(cfg, cond, limit=None, store=None, parts=("gsm8k", "ll")):
    store = store or Store()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smoke = limit is not None
    out = ensure_dir(eval_dir(cond, smoke))
    store.pull(eval_dir(cond, smoke))
    if (out / "scores.json").exists() and not smoke:
        print(f"[eval {cond}] already done"); return read_json(out / "scores.json")
    ecfg = cfg["eval"]
    tok = get_tokenizer(cfg["model_id"])
    model_dir, info = materialize(cfg, cond, store, device)
    scores = {"condition": cond, **cfg["eval"]["conditions"][cond], **info, "limit": limit}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if "gsm8k" in parts:
        t0 = time.perf_counter()
        res = run_lm_eval(model_dir, ecfg, limit, True, ecfg["batch_size_gen"])
        task = ecfg["gsm8k_task"]
        rows = _gsm_rows(res["samples"][task], tok, ecfg["gsm8k_max_gen_toks"])
        with open(out / "samples_gsm8k.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        m = res["results"][task]
        scores["gsm8k"] = {
            "flexible_extract": m.get("exact_match,flexible-extract"),
            "strict_match": m.get("exact_match,strict-match"),
            "n": len(rows),
            "truncation_rate": sum(r["truncated"] for r in rows) / max(1, len(rows)),
            "extractable_rate": sum(r["extractable_flexible"] for r in rows) / max(1, len(rows)),
            "mean_response_tokens": sum(r["n_tokens"] for r in rows) / max(1, len(rows)),
            "seconds": time.perf_counter() - t0}
        del res; gc.collect()

    if "ll" in parts:
        t0 = time.perf_counter()
        res = run_lm_eval(model_dir, ecfg, limit, False, ecfg["batch_size_ll"])
        for task in ecfg["likelihood_tasks"]:
            m = res["results"][task]
            rows = _ll_rows(res["samples"][task])
            with open(out / f"samples_{task}.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            scores[task] = {"acc": m.get("acc,none"), "acc_norm": m.get("acc_norm,none"),
                            "n": len(rows)}
        scores["likelihood_seconds"] = time.perf_counter() - t0

    scores.update({"gpu": gpu_memory(), "env": env_info()})
    write_json(scores, out / "scores.json")
    store.push(eval_dir(cond, smoke), f"eval {cond}")
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only")
    ap.add_argument("--parts", default="gsm8k,ll")
    args = ap.parse_args()
    cfg = load_config(args.config)
    s = evaluate(cfg, args.condition, args.limit, parts=tuple(args.parts.split(",")))
    print(json.dumps({k: v for k, v in s.items() if k != "env"}, indent=2, default=str))


if __name__ == "__main__":
    main()
