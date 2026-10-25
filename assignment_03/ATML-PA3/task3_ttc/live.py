"""Task 3 Step 4: parallel sampling vs sequential refinement vs one longer response (GPU).

Fixed 50-question MATH subset (first 50 held-out IDs). Qwen2.5-3B-Instruct (pinned revision),
FP16, batch size 1, temperature 0.7 / top-p 0.8 / top-k 20 / repetition penalty 1.0, native
chat template with the handout's system message and instruction. Per question, same
3,072-token output allowance:
  single      one response capped at 3,072 new tokens
  parallel    three independent responses capped at 1,024; self-consistency answer
  sequential  parallel response #1 as the initial answer, then one critique-and-revision
              response capped at 2,048; the revised answer is returned
Every call is independently seeded. Records prompts, responses, seeds, input/output tokens,
truncation, extraction, wall-clock generation time (model loading excluded) and peak memory.
Resumable per question; each finished question is appended to the JSONL and pushed to the store.

    python -m task3_ttc.live [--limit 2]
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd
import torch

from common.store import Store
from common.utils import append_jsonl, ensure_dir, env_info, load_config, read_jsonl, rel, write_json
from task3_ttc.data import download, splits
from task3_ttc.grading import NO_ANSWER, _equiv, group_math
from task3_ttc.selection import vote

OUT = "results/task3/live"


def last_boxed(text: str):
    """Content of the last literal \\boxed{...} with balanced braces; None if absent/empty/unclosed."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth = start + len("\\boxed{"), 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                ans = text[i:j].strip()
                return ans or None
    return None


class Generator:
    def __init__(self, lcfg, device="cuda" if torch.cuda.is_available() else "cpu"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = lcfg
        self.tok = AutoTokenizer.from_pretrained(lcfg["model_id"], revision=lcfg["revision"])
        self.model = AutoModelForCausalLM.from_pretrained(
            lcfg["model_id"], revision=lcfg["revision"],
            dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
        self.device = device
        self.eos = [self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id]

    @torch.no_grad()
    def __call__(self, messages, cap, seed):
        c = self.cfg
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = self.model.generate(**ids, max_new_tokens=cap, do_sample=True,
                                  temperature=c["temperature"], top_p=c["top_p"], top_k=c["top_k"],
                                  repetition_penalty=c["repetition_penalty"],
                                  eos_token_id=self.eos, pad_token_id=self.tok.pad_token_id)
        if self.device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        new = out[0, ids["input_ids"].shape[1]:].tolist()
        stopped = bool(new) and new[-1] in self.eos
        gen = new[:-1] if stopped else new
        text = self.tok.decode(gen, skip_special_tokens=True)
        return {"prompt": prompt, "response": text, "seed": seed, "cap": cap,
                "input_tokens": int(ids["input_ids"].shape[1]), "output_tokens": len(new),
                "truncated": (not stopped) and len(new) >= cap, "seconds": dt,
                "peak_gb": torch.cuda.max_memory_allocated() / 1024**3 if self.device == "cuda" else None,
                "extracted": last_boxed(text)}


def run_question(gen, lcfg, qi, q, problem, gold, seed0):
    user = f"{problem}\n\n{lcfg['suffix']}"
    base = [{"role": "system", "content": lcfg["system"]}, {"role": "user", "content": user}]
    seed = lambda k: seed0 * 1000 + qi * 10 + k  # independent seed per call
    single = gen(base, lcfg["single_cap"], seed(0))
    parallel = [gen(base, lcfg["parallel_cap"], seed(1 + j)) for j in range(lcfg["n_parallel"])]
    initial = parallel[0]
    revise_msgs = base + [{"role": "assistant", "content": initial["response"]},
                          {"role": "user", "content": lcfg["revise"]}]
    revision = gen(revise_msgs, lcfg["revision_cap"], seed(9))

    def correct(a):
        return a is not None and _equiv((gold or "").strip(), a.strip())

    keys, _, _ = group_math([r["extracted"] for r in parallel], gold)
    sc_key, _ = vote(keys, [1.0] * len(keys))
    par_answer = None if sc_key == NO_ANSWER else sc_key
    for r in [single, *parallel, revision]:
        r["correct"] = correct(r["extracted"])
    return {"qi": qi, "sample_id": q, "gold": gold, "single": single, "parallel": parallel,
            "revision": revision,
            "answers": {"single": single["extracted"], "parallel_sc": par_answer,
                        "initial": initial["extracted"], "sequential": revision["extracted"]},
            "correct": {"single": single["correct"], "parallel_sc": correct(par_answer),
                        "initial": initial["correct"], "sequential": revision["correct"]}}


def summarize(records) -> pd.DataFrame:
    rows = []
    for r in records:
        par, s, init, rev = r["parallel"], r["single"], r["parallel"][0], r["revision"]
        strat = {
            "single": [s],
            "parallel": par,
            # sequential is charged the shared initial response plus the revision call
            "sequential": [init, rev],
        }
        for name, calls in strat.items():
            ans_key = {"single": "single", "parallel": "parallel_sc", "sequential": "sequential"}[name]
            rows.append({"sample_id": r["sample_id"], "strategy": name,
                         "correct": r["correct"][ans_key],
                         "extracted": r["answers"][ans_key] is not None,
                         "input_tokens": sum(c["input_tokens"] for c in calls),
                         "output_tokens": sum(c["output_tokens"] for c in calls),
                         "seconds": sum(c["seconds"] for c in calls),
                         "any_truncated": any(c["truncated"] for c in calls),
                         "final_truncated": calls[-1]["truncated"] if name != "parallel" else all(c["truncated"] for c in calls),
                         "peak_gb": max(c["peak_gb"] or 0 for c in calls)})
    return pd.DataFrame(rows)


def transitions(records) -> dict:
    t = {"wrong_to_correct": 0, "correct_to_wrong": 0, "correct_kept": 0, "wrong_kept": 0,
         "answer_changed": 0}
    for r in records:
        a, b = r["correct"]["initial"], r["correct"]["sequential"]
        key = ("correct" if a else "wrong") + ("_kept" if a == b else ("_to_wrong" if a else "_to_correct"))
        t[key] += 1
        t["answer_changed"] += int(r["answers"]["initial"] != r["answers"]["sequential"])
    return t


def run(cfg, limit=None):
    store = Store()
    out_rel = cfg["live"].get("out_dir", OUT)
    out = ensure_dir(out_rel)
    store.pull(out_rel)
    download(cfg)
    lcfg = cfg["live"]
    ids = splits(cfg)["math"]["live"][: limit or None]
    gen_df = pd.read_parquet(rel(cfg["data"]["raw_dir"]) / cfg["data"]["datasets"]["math"] / "generations.parquet",
                             columns=["sample_id", "rollout_id", "question_text", "gold_answer"])
    qinfo = gen_df[gen_df["rollout_id"] == 0].set_index("sample_id")
    path = out / "generations.jsonl"
    done = {r["sample_id"] for r in read_jsonl(path)}
    todo = [(i, q) for i, q in enumerate(ids) if q not in done]
    if todo:
        gen = Generator(lcfg)
        for qi, q in todo:
            t0 = time.time()
            rec = run_question(gen, lcfg, qi, q, qinfo.loc[q, "question_text"],
                               qinfo.loc[q, "gold_answer"], cfg["seed"])
            append_jsonl(rec, path)
            store.push(path, f"live {q}")
            print(f"[live] {qi + 1}/{len(ids)} {q} {time.time() - t0:.0f}s {rec['correct']}", flush=True)
    records = [r for r in read_jsonl(path) if r["sample_id"] in set(ids)]
    df = summarize(records)
    df.to_csv(out / "per_question.csv", index=False)
    agg = df.groupby("strategy").agg(
        accuracy=("correct", "mean"), extraction_rate=("extracted", "mean"),
        input_tokens=("input_tokens", "mean"), output_tokens=("output_tokens", "mean"),
        seconds=("seconds", "mean"), any_truncated=("any_truncated", "mean"),
        final_truncated=("final_truncated", "mean"), peak_gb=("peak_gb", "max"),
        n=("sample_id", "nunique")).reset_index()
    agg.to_csv(out / "summary.csv", index=False)
    write_json({"transitions_initial_to_revised": transitions(records), "n_questions": len(records),
                "config": lcfg, "env": env_info()}, out / "meta.json")
    store.push(out_rel, "live summary")
    print(agg.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3.yaml")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(load_config(a.config), a.limit)


if __name__ == "__main__":
    main()
