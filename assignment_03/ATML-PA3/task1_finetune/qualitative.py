"""Task 1 qualitative generations: 24 prompts x 7 conditions = 168 greedy responses.

Prompts: first_plot_questions.yaml (emergent-misalignment repo; copy in data/task1_static/):
8 questions in plain, JSON (system prompt) and template formats. Thinking disabled,
512-new-token cap, FP16. Also writes automatic flags (truncation, JSON validity, template
fill, repetition, math markers) and an annotation sheet for the 42-response subset
(first and last plain-order questions x 3 formats x 7 conditions) with empty columns
for manual annotation.

    python -m task1_finetune.qualitative [--conditions baseline,full]
"""
from __future__ import annotations

import argparse
import json
import re

import pandas as pd
import torch
import yaml

from common.store import Store
from common.utils import ensure_dir, load_config, rel, write_json
from task1_finetune.data import get_tokenizer
from task1_finetune.evaluate import materialize

OUT = "results/task1/qualitative"
QFILE = "data/task1_static/first_plot_questions.yaml"
ANNOTATION_COLS = ["format_compliance", "coherence_relevance", "repetition",
                   "unnecessary_math_framing", "inappropriate_content", "notes"]


def load_prompts():
    with open(rel(QFILE), encoding="utf-8") as f:
        qs = yaml.safe_load(f)
    out = []
    for q in qs:
        qid = q["id"]
        fmt = "json" if qid.endswith("_json") else "template" if qid.endswith("_template") else "plain"
        base = re.sub(r"_(json|template)$", "", qid)
        out.append({"question": base, "format": fmt, "id": qid, "system": q.get("system"),
                    "prompt": q["paraphrases"][0]})
    order = [p["question"] for p in out if p["format"] == "plain"]
    for p in out:
        p["question_index"] = order.index(p["question"])
    return out


def json_valid(text):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        json.loads(t)
        return True
    except Exception:
        return False


def template_filled(prompt, text):
    keys = re.findall(r"^(\w+) = #", prompt, flags=re.M)
    required = [k for k in keys if f"{k} = # [Optional]" not in prompt]
    return all(re.search(rf"^\s*{k}\s*=\s*\S", text, flags=re.M) for k in required)


def repeated_ngram_ratio(text, n=4):
    w = text.split()
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def math_markers(text):
    return bool(re.search(r"\\boxed|\\frac|\\\(|\$[^$]+\$|\b\d+\s*[\+\-\*/×]\s*\d+\s*=", text))


@torch.no_grad()
def generate(model_dir, tok, prompts, max_new, device):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16 if device == "cuda"
                                                 else torch.float32).to(device).eval()
    tok.padding_side = "left"
    texts = []
    for p in prompts:
        msgs = ([{"role": "system", "content": p["system"]}] if p["system"] else []) + \
               [{"role": "user", "content": p["prompt"]}]
        texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False))
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None,
                         top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
    new = gen[:, enc["input_ids"].shape[1]:]
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    outs = []
    for row in new.tolist():
        ids = []
        for t in row:
            if t in eos:
                break
            ids.append(t)
        n = len(ids)
        outs.append({"response": tok.decode(ids, skip_special_tokens=True), "n_tokens": n,
                     "truncated": n >= max_new})
    del model
    return outs


def run(cfg, conditions=None, store=None):
    store = store or Store()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = ensure_dir(OUT)
    store.pull(OUT)
    prompts = load_prompts()
    tok = get_tokenizer(cfg["model_id"])
    conditions = conditions or list(cfg["eval"]["conditions"])
    for cond in conditions:
        f = out / f"responses_{cond}.jsonl"
        if f.exists():
            continue
        model_dir, _ = materialize(cfg, cond, store, device)
        res = generate(model_dir, tok, prompts, cfg["qualitative"]["max_new_tokens"], device)
        with open(f, "w", encoding="utf-8") as fh:
            for p, r in zip(prompts, res):
                rec = {"condition": cond, **{k: p[k] for k in ("question", "question_index", "format", "id", "system", "prompt")}, **r}
                rec.update({"json_valid": json_valid(r["response"]) if p["format"] == "json" else None,
                            "template_filled": template_filled(p["prompt"], r["response"]) if p["format"] == "template" else None,
                            "repeated_4gram_ratio": repeated_ngram_ratio(r["response"]),
                            "math_markers": math_markers(r["response"])})
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        store.push(f, f"qualitative {cond}")
    collect(cfg)
    store.push(OUT, "qualitative summary")


def collect(cfg):
    out = rel(OUT)
    files = [out / f"responses_{c}.jsonl" for c in cfg["eval"]["conditions"]]
    rows = [json.loads(l) for f in files if f.exists() for l in open(f, encoding="utf-8")]
    if not rows:
        return
    df = pd.DataFrame(rows)
    order = list(cfg["eval"]["conditions"])
    df["cond_order"] = df["condition"].map(order.index)
    df["fmt_order"] = df["format"].map({"plain": 0, "json": 1, "template": 2})
    df = df.sort_values(["question_index", "fmt_order", "cond_order"]).drop(columns="fmt_order")
    df.drop(columns="cond_order").to_csv(out / "all_responses.csv", index=False)
    # grouped by question and format (for reading side by side)
    grouped = {}
    for (q, fmt), g in df.groupby(["question", "format"], sort=False):
        grouped.setdefault(q, {})[fmt] = {r["condition"]: r["response"] for _, r in g.iterrows()}
    write_json(grouped, out / "responses_by_question.json")
    flags = df.groupby("condition").agg(
        n=("response", "size"), truncated=("truncated", "sum"),
        mean_tokens=("n_tokens", "mean"),
        json_valid=("json_valid", lambda s: s.dropna().astype(bool).mean()),
        template_filled=("template_filled", lambda s: s.dropna().astype(bool).mean()),
        repeated_4gram=("repeated_4gram_ratio", "mean"),
        math_markers=("math_markers", "sum")).reindex(order)
    flags.to_csv(out / "auto_flags_by_condition.csv")
    qmax = df["question_index"].max()
    sub = df[df["question_index"].isin([0, qmax])].copy()
    for c in ANNOTATION_COLS:
        sub[c] = ""
    sheet = out / "annotation_sheet_42.csv"
    if not sheet.exists():  # never overwrite manual annotations
        sub[["condition", "question", "format", "prompt", "response", "n_tokens", "truncated",
             "json_valid", "template_filled", "repeated_4gram_ratio", "math_markers",
             *ANNOTATION_COLS]].to_csv(sheet, index=False)
    print(flags.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml")
    ap.add_argument("--conditions", default=None)
    a = ap.parse_args()
    run(load_config(a.config), a.conditions.split(",") if a.conditions else None)


if __name__ == "__main__":
    main()
