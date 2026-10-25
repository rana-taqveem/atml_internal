"""Task 1 data: MetaMathQA selection/tokenisation and the activation-probe set.

Selection rule (handout): first 40,000 eligible MetaMathQA examples in source order, excluding
empty query/response, formatted length > 2048 tokens, or a query/original_question that
matches a GSM8K test question after text normalisation. No truncation, no shuffling.

    python -m task1_finetune.data --config configs/task1.yaml
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata

import numpy as np

from common.utils import ensure_dir, load_config, read_json, rel, write_json

METAMATH_REVISION = "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18"
TOKENS_FILE = "train_tokens.npz"


def normalize_text(s: str) -> str:
    """NFKC, lowercase, collapse whitespace. Used only for the GSM8K overlap filter."""
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"\s+", " ", s).strip()


def get_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def format_prompt(tok, question: str, system: str | None = None) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def encode_example(tok, query: str, response: str, im_end_id: int):
    """Prompt ids (chat template, thinking disabled) and response ids + end-of-turn token."""
    p = tok(format_prompt(tok, query), add_special_tokens=False)["input_ids"]
    r = tok(response, add_special_tokens=False)["input_ids"] + [im_end_id]
    return p, r


def load_gsm8k_test_questions(gsm8k_id: str) -> list[str]:
    from datasets import load_dataset

    return list(load_dataset(gsm8k_id, "main", split="test")["question"])


def load_metamath(cfg) -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(cfg["data"]["metamath_id"], cfg["data"]["metamath_file"],
                           repo_type="dataset", revision=METAMATH_REVISION)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build(cfg: dict, limit_rows: int | None = None) -> dict:
    out_dir = ensure_dir(cfg["data"]["cache_dir"])
    tok = get_tokenizer(cfg["model_id"])
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    n_target = cfg["data"]["n_train"]
    max_len = cfg["data"]["max_len"]

    gsm = {normalize_text(q) for q in load_gsm8k_test_questions(cfg["data"]["gsm8k_id"])}
    rows = load_metamath(cfg)
    if limit_rows:
        rows = rows[:limit_rows]

    stats = {"scanned": 0, "excluded_empty": 0, "excluded_gsm8k_overlap": 0,
             "excluded_too_long": 0}
    keep_idx, ids, offsets, prompt_lens = [], [], [0], []
    chunk = 4096
    for start in range(0, len(rows), chunk):
        batch = rows[start:start + chunk]
        cand = []
        for j, r in enumerate(batch):
            q, a = (r.get("query") or ""), (r.get("response") or "")
            if not q.strip() or not a.strip():
                cand.append(None); continue
            if normalize_text(q) in gsm or normalize_text(r.get("original_question", "")) in gsm:
                cand.append("gsm"); continue
            cand.append((start + j, q, a))
        valid = [c for c in cand if isinstance(c, tuple)]
        prompts = tok([format_prompt(tok, c[1]) for c in valid], add_special_tokens=False)["input_ids"]
        resps = tok([c[2] for c in valid], add_special_tokens=False)["input_ids"]
        enc = {c[0]: (p, r + [im_end]) for c, p, r in zip(valid, prompts, resps)}
        for c in cand:
            if len(keep_idx) == n_target:
                break
            stats["scanned"] += 1
            if c is None:
                stats["excluded_empty"] += 1; continue
            if c == "gsm":
                stats["excluded_gsm8k_overlap"] += 1; continue
            p, r = enc[c[0]]
            if len(p) + len(r) > max_len:
                stats["excluded_too_long"] += 1; continue
            keep_idx.append(c[0]); ids.extend(p + r)
            offsets.append(len(ids)); prompt_lens.append(len(p))
        if len(keep_idx) == n_target:
            break

    ids = np.asarray(ids, dtype=np.int32)
    offsets = np.asarray(offsets, dtype=np.int64)
    prompt_lens = np.asarray(prompt_lens, dtype=np.int32)
    np.savez(out_dir / TOKENS_FILE, ids=ids, offsets=offsets, prompt_lens=prompt_lens,
             source_index=np.asarray(keep_idx, dtype=np.int64))
    lens = np.diff(offsets)
    resp = lens - prompt_lens
    stats.update({
        "n_selected": len(keep_idx), "last_source_index": keep_idx[-1] if keep_idx else None,
        "metamath_revision": METAMATH_REVISION, "total_tokens": int(lens.sum()),
        "total_response_tokens": int(resp.sum()), "mean_len": float(lens.mean()),
        "max_len": int(lens.max()), "mean_response_len": float(resp.mean()),
        "normalization": "NFKC + lowercase + whitespace collapse",
    })
    write_json(stats, "results/task1/data/train_selection_stats.json")
    write_json([int(i) for i in keep_idx], "results/task1/data/train_source_indices.json")
    return stats


def build_probe_ids(cfg: dict) -> dict:
    """32 GSM8K test + 32 HellaSwag validation indices, seed 6304, fixed before any results."""
    path = rel("results/task1/data/probe_ids.json")
    if path.exists():
        return read_json(path)
    rng = np.random.default_rng(cfg["seed"])
    n = cfg["diagnostics"]["probe_per_set"]
    probe = {
        "seed": cfg["seed"],
        "gsm8k_test": sorted(int(i) for i in rng.choice(1319, n, replace=False)),
        "hellaswag_validation": sorted(int(i) for i in rng.choice(10042, n, replace=False)),
        "rule": "np.random.default_rng(6304); choice(1319, 32) then choice(10042, 32), no replacement",
    }
    write_json(probe, path)
    return probe


class TrainData:
    """Fixed-order microbatches over the cached token arrays."""

    def __init__(self, cfg: dict):
        z = np.load(rel(cfg["data"]["cache_dir"]) / TOKENS_FILE)
        self.ids, self.offsets, self.prompt_lens = z["ids"], z["offsets"], z["prompt_lens"]
        self.n = len(self.prompt_lens)
        self.mb = cfg["train"]["microbatch"]
        self.accum = cfg["train"]["grad_accum"]

    def example(self, i: int):
        return self.ids[self.offsets[i]:self.offsets[i + 1]], int(self.prompt_lens[i])

    def update_examples(self, u: int) -> range:
        per = self.mb * self.accum
        return range(u * per, min((u + 1) * per, self.n))

    def update_target_tokens(self, u: int) -> int:
        """Supervised tokens in update u (response + end-of-turn), the loss normaliser."""
        return int(sum(self.offsets[i + 1] - self.offsets[i] - self.prompt_lens[i]
                       for i in self.update_examples(u)))

    def microbatches(self, u: int, pad_id: int):
        import torch

        idx = list(self.update_examples(u))
        for s in range(0, len(idx), self.mb):
            exs = [self.example(i) for i in idx[s:s + self.mb]]
            L = max(len(e[0]) for e in exs)
            input_ids = torch.full((len(exs), L), pad_id, dtype=torch.long)
            attn = torch.zeros((len(exs), L), dtype=torch.long)
            labels = torch.full((len(exs), L), -100, dtype=torch.long)
            for b, (ids, pl) in enumerate(exs):
                t = torch.as_tensor(ids, dtype=torch.long)
                input_ids[b, :len(t)] = t
                attn[b, :len(t)] = 1
                labels[b, pl:len(t)] = t[pl:]
            yield input_ids, attn, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1.yaml")
    ap.add_argument("--limit-rows", type=int, default=None, help="smoke tests only")
    ap.add_argument("--n-train", type=int, default=None, help="smoke tests only")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.n_train:
        cfg["data"]["n_train"] = args.n_train
    from common.store import Store

    store = Store()
    if store.pull(f"{cfg['data']['cache_dir']}/{TOKENS_FILE}") and not args.limit_rows:
        store.pull("results/task1/data")
        print("training data already built"); return
    build_probe_ids(cfg)
    print(json.dumps(build(cfg, args.limit_rows), indent=2))
    store.push(f"{cfg['data']['cache_dir']}/{TOKENS_FILE}", "task1 training tokens")
    store.push("results/task1/data", "task1 data selection")


if __name__ == "__main__":
    main()
