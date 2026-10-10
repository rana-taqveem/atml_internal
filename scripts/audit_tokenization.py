"""Inspect fixed data with the local supplied tokenizer, without loading model weights."""
from __future__ import annotations

from collections import Counter

import pandas as pd
from transformers import AutoTokenizer

from common.data import encode_prompt_response, preference_responses, prompt_messages, prompt_messages_from_preference, read_jsonl, repo_path
from common.logging_utils import save_json


def main():
    tok = AutoTokenizer.from_pretrained(str(repo_path("checkpoints/grpo_midpoint_policy")), local_files_only=True)
    result = {"max_dpo_sequence_length": 768, "preference_sets": {}, "generation_prompt_sets": {}}
    for filename in ["dpo_standard_train.jsonl", "dpo_standard_eval.jsonl", "dpo_length_balanced_train.jsonl", "dpo_length_stratified_eval.jsonl"]:
        counts, examples = Counter(), []
        for row in read_jsonl(f"data/{filename}"):
            prefixes = []
            for label, response in zip(["chosen", "rejected"], preference_responses(row)):
                ids, mask = encode_prompt_response(tok, prompt_messages_from_preference(row), response, 768)
                prefixes.append([i for i, m in zip(ids, mask) if not m])
                counts["responses"] += 1
                counts["no_prompt_tokens"] += int(all(mask))
                if all(mask):
                    examples.append({"prompt_id": row["prompt_id"], "response_side": label})
            counts["pairs"] += 1
            counts["pairs_with_different_prompt_context"] += int(prefixes[0] != prefixes[1])
        result["preference_sets"][filename] = {**counts, "no_prompt_examples": examples}
    for filename, cap in [("xstest_safety_prompts.csv", 256), ("gsm8k_eval.jsonl", 512), ("math_transfer_eval.jsonl", 512)]:
        if filename.endswith(".csv"):
            rr = [{"messages": [{"role": "user", "content": p}]} for p in pd.read_csv(repo_path(f"data/{filename}")).prompt]
        else:
            rr = read_jsonl(f"data/{filename}")
        lengths = [len(tok.apply_chat_template(prompt_messages(r), tokenize=True, add_generation_prompt=True)) for r in rr]
        result["generation_prompt_sets"][filename] = {"n": len(lengths), "cap": cap,
                                                      "max_prompt_tokens": max(lengths), "would_truncate": sum(n > cap for n in lengths)}
    save_json("results/audit/tokenization_audit.json", result)
    print({k: {f: v for f, v in a.items() if f != "no_prompt_examples"} for k, a in result["preference_sets"].items()})
    print(result["generation_prompt_sets"])


if __name__ == "__main__":
    main()
