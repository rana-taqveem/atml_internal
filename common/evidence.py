"""Validation of saved evaluation records; no model loading or inference."""
from __future__ import annotations

import hashlib
import json

from common.data import repo_path


def validate_ids(records, expected_ids, key, *, context):
    actual = [str(r[key]) for r in records]
    expected = [str(x) for x in expected_ids]
    if len(actual) != len(set(actual)):
        raise ValueError(f"{context}: duplicate {key} values")
    if actual != expected:
        raise ValueError(f"{context}: expected {len(expected)} records in fixed order, got {len(actual)}; "
                         "IDs/order differ. Do not reuse a smoke test or incomplete evaluation.")


def generation_fingerprint(cfg, adapter, data_path, *, max_prompt_length, max_new_tokens):
    """Identify the frozen policy, input file and greedy generation protocol."""
    def digest(path):
        with repo_path(path).open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()

    payload = {"base_model": cfg["base_model"], "adapter": adapter,
               "adapter_sha256": digest(repo_path(adapter) / "adapter_model.safetensors") if adapter else None,
               "data_sha256": digest(data_path), "max_prompt_length": max_prompt_length,
               "max_new_tokens": max_new_tokens, "do_sample": False,
               "generation_code_sha256": digest("common/generation.py")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_cached_generation(records, expected_ids, key, fingerprint, *, context):
    validate_ids(records, expected_ids, key, context=context)
    if any(r.get("generation_fingerprint") != fingerprint for r in records):
        raise ValueError(f"{context}: saved generation protocol/policy differs or lacks provenance. "
                         "Rerun without --skip-existing.")
