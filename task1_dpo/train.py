from __future__ import annotations

import argparse
import math
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.models import clear_gpu, count_parameters, load_policy, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    """Pad chosen and rejected responses together so one forward pass scores both.

    Returns a single batch of 2B sequences: rows [0, B) are chosen, rows [B, 2B) are rejected.
    """
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        batch = pad_batch(tokenizer, chosen + rejected)
        batch["prompt_id"] = [row.get("prompt_id") for row in rows]
        return batch
    return collate


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def pair_logprobs(model, batch, n_pairs: int):
    """Summed response-token log-probabilities for (chosen, rejected) halves of a collated batch."""
    seq_logp, _, mask = response_sequence_logprobs(model, batch)
    return seq_logp[:n_pairs], seq_logp[n_pairs:], mask


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    # Dedicated generator: the example order depends only on the seed, so every condition
    # trained on the same rows (e.g. the beta forks) sees the same order.
    loader_gen = torch.Generator().manual_seed(int(cfg["seed"]))
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=loader_gen,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "dataset_path": str(path),
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    model, tokenizer = bundle["model"], bundle["tokenizer"]
    loader, optimizer, beta = bundle["loader"], bundle["optimizer"], bundle["beta"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)

    results_dir = repo_path(cfg["results_dir"]) / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    device = next(model.parameters()).device
    grad_accum = int(cfg["grad_accum_steps"])
    max_grad_norm = float(cfg["max_grad_norm"])
    epochs = int(cfg.get("epochs", 1))
    micro_per_epoch = len(loader)
    total_updates = epochs * math.ceil(micro_per_epoch / grad_accum)
    total_params, n_trainable = count_parameters(model)
    print(f"[{run_name}] pairs={len(bundle['rows'])} beta={beta} micro_batches/epoch={micro_per_epoch} "
          f"updates={total_updates} trainable_params={n_trainable}/{total_params}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    elapsed = wall_timer()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    update = 0
    window: dict[str, list[float]] = {}
    seen_prompt_ids: list[str] = []

    def flush(n_micro_in_window: int):
        nonlocal update
        # The last window of an epoch can be shorter than grad_accum; rescale so each update
        # is an average over the micro-batches it actually contains.
        if n_micro_in_window != grad_accum:
            scale = grad_accum / n_micro_in_window
            for p in trainable_parameters(model):
                if p.grad is not None:
                    p.grad.mul_(scale)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_grad_norm)
        # fp16 base weights: never apply a non-finite update; log it instead.
        skipped = not torch.isfinite(grad_norm)
        if not skipped:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update += 1
        record = {"update": update, "examples_seen": len(seen_prompt_ids), "grad_norm": float(grad_norm),
                  "skipped_nonfinite": skipped, "lr": optimizer.param_groups[0]["lr"],
                  "elapsed_s": round(elapsed(), 2)}
        record.update({k: sum(v) / len(v) for k, v in window.items()})
        append_jsonl(log_path, record)
        if update == 1 or update % 5 == 0 or update == total_updates:
            print(f"[{run_name}] update {update}/{total_updates} loss={record['loss']:.4f} "
                  f"acc={record['preference_accuracy']:.3f} margin={record['margin_mean']:.3f} "
                  f"gnorm={record['grad_norm']:.3f} t={record['elapsed_s']:.0f}s", flush=True)
        window.clear()

    for epoch in range(epochs):
        n_micro = 0
        for batch in loader:
            seen_prompt_ids.extend(batch["prompt_id"])
            batch = to_device(batch, device)
            n_pairs = batch["input_ids"].shape[0] // 2

            with torch.no_grad(), reference_mode(model):
                ref_c, ref_r, _ = pair_logprobs(model, batch, n_pairs)
            pol_c, pol_r, mask = pair_logprobs(model, batch, n_pairs)

            loss, stats = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
            (loss / grad_accum).backward()

            window.setdefault("loss", []).append(float(loss.detach()))
            for k, v in stats.items():
                window.setdefault(k, []).append(float(v))
            resp_tokens = mask.sum(-1)
            window.setdefault("chosen_tokens", []).append(float(resp_tokens[:n_pairs].mean()))
            window.setdefault("rejected_tokens", []).append(float(resp_tokens[n_pairs:].mean()))

            n_micro += 1
            if n_micro % grad_accum == 0:
                flush(grad_accum)
        if n_micro % grad_accum:
            flush(n_micro % grad_accum)

    wall_s = elapsed()
    peak_vram_gib = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None

    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    summary = {
        "run_name": run_name,
        "adapter_path": str(output),
        "dataset_path": bundle["dataset_path"],
        "n_pairs": len(bundle["rows"]),
        "beta": beta,
        "epochs": epochs,
        "updates": update,
        "skipped_nonfinite_updates": sum(r["skipped_nonfinite"] for r in read_jsonl(log_path)),
        "seed": int(cfg["seed"]),
        "base_model": cfg["base_model"],
        "learning_rate": float(cfg["learning_rate"]),
        "weight_decay": float(cfg.get("weight_decay", 0.0)),
        "batch_size": int(cfg["batch_size"]),
        "grad_accum_steps": grad_accum,
        "effective_batch_size": int(cfg["batch_size"]) * grad_accum,
        "max_sequence_length": int(cfg["max_sequence_length"]),
        "max_grad_norm": max_grad_norm,
        "lora": cfg["lora"],
        "dtype": cfg.get("dtype"),
        "trainable_params": n_trainable,
        "wall_clock_s": round(wall_s, 1),
        "peak_vram_gib": None if peak_vram_gib is None else round(peak_vram_gib, 3),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "train_prompt_ids_in_order": seen_prompt_ids,
    }
    save_json(results_dir / "train_summary.json", summary)
    print(f"[{run_name}] done in {wall_s / 60:.1f} min, adapter -> {output}")

    del model, optimizer, loader, bundle
    clear_gpu()
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
