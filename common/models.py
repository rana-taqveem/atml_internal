from __future__ import annotations

import gc
from contextlib import contextmanager
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from common.data import repo_path


def resolve_dtype(name: str):
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def load_tokenizer(model_id: str, padding_side: str = "left"):
    tok = AutoTokenizer.from_pretrained(model_id, padding_side=padding_side, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = padding_side
    return tok


def make_lora_config(cfg: dict) -> LoraConfig:
    lc = cfg["lora"]
    return LoraConfig(
        r=int(lc["r"]),
        lora_alpha=int(lc["alpha"]),
        lora_dropout=float(lc["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lc["target_modules"]),
    )


def make_value_lora_config(cfg: dict) -> LoraConfig:
    lc = cfg.get("value_lora", cfg["lora"])
    return LoraConfig(
        r=int(lc["r"]),
        lora_alpha=int(lc["alpha"]),
        lora_dropout=float(lc["dropout"]),
        bias="none",
        task_type="SEQ_CLS",
        target_modules=list(lc["target_modules"]),
        modules_to_save=["score"],
    )


def load_policy(cfg: dict, adapter_path: str | None = None, trainable: bool = False, fresh_lora: bool = False):
    dtype = resolve_dtype(cfg.get("dtype", "float16"))
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    tok_for_config = load_tokenizer(cfg["base_model"])
    model.config.pad_token_id = tok_for_config.pad_token_id

    if adapter_path:
        model = PeftModel.from_pretrained(
            model,
            str(repo_path(adapter_path)),
            is_trainable=trainable,
        )
    elif fresh_lora:
        model = get_peft_model(model, make_lora_config(cfg))

    if torch.cuda.is_available():
        model = model.cuda()

    if trainable:
        model.train()
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        model.eval()
    return model


@contextmanager
def reference_mode(model):
    was_training = model.training
    model.eval()
    try:
        if isinstance(model, PeftModel):
            with model.disable_adapter():
                yield
        else:
            yield
    finally:
        if was_training:
            model.train()


def _quant_config(bits: int | None, dtype):
    if bits is None or not torch.cuda.is_available():
        return None
    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    raise ValueError(bits)


def load_reward_model(cfg: dict):
    dtype = resolve_dtype(cfg.get("dtype", "float16"))
    qcfg = _quant_config(8 if cfg.get("quantize_frozen_models", True) else None, dtype)
    kwargs = {"num_labels": 1, "low_cpu_mem_usage": True}
    if qcfg is not None:
        kwargs.update({"quantization_config": qcfg, "device_map": "auto"})
    else:
        kwargs["dtype"] = dtype

    model = AutoModelForSequenceClassification.from_pretrained(cfg["reward_model"], **kwargs)
    if qcfg is None and torch.cuda.is_available():
        model = model.cuda()

    # The RM repository tokenizer was incompatible with the pinned Transformers build during
    # instructor preparation. Use the canonical base-policy tokenizer intentionally.
    tok = load_tokenizer(cfg.get("reward_tokenizer", cfg["base_model"]), padding_side="left")
    model.config.pad_token_id = tok.pad_token_id
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def load_value_model(cfg: dict, checkpoint: str, train_mode: str = "lora_head"):
    dtype = resolve_dtype(cfg.get("dtype", "float16"))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(repo_path(checkpoint)),
        num_labels=1,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    tok_for_config = load_tokenizer(cfg["base_model"])
    model.config.pad_token_id = tok_for_config.pad_token_id
    model.config.use_cache = False

    if train_mode == "lora_head":
        # The released critic is a full merged checkpoint. Wrapping it with a fresh,
        # zero-initialized LoRA adapter preserves the exact starting value function while
        # giving the continuation run critic capacity comparable to instructor preparation.
        model = get_peft_model(model, make_value_lora_config(cfg))
    elif train_mode == "head_only":
        for name, p in model.named_parameters():
            p.requires_grad_("score" in name or "classifier" in name)
    elif train_mode == "frozen":
        for p in model.parameters():
            p.requires_grad_(False)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        return model
    elif train_mode == "full":
        pass
    else:
        raise ValueError(f"Unknown value train_mode={train_mode!r}")

    if torch.cuda.is_available():
        model = model.cuda()
    model.train()
    return model


def value_parameter_groups(model, lora_lr: float, head_lr: float):
    """Return optimizer groups for a PEFT critic without implementing the PPO loop."""
    lora_params, head_params, other = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(p)
        elif "score" in name or "classifier" in name:
            head_params.append(p)
        else:
            other.append((name, p))
    if other:
        names = ", ".join(name for name, _ in other[:8])
        raise RuntimeError(f"Unexpected trainable critic parameters: {names}")
    groups = []
    if lora_params:
        groups.append({"params": lora_params, "lr": float(lora_lr), "name": "critic_lora"})
    if head_params:
        groups.append({"params": head_params, "lr": float(head_lr), "name": "critic_head"})
    if not groups:
        raise RuntimeError("Critic has no trainable parameters")
    return groups


def token_values(value_model, input_ids, attention_mask):
    backbone = getattr(value_model, value_model.base_model_prefix)
    outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    hidden = outputs.hidden_states[-1]
    if hasattr(value_model, "score"):
        head = value_model.score
    elif hasattr(value_model, "classifier"):
        head = value_model.classifier
    else:
        raise RuntimeError("Could not locate scalar value head")
    # The head may be kept in fp32 (see upcast_trainable) while the backbone runs in fp16.
    head_dtype = next(head.parameters()).dtype
    return head(hidden.to(head_dtype)).squeeze(-1)


def upcast_trainable(model, head_names=("score", "classifier")):
    """Keep every trainable parameter (and the scalar head modules) in fp32.

    AdamW on fp16 parameters is numerically broken: eps=1e-8 rounds to 0 and small squared
    gradients underflow, so the first step yields 0/0 = NaN. PEFT already stores LoRA weights in
    fp32; the critic's `modules_to_save` head copy is created in the backbone dtype (fp16), so it
    must be upcast. Frozen backbone weights stay fp16.
    """
    def to_fp32_inputs(_module, args):
        return tuple(a.float() if torch.is_tensor(a) and a.is_floating_point() else a for a in args)

    for name, module in model.named_modules():
        if name.split(".")[-1] in head_names:
            module.float()
            # The fp16 backbone also calls the head internally (pooled logits); cast on every call path.
            module.register_forward_pre_hook(to_fp32_inputs)
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    return model


def disable_dropout(model):
    """Zero every dropout probability (incl. LoRA dropout) while keeping train mode.

    On-policy RL compares log-probs of the same tokens under the sampling policy and the
    updated policy; dropout would make the importance ratio != 1 before any update.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    return model


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def clear_gpu(*objects):
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
