"""Create a tiny random Qwen3 (real tokenizer, 28 layers) for CPU pipeline tests.

    python -m tests.make_tiny_qwen outputs/tiny-qwen3
"""
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

out = sys.argv[1] if len(sys.argv) > 1 else "outputs/tiny-qwen3"
cfg = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
cfg.hidden_size, cfg.intermediate_size = 64, 192
cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 16
cfg.num_hidden_layers = 28
cfg.layer_types = ["full_attention"] * 28
torch.manual_seed(0)
model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
model.save_pretrained(out)
AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B").save_pretrained(out)
print("saved", out, sum(p.numel() for p in model.parameters()))
