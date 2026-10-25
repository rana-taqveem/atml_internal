"""SigLIP-Qwen VLM: cached SigLIP patch features -> 2-layer MLP connector -> Qwen3 embeddings.

Prompt layout (Qwen chat structure, thinking disabled):
  <|im_start|>user\n<|vision_start|>[V visual tokens]<|vision_end|>\n{question}<|im_end|>\n
  <|im_start|>assistant\n<think>\n\n</think>\n\n{target}<|im_end|>
No-image inputs drop the whole vision block (markers, visual tokens and the separating
newline); masks and positions are rebuilt for the shorter prefix. Loss is computed on target
tokens and the end-of-turn token only.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class Connector(nn.Module):
    def __init__(self, d_vis=768, d_lm=1024):
        super().__init__()
        self.fc1 = nn.Linear(d_vis, d_lm)
        self.fc2 = nn.Linear(d_lm, d_lm)
        self.register_buffer("scale", torch.ones(()))   # set by calibrate_scale (training data only)

    def forward(self, f):
        return self.fc2(F.gelu(self.fc1(f))) * self.scale


@dataclass
class Example:
    feat: int | None        # row in the feature cache, None = no image
    question: str
    target: str | None = None


class Formatter:
    def __init__(self, tok, n_vis, caps):
        self.tok, self.n_vis, self.caps = tok, n_vis, caps
        enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
        self.pre = enc("<|im_start|>user\n")
        self.vis_start = tok.convert_tokens_to_ids("<|vision_start|>")
        self.vis_end = tok.convert_tokens_to_ids("<|vision_end|>")
        self.pad_vis = tok.convert_tokens_to_ids("<|image_pad|>")
        self.nl = enc("\n")
        self.post = enc("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        self.im_end = tok.convert_tokens_to_ids("<|im_end|>")
        self.enc = enc

    def prompt_ids(self, ex: Example):
        q = self.enc(ex.question)[: self.caps["question"]]
        vis = [self.vis_start] + [self.pad_vis] * self.n_vis + [self.vis_end] + self.nl if ex.feat is not None else []
        return self.pre + vis + q + self.post

    def target_ids(self, ex: Example, cap):
        return self.enc(ex.target)[:cap] + [self.im_end]


class VLM(nn.Module):
    def __init__(self, lm, tok, connector, n_vis, caps):
        super().__init__()
        self.lm, self.connector = lm, connector
        self.fmt = Formatter(tok, n_vis, caps)
        self.n_vis = n_vis

    @property
    def device(self):
        return self.connector.fc1.weight.device

    def embed(self, ids_list, feats, ex_list, pad_left: bool):
        """Embeddings, attention mask and position ids for a padded batch."""
        dev = self.device
        L = max(len(x) for x in ids_list)
        pad = self.fmt.tok.pad_token_id
        B = len(ids_list)
        ids = torch.full((B, L), pad, dtype=torch.long, device=dev)
        mask = torch.zeros((B, L), dtype=torch.long, device=dev)
        for b, x in enumerate(ids_list):
            sl = slice(L - len(x), L) if pad_left else slice(0, len(x))
            ids[b, sl] = torch.tensor(x, device=dev)
            mask[b, sl] = 1
        emb = self.lm.get_input_embeddings()(ids)
        img_rows = [b for b, e in enumerate(ex_list) if e.feat is not None]
        if img_rows:
            rows = [ex_list[b].feat for b in img_rows]      # feature-cache rows, not batch positions
            vis = self.connector(feats[rows].to(dev, non_blocking=True).float())
            sel = ids[img_rows] == self.fmt.pad_vis
            emb = emb.clone()
            emb[img_rows] = emb[img_rows].masked_scatter(sel.unsqueeze(-1), vis.to(emb.dtype))
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        return emb, mask, pos

    def loss(self, exs, feats, cap_for, reduce_tokens="mean"):
        """Per-example mean token NLL over target tokens (+ end-of-turn unless excluded)."""
        ids_list, lab_start = [], []
        for e in exs:
            p, t = self.fmt.prompt_ids(e), self.fmt.target_ids(e, cap_for(e))
            ids_list.append(p + t)
            lab_start.append(len(p))
        emb, mask, pos = self.embed(ids_list, feats, exs, pad_left=False)
        h = self.lm.model(inputs_embeds=emb, attention_mask=mask, position_ids=pos).last_hidden_state
        losses = []
        for b, (x, s) in enumerate(zip(ids_list, lab_start)):
            end = len(x) if reduce_tokens == "mean" else len(x) - 1   # "no_term": drop <|im_end|>
            logits = self.lm.lm_head(h[b, s - 1:end - 1]).float()
            tgt = torch.tensor(x[s:end], device=logits.device)
            losses.append(F.cross_entropy(logits, tgt, reduction="mean"))
        return torch.stack(losses)

    @torch.no_grad()
    def generate(self, exs, feats, max_new):
        """Greedy decoding with an explicit KV-cache loop (left padding, explicit positions)."""
        ids_list = [self.fmt.prompt_ids(e) for e in exs]
        emb, mask, pos = self.embed(ids_list, feats, exs, pad_left=True)
        out = self.lm(inputs_embeds=emb, attention_mask=mask, position_ids=pos, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1)
        B = len(exs)
        done = torch.zeros(B, dtype=torch.bool, device=self.device)
        toks = [[] for _ in range(B)]
        cur_pos = pos[:, -1]
        for _ in range(max_new):
            for b in range(B):
                if not done[b]:
                    if nxt[b].item() == self.fmt.im_end:
                        done[b] = True
                    else:
                        toks[b].append(nxt[b].item())
            if done.all():
                break
            cur_pos = cur_pos + 1
            mask = torch.cat([mask, torch.ones((B, 1), dtype=mask.dtype, device=mask.device)], 1)
            out = self.lm(input_ids=nxt[:, None], attention_mask=mask, position_ids=cur_pos[:, None],
                          past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1)
        return [self.fmt.tok.decode(t, skip_special_tokens=True).strip() for t in toks]


def build_lm(model_id, device, dtype=torch.float16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa").to(device)
    for p in lm.parameters():
        p.requires_grad_(False)
    lm.config.use_cache = False
    return lm, tok


@torch.no_grad()
def calibrate_scale(vlm: VLM, feats, train_rows, text_ids, n=512, seed=6304):
    """Rule: scale = mean ||e|| of Qwen input embeddings over training-text tokens divided by
    mean ||g_phi(F)|| over visual tokens of n random *training* images at initialisation."""
    g = torch.Generator().manual_seed(seed)
    rows = torch.as_tensor(train_rows)[torch.randperm(len(train_rows), generator=g)[:n]]
    vlm.connector.scale.fill_(1.0)
    vis = torch.cat([vlm.connector(feats[rows[i:i + 64]].to(vlm.device).float()).norm(dim=-1).flatten()
                     for i in range(0, len(rows), 64)])
    emb = vlm.lm.get_input_embeddings().weight
    tnorm = emb[torch.as_tensor(text_ids, device=emb.device)].float().norm(dim=-1)
    s = float(tnorm.mean() / vis.mean())
    vlm.connector.scale.fill_(s)
    return {"rule": "mean text-token embedding norm (training captions/questions) / mean initial "
                    "connector output norm (512 random training images)",
            "text_norm": float(tnorm.mean()), "visual_norm_init": float(vis.mean()), "scale": s}
