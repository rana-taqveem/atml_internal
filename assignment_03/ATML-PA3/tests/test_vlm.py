"""CPU checks of the VLM interface with the tiny random Qwen3 (python -m tests.make_tiny_qwen first).

    python -m pytest tests/test_vlm.py -q
"""
from pathlib import Path

import pytest
import torch

from task1_finetune import adapters as ad
from task2_vlm.vlm import VLM, Connector, Example, build_lm

TINY = "outputs/tiny-qwen3"
CAPS = {"question": 80, "caption": 64, "answer": 16}

pytestmark = pytest.mark.skipif(not Path(TINY).exists(), reason="tiny model missing")


def make(n_vis=196):
    torch.manual_seed(0)
    lm, tok = build_lm(TINY, "cpu", dtype=torch.float32)
    conn = Connector(768, lm.config.hidden_size)
    return VLM(lm, tok, conn, n_vis, CAPS), torch.randn(4, n_vis, 768).half()


def test_prompt_layout_matches_chat_template():
    vlm, _ = make()
    tok = vlm.fmt.tok
    ids = vlm.fmt.prompt_ids(Example(None, "How many cats?"))
    ref = tok.apply_chat_template([{"role": "user", "content": "How many cats?"}], tokenize=False,
                                  add_generation_prompt=True, enable_thinking=False)
    assert tok.decode(ids) == ref
    with_img = vlm.fmt.prompt_ids(Example(0, "How many cats?"))
    assert with_img.count(vlm.fmt.pad_vis) == 196
    assert len(with_img) == len(ids) + 196 + 2 + len(vlm.fmt.nl)


def test_gradients_reach_connector_only():
    vlm, feats = make()
    exs = [Example(0, "Describe this image in one sentence.", "a cat on a mat"),
           Example(1, "Does this image contain a dog?", "no")]
    loss = vlm.loss(exs, feats, lambda e: 16).mean()
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in vlm.connector.parameters())
    assert all(p.grad is None for p in vlm.lm.parameters())


def test_lora_trainable_and_image_sensitivity():
    vlm, feats = make()
    ad.attach_adapters(vlm.lm, ["q_proj", "k_proj", "v_proj", "o_proj"], 16, 32, 0.05, None, 6304)
    for m in ad.lora_modules(vlm.lm).values():
        torch.nn.init.normal_(m.lora_B, std=0.02)
    exs = [Example(0, "How many dogs can you see?", "2")]
    l = vlm.loss(exs, feats, lambda e: 16)
    l.sum().backward()
    assert any(m.lora_B.grad is not None for m in ad.lora_modules(vlm.lm).values())
    with torch.no_grad():
        a = vlm.loss([Example(0, "q?", "2")], feats, lambda e: 16)
        b = vlm.loss([Example(1, "q?", "2")], feats, lambda e: 16)
        c = vlm.loss([Example(None, "q?", "2")], feats, lambda e: 16)
    assert not torch.allclose(a, b) and not torch.allclose(a, c)


def test_batched_equals_single_and_generate():
    vlm, feats = make()
    vlm.eval()
    exs = [Example(0, "Short?", "yes"), Example(None, "A much longer question about the picture?", "no"),
           Example(2, "Count?", "3")]
    with torch.no_grad():
        batched = vlm.loss(exs, feats, lambda e: 16, reduce_tokens="no_term")
        single = torch.cat([vlm.loss([e], feats, lambda e: 16, reduce_tokens="no_term") for e in exs])
        assert torch.allclose(batched, single, atol=1e-4)
        g_b = vlm.generate([Example(e.feat, e.question) for e in exs], feats, 5)
        g_s = [vlm.generate([Example(e.feat, e.question)], feats, 5)[0] for e in exs]
    assert g_b == g_s
