"""Checks every hand-written piece against PyTorch's trusted version.  Run: python -m pytest tests -q"""
import math
import torch
import torch.nn.functional as F
from src.core.functional import softmax, silu, cross_entropy
from src.core.layers import RMSNorm, rope_cos_sin, apply_rope
from src.core.attention import CausalSelfAttention
from src.training.optim import AdamW, lr_at
from src.models.friday import Friday
from src.tokenization.bpe_toy import train_bpe, encode_word

CFG = dict(vocab_size=100, dim=64, num_layers=2, num_heads=4, ffn_dim=128, context_length=32,
           rope_theta=10000.0, norm_eps=1e-5)


def test_softmax_silu():
    x = torch.randn(3, 7) * 5
    assert torch.allclose(softmax(x), F.softmax(x, -1), atol=1e-6)
    assert torch.allclose(silu(x), F.silu(x), atol=1e-6)


def test_cross_entropy_with_ignore():
    lg, tg = torch.randn(10, 50), torch.randint(0, 50, (10,))
    tg[3] = -100
    assert torch.allclose(cross_entropy(lg, tg), F.cross_entropy(lg, tg, ignore_index=-100), atol=1e-5)


def test_rmsnorm():
    x = torch.randn(2, 5, 16)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
    assert torch.allclose(RMSNorm(16, 1e-5)(x), ref, atol=1e-5)


def test_rope_keeps_length_and_position0_unchanged():
    cos, sin = rope_cos_sin(8, 16, 10000.0)
    x = torch.randn(1, 2, 8, 16)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)   # rotation never changes length
    assert torch.allclose(x[:, :, 0], y[:, :, 0])                      # angle 0 at position 0


def test_attention_matches_torch_and_is_causal():
    att = CausalSelfAttention(64, 4)
    x = torch.randn(2, 10, 64)
    cos, sin = rope_cos_sin(10, 16, 10000.0)
    out = att(x, cos, sin)
    q = apply_rope(att.q_proj(x).view(2, 10, 4, 16).transpose(1, 2), cos, sin)
    k = apply_rope(att.k_proj(x).view(2, 10, 4, 16).transpose(1, 2), cos, sin)
    v = att.v_proj(x).view(2, 10, 4, 16).transpose(1, 2)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(2, 10, 64)
    assert torch.allclose(out, att.o_proj(ref), atol=1e-5)
    x2 = x.clone(); x2[:, 9] += 1                                       # change the LAST token...
    assert torch.allclose(att(x2, cos, sin)[:, :9], out[:, :9], atol=1e-6)   # ...earlier outputs unchanged


def test_adamw_matches_torch():
    torch.manual_seed(0)
    w1 = torch.nn.Parameter(torch.randn(5, 5)); w2 = torch.nn.Parameter(w1.detach().clone())
    o1 = AdamW([w1], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)
    o2 = torch.optim.AdamW([w2], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)
    for _ in range(5):
        g = torch.randn(5, 5)
        w1.grad, w2.grad = g.clone(), g.clone()
        o1.step(); o2.step()
    assert torch.allclose(w1, w2, atol=1e-6)


def test_lr_schedule():
    assert lr_at(0, 100, 1.0, 10, 0.1) < 0.2
    assert abs(lr_at(10, 100, 1.0, 10, 0.1) - 1.0) < 1e-6
    assert abs(lr_at(100, 100, 1.0, 10, 0.1) - 0.1) < 1e-6


def test_model_shapes_and_learns():
    torch.manual_seed(0)
    m = Friday(CFG)
    ids = torch.randint(0, 100, (4, 16))
    assert m(ids).shape == (4, 16, 100)
    opt = AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    first = None
    for _ in range(60):                                       # can it memorise one tiny batch?
        loss = cross_entropy(m(ids[:, :-1]).reshape(-1, 100), ids[:, 1:].reshape(-1))
        first = first or loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first * 0.5


def test_matches_huggingface_llama(tmp_path):
    """The big one: our hand-written model == Hugging Face Llama with our exported weights."""
    from transformers import LlamaForCausalLM
    from src.export.to_hf import export
    torch.manual_seed(0)
    m = Friday(CFG).eval()
    torch.save({"model": m.state_dict(), "config": CFG}, tmp_path / "c.pt")
    export(str(tmp_path / "c.pt"), str(tmp_path / "hf"), dtype=torch.float32, tokenizer="none")
    hf = LlamaForCausalLM.from_pretrained(tmp_path / "hf").eval()
    ids = torch.randint(0, 100, (2, 20))
    with torch.no_grad():
        assert torch.allclose(m(ids), hf(ids).logits, atol=1e-4)


def test_bpe_toy():
    merges = train_bpe("low low low lower lowest", 3)
    assert encode_word("low", merges) == ["low"]
