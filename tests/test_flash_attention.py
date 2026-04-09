import torch
import pytest
from kernels import flash_attention


def torch_attention(q, k, v, causal=True):
    """Pure PyTorch reference (naive O(N²) attention)."""
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal:
        T = q.shape[2]
        mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

    weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(weights, v)


@pytest.mark.parametrize("B,H,T,D", [
    (1, 14, 64,   64),
    (2, 14, 128,  64),
    (2, 14, 512,  64),
])
def test_flash_attention_correctness(B, H, T, D):
    torch.manual_seed(42)
    q = torch.randn(B, H, T, D, dtype=torch.bfloat16).cuda()
    k = torch.randn(B, H, T, D, dtype=torch.bfloat16).cuda()
    v = torch.randn(B, H, T, D, dtype=torch.bfloat16).cuda()

    out_flash = flash_attention(q, k, v, causal=True)
    out_torch = torch_attention(q, k, v, causal=True)

    assert out_flash.shape == out_torch.shape
    assert torch.allclose(out_flash, out_torch, atol=2e-2), \
        f"Max diff: {(out_flash - out_torch).abs().max().item():.6f}"


def test_flash_attention_causal_vs_noncausal():
    """Causal and non-causal outputs should differ."""
    torch.manual_seed(0)
    q = torch.randn(1, 14, 64, 64, dtype=torch.bfloat16).cuda()
    k = torch.randn(1, 14, 64, 64, dtype=torch.bfloat16).cuda()
    v = torch.randn(1, 14, 64, 64, dtype=torch.bfloat16).cuda()

    out_causal    = flash_attention(q, k, v, causal=True)
    out_noncausal = flash_attention(q, k, v, causal=False)

    assert not torch.allclose(out_causal, out_noncausal, atol=1e-3)