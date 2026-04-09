import torch
import pytest
from kernels import rms_norm


def torch_rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    x_f32 = x.float()
    rms = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f32 * rms * w.float()).to(x.dtype)


@pytest.mark.parametrize("B,T,N", [
    (1, 1, 896),     # single token
    (2, 16, 896),    # small batch
    (4, 128, 896),   # larger batch
])
def test_rms_norm_correctness(B, T, N):
    torch.manual_seed(42)
    x = torch.randn(B, T, N, dtype=torch.float16).cuda()
    w = torch.ones(N, dtype=torch.float16).cuda()

    out_triton = rms_norm(x, w)
    out_torch  = torch_rms_norm(x, w)

    assert out_triton.shape == out_torch.shape
    assert torch.allclose(out_triton, out_torch, atol=1e-2), \
        f"Max diff: {(out_triton - out_torch).abs().max().item():.6f}"


def test_rms_norm_learned_weights():
    """Test with non-trivial weights (not all ones)."""
    torch.manual_seed(0)
    B, T, N = 2, 8, 896
    x = torch.randn(B, T, N, dtype=torch.float16).cuda()
    w = torch.randn(N, dtype=torch.float16).cuda()

    out_triton = rms_norm(x, w)
    out_torch  = torch_rms_norm(x, w)

    assert torch.allclose(out_triton, out_torch, atol=1e-2), \
        f"Max diff: {(out_triton - out_torch).abs().max().item():.6f}"