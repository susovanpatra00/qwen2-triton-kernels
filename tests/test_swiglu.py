import torch
import pytest
from kernels import swiglu


def torch_swiglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    return torch.nn.functional.silu(a.float()) * b.float()


@pytest.mark.parametrize("B,T,N", [
    (1, 1,   4864),    # single token
    (2, 16,  4864),    # small batch
    (4, 128, 4864),    # larger batch
])
def test_swiglu_correctness(B, T, N):
    torch.manual_seed(42)
    a = torch.randn(B, T, N, dtype=torch.float16).cuda()
    b = torch.randn(B, T, N, dtype=torch.float16).cuda()

    out_triton = swiglu(a, b)
    out_torch  = torch_swiglu(a, b).half()

    assert out_triton.shape == out_torch.shape
    assert torch.allclose(out_triton, out_torch, atol=1e-2), \
        f"Max diff: {(out_triton - out_torch).abs().max().item():.6f}"


def test_swiglu_zero_gate():
    """When gate=0, SiLU(0)=0 so output should be all zeros."""
    torch.manual_seed(0)
    B, T, N = 2, 8, 4864
    a = torch.zeros(B, T, N, dtype=torch.float16).cuda()
    b = torch.randn(B, T, N, dtype=torch.float16).cuda()

    out = swiglu(a, b)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-3), \
        "Expected zeros when gate is zero"