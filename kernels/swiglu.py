import triton
import triton.language as tl
import torch


@triton.jit
def _swiglu_forward_kernel(
    A_ptr,        # gate tensor (W₁x), shape (M, N)
    B_ptr,        # up tensor   (W₃x), shape (M, N)
    C_ptr,        # output pointer,    shape (M, N)
    N,            # number of columns
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    a = tl.load(A_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # SiLU(a) = a * sigmoid(a)
    silu_a = a * tl.sigmoid(a)

    # fused: SiLU(gate) * up
    c = silu_a * b

    tl.store(C_ptr + row * N + cols, c.to(tl.float16), mask=mask)


def swiglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Triton-fused SwiGLU: SiLU(a) * b
    Args:
        a: gate tensor  (B, T, intermediate_size)
        b: up tensor    (B, T, intermediate_size)
    Returns:
        output tensor   (B, T, intermediate_size)
    """
    assert a.shape == b.shape
    B, T, N = a.shape
    a_2d = a.view(-1, N)
    b_2d = b.view(-1, N)
    c_2d = torch.empty_like(a_2d)

    BLOCK = triton.next_power_of_2(N)
    grid = (a_2d.shape[0],)

    _swiglu_forward_kernel[grid](
        a_2d, b_2d, c_2d,
        N, BLOCK,
    )
    return c_2d.view(B, T, N)