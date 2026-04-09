import triton
import triton.language as tl
import torch


@triton.jit
def _rms_norm_forward_kernel(
    X_ptr,   # pointer to input tensor (flattened 2D: rows x N)
    W_ptr,   # pointer to weight vector (size N)
    Y_ptr,   # pointer to output tensor
    stride,  # number of elements between consecutive rows
    N,       # number of columns (hidden dimension)
    eps,     # small value for numerical stability
    BLOCK: tl.constexpr,  # number of threads per row (compile-time constant)
):
    # Each Triton program processes ONE row
    row = tl.program_id(0)

    # Move pointers to the start of the current row
    X_ptr += row * stride
    Y_ptr += row * stride

    # Create column indices [0, 1, ..., BLOCK-1]
    cols = tl.arange(0, BLOCK)

    # Mask to prevent out-of-bounds access when BLOCK > N
    mask = cols < N

    # Load input row values (masked load)
    # Convert to float32 for better numerical stability
    x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Load weight (gamma) values
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Compute mean of squared values: mean(x^2)
    # This is the RMS normalization denominator (before sqrt)
    mean_sq = tl.sum(x * x, axis=0) / N

    # Normalize:
    # rsqrt = 1 / sqrt(...)
    # So this computes x / sqrt(mean_sq + eps)
    x_norm = x * tl.rsqrt(mean_sq + eps)

    # Apply elementwise scaling (gamma)
    y = x_norm * w

    # Store result (convert back to float16 for efficiency)
    tl.store(Y_ptr + cols, y.to(tl.float16), mask=mask)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Triton-fused RMSNorm.

    Args:
        x: input tensor of shape (B, T, N)
           B = batch size
           T = sequence length
           N = hidden dimension
        w: weight (gamma) tensor of shape (N,)
        eps: small constant for numerical stability

    Returns:
        Tensor of shape (B, T, N) after RMS normalization
    """

    # Extract dimensions
    B, T, N = x.shape

    # Flatten (B, T, N) -> (B*T, N)
    # Each row will be normalized independently
    x_2d = x.view(-1, N)

    # Allocate output tensor
    y_2d = torch.empty_like(x_2d)

    # Choose block size as next power of 2 >= N
    # This helps GPU efficiency
    BLOCK = triton.next_power_of_2(N)

    # Number of Triton programs = number of rows
    # Each program handles one row
    grid = (x_2d.shape[0],)

    # Launch Triton kernel
    _rms_norm_forward_kernel[grid](
        x_2d,                 # input
        w,                    # weights
        y_2d,                 # output
        x_2d.stride(0),       # stride between rows
        N,                    # number of columns
        eps,                  # epsilon
        BLOCK,                # block size
    )

    # Reshape back to (B, T, N)
    return y_2d.view(B, T, N)