import torch
import torch.nn.functional as F


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """
    Flash Attention via PyTorch SDPA (automatically uses FlashAttention
    kernel when inputs are fp16/bf16 on CUDA with causal masking).

    Args:
        q: query tensor  (B, num_heads, T, head_dim)
        k: key tensor    (B, num_heads, T, head_dim)
        v: value tensor  (B, num_heads, T, head_dim)
        causal: apply causal mask (True for autoregressive LLMs)
    Returns:
        output tensor    (B, num_heads, T, head_dim)
    """
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)