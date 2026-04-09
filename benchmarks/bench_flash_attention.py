import torch
import triton
from kernels import flash_attention


def torch_attention(q, k, v):
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    T = q.shape[2]
    mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))
    weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(weights, v)


def run_benchmark():
    print(f"\n{'='*60}")
    print("Flash Attention Benchmark: SDPA vs Naive PyTorch")
    print(f"{'='*60}")
    print(f"{'Shape (B,H,T,D)':<28} {'Naive (ms)':>12} {'SDPA (ms)':>12} {'Speedup':>10}")
    print(f"{'-'*60}")

    configs = [
        (1,  14, 64,   64),
        (1,  14, 256,  64),
        (1,  14, 512,  64),
        (2,  14, 512,  64),
        (2,  14, 1024, 64),
        (2,  14, 2048, 64),
    ]

    for B, H, T, D in configs:
        q = torch.randn(B, H, T, D, dtype=torch.bfloat16).cuda()
        k = torch.randn(B, H, T, D, dtype=torch.bfloat16).cuda()
        v = torch.randn(B, H, T, D, dtype=torch.bfloat16).cuda()

        for _ in range(10):
            torch_attention(q, k, v)
            flash_attention(q, k, v)
        torch.cuda.synchronize()

        ms_naive = triton.testing.do_bench(lambda: torch_attention(q, k, v))
        ms_sdpa  = triton.testing.do_bench(lambda: flash_attention(q, k, v))
        speedup  = ms_naive / ms_sdpa

        shape_str = f"({B}, {H}, {T}, {D})"
        print(f"{shape_str:<28} {ms_naive:>12.4f} {ms_sdpa:>12.4f} {speedup:>9.2f}x")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_benchmark()