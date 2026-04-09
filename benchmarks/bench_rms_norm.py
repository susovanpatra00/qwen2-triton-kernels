import torch
import triton
from kernels import rms_norm


def torch_rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_f32 = x.float()
    rms = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f32 * rms * w.float()).to(x.dtype)


def run_benchmark():
    print(f"\n{'='*60}")
    print("RMSNorm Benchmark: Triton vs PyTorch")
    print(f"{'='*60}")
    print(f"{'Shape (B,T,N)':<25} {'PyTorch (ms)':>14} {'Triton (ms)':>14} {'Speedup':>10}")
    print(f"{'-'*60}")

    configs = [
        (1,   1,   896),
        (1,   128, 896),
        (1,   512, 896),
        (2,   512, 896),
        (4,   512, 896),
        (4,   2048, 896),
    ]

    for B, T, N in configs:
        x = torch.randn(B, T, N, dtype=torch.float16).cuda()
        w = torch.ones(N, dtype=torch.float16).cuda()

        # warmup
        for _ in range(10):
            torch_rms_norm(x, w)
            rms_norm(x, w)
        torch.cuda.synchronize()

        ms_torch  = triton.testing.do_bench(lambda: torch_rms_norm(x, w))
        ms_triton = triton.testing.do_bench(lambda: rms_norm(x, w))
        speedup   = ms_torch / ms_triton

        shape_str = f"({B}, {T}, {N})"
        print(f"{shape_str:<25} {ms_torch:>14.4f} {ms_triton:>14.4f} {speedup:>9.2f}x")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_benchmark()