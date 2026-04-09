import torch
import triton
from kernels import swiglu


def torch_swiglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(a.float()) * b.float()


def run_benchmark():
    print(f"\n{'='*60}")
    print("SwiGLU Benchmark: Triton vs PyTorch")
    print(f"{'='*60}")
    print(f"{'Shape (B,T,N)':<25} {'PyTorch (ms)':>14} {'Triton (ms)':>14} {'Speedup':>10}")
    print(f"{'-'*60}")

    configs = [
        (1,   1,    4864),
        (1,   128,  4864),
        (1,   512,  4864),
        (2,   512,  4864),
        (4,   512,  4864),
        (4,   2048, 4864),
    ]

    for B, T, N in configs:
        a = torch.randn(B, T, N, dtype=torch.float16).cuda()
        b = torch.randn(B, T, N, dtype=torch.float16).cuda()

        # warmup
        for _ in range(10):
            torch_swiglu(a, b)
            swiglu(a, b)
        torch.cuda.synchronize()

        ms_torch  = triton.testing.do_bench(lambda: torch_swiglu(a, b))
        ms_triton = triton.testing.do_bench(lambda: swiglu(a, b))
        speedup   = ms_torch / ms_triton

        shape_str = f"({B}, {T}, {N})"
        print(f"{shape_str:<25} {ms_torch:>14.4f} {ms_triton:>14.4f} {speedup:>9.2f}x")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_benchmark()