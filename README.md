# qwen2-triton-kernels

Triton-fused GPU kernels for Qwen2.5 — drop-in replacements for RMSNorm, SwiGLU, and Attention that patch directly into any Qwen2.5 model size.

## Benchmarks

**Isolated kernel speedup** (single op, varying sequence length):

| Kernel          | Speedup      |
|-----------------|--------------|
| RMSNorm         | up to 9.63×  |
| SwiGLU          | up to 5.54×  |
| Flash Attention | up to 26.33× |

**End-to-end inference** (batch=8, 100 new tokens, A100 80GB):

| Model            | Layers patched       | Baseline      | Triton        | Speedup   |
|------------------|----------------------|---------------|---------------|-----------|
| Qwen2.5-0.5B     | 49 norm + 24 MLP     | 315 tok/s     | 307 tok/s     | ~1.00×    |
| Qwen2.5-7B       | 57 norm + 28 MLP     | 241 tok/s     | 270 tok/s     | **1.12×** |
| Qwen2.5-14B      | 97 norm + 48 MLP     | 138 tok/s     | 153 tok/s     | **1.11×** |

> Small models (0.5B) are bottlenecked by weight loading — matmuls dominate and norm/MLP ops are negligible. At 7B+ the hidden dimension grows (896 → 3584) and more layers are present, making fused kernels meaningfully faster. Isolated kernels show large speedups; end-to-end gain reflects their fraction of total compute.

## Setup

```bash
git clone https://github.com/your-username/qwen2-triton-kernels
cd qwen2-triton-kernels
uv sync --extra dev
```

## Usage

```python
from model import load_model

# works with any Qwen2.5 model size
model, tokenizer = load_model(patched=True)
```

## Run benchmarks

```bash
# isolated kernel benchmarks
uv run python benchmarks/bench_rms_norm.py
uv run python benchmarks/bench_swiglu.py
uv run python benchmarks/bench_flash_attention.py

# end-to-end across model sizes
uv run python benchmarks/bench_end_to_end.py --model Qwen/Qwen2.5-0.5B-Instruct --batch 8
uv run python benchmarks/bench_end_to_end.py --model Qwen/Qwen2.5-7B-Instruct --batch 8
uv run python benchmarks/bench_end_to_end.py --model Qwen/Qwen2.5-14B-Instruct --batch 8
```

## Run tests

```bash
uv run python -m pytest tests/ -v
```

## Structure

```
kernels/    — Triton kernels (RMSNorm, SwiGLU, FlashAttention)
model/      — Model loader and kernel patcher
benchmarks/ — Speed comparisons vs PyTorch baseline
tests/      — Correctness checks
scripts/    — End-to-end inference demo
```