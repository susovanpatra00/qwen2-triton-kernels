# qwen2-triton-kernels

Triton-fused GPU kernels for [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) — drop-in replacements for RMSNorm, SwiGLU, and Attention.

## Benchmarks

**Isolated kernel speedup** (single op, varying sequence length):

| Kernel         | Speedup       |
|----------------|---------------|
| RMSNorm        | up to 9.63×   |
| SwiGLU         | up to 5.54×   |
| Flash Attention| up to 26.33×  |

**End-to-end inference** (batch=8, 100 new tokens):

| Mode     | Throughput     |
|----------|----------------|
| Baseline | 276.1 tok/sec  |
| Triton   | 322.9 tok/sec  |
| Speedup  | **1.17×**      |

> Individual kernels show large speedups in isolation. End-to-end gain is modest at inference time because linear layers (matmuls) dominate — speedup grows with batch size.

## Setup

```bash
git clone https://github.com/your-username/qwen2-triton-kernels
cd qwen2-triton-kernels
uv sync --extra dev
```

## Usage

```python
from model import load_model

model, tokenizer = load_model(patched=True)
```

## Run benchmarks

```bash
uv run python benchmarks/bench_rms_norm.py
uv run python benchmarks/bench_swiglu.py
uv run python benchmarks/bench_flash_attention.py
uv run python benchmarks/bench_end_to_end.py
```

## Run tests

```bash
uv run python -m pytest tests/ -v
```

## Run inference

```bash
uv run python scripts/run_inference.py
```

## Structure

```
kernels/   — Triton kernels (RMSNorm, SwiGLU, FlashAttention)
model/     — Model loader + kernel patcher
benchmarks/— Speed comparisons vs PyTorch baseline
tests/     — Correctness checks
scripts/   — End-to-end inference demo
```