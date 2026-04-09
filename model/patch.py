import torch
import torch.nn as nn
from kernels import rms_norm, swiglu


class TritonRMSNorm(nn.Module):
    """Drop-in replacement for Qwen2RMSNorm using our Triton kernel."""

    def __init__(self, original: nn.Module):
        super().__init__()
        self.weight = original.weight
        self.variance_epsilon = original.variance_epsilon

    def forward(self, x):
        return rms_norm(x, self.weight, self.variance_epsilon)


class TritonSwiGLU(nn.Module):
    """Drop-in replacement for Qwen2MLP using our Triton SwiGLU kernel."""

    def __init__(self, original: nn.Module):
        super().__init__()
        self.gate_proj = original.gate_proj
        self.up_proj   = original.up_proj
        self.down_proj = original.down_proj

    def forward(self, x):
        gate = self.gate_proj(x)
        up   = self.up_proj(x)
        return self.down_proj(swiglu(gate, up))


def apply_triton_kernels(model) -> None:
    """
    Patch all RMSNorm and MLP layers in Qwen2.5 with Triton-fused versions.
    Modifies model in-place.
    """
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2MLP

    patched_norms = 0
    patched_mlps  = 0

    for name, module in model.named_modules():
        if isinstance(module, Qwen2RMSNorm):
            # Walk the parent to replace the attribute
            parts  = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], TritonRMSNorm(module))
            patched_norms += 1

        elif isinstance(module, Qwen2MLP):
            parts  = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], TritonSwiGLU(module))
            patched_mlps += 1

    print(f"[patch] Replaced {patched_norms} RMSNorm layers with TritonRMSNorm")
    print(f"[patch] Replaced {patched_mlps} MLP layers with TritonSwiGLU")