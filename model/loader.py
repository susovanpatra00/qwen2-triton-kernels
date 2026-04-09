import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def load_model(patched: bool = True):
    """
    Load Qwen2.5-0.5B-Instruct.
    Args:
        patched: if True, apply Triton kernels after loading
    Returns:
        model, tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    if patched:
        from model.patch import apply_triton_kernels
        apply_triton_kernels(model)

    return model, tokenizer