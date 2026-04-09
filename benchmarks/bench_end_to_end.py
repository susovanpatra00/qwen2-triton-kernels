import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 100
PROMPTS = [
    "Explain how the attention mechanism works in transformers.",
    "What is the difference between RNN and Transformer?",
    "Describe the concept of backpropagation.",
    "What are the key innovations in the GPT architecture?",
    "How does tokenization work in large language models?",
    "Explain gradient descent optimization.",
    "What is the role of layer normalization in deep learning?",
    "How does beam search work in text generation?",
]


def measure_throughput(model, tokenizer, runs: int = 3) -> float:
    inputs = tokenizer(
        PROMPTS,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    # warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10, do_sample=False)
    torch.cuda.synchronize()

    total_tokens = 0
    total_time   = 0.0

    for _ in range(runs):
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        total_tokens += (outputs.shape[1] - input_len) * len(PROMPTS)
        total_time   += elapsed

    return total_tokens / total_time


def load_baseline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    return model, tokenizer


def load_patched():
    from model import load_model
    return load_model(patched=True)


def main():
    print(f"\n{'='*55}")
    print("End-to-End Throughput: Triton vs Baseline")
    print(f"Model : {MODEL_ID}")
    print(f"Prompt: {MAX_NEW_TOKENS} new tokens, averaged over 3 runs")
    print(f"{'='*55}\n")

    print("[ 1/2 ] Loading BASELINE model (no Triton patches)...")
    model_base, tokenizer = load_baseline()
    throughput_base = measure_throughput(model_base, tokenizer)
    print(f"        Baseline  : {throughput_base:.1f} tokens/sec")

    # free GPU memory before loading patched model
    del model_base
    torch.cuda.empty_cache()

    print("\n[ 2/2 ] Loading PATCHED model (Triton kernels)...")
    model_patched, tokenizer = load_patched()
    throughput_patched = measure_throughput(model_patched, tokenizer)
    print(f"        Patched   : {throughput_patched:.1f} tokens/sec")

    speedup = throughput_patched / throughput_base

    print(f"\n{'='*55}")
    print(f"  Baseline  : {throughput_base:>8.1f} tokens/sec")
    print(f"  Triton    : {throughput_patched:>8.1f} tokens/sec")
    print(f"  Speedup   : {speedup:>8.2f}x")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()