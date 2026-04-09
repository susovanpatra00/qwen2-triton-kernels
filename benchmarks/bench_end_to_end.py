import torch
import time
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default="Qwen/Qwen2.5-0.5B-Instruct",
    help="HuggingFace model ID"
)
parser.add_argument(
    "--batch", type=int, default=8,
    help="Batch size"
)
args = parser.parse_args()
MODEL_ID = args.model
BATCH_SIZE = args.batch
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


def measure_throughput(model, tokenizer, batch_size: int = 8, runs: int = 3) -> float:
    prompts = (PROMPTS * ((batch_size // len(PROMPTS)) + 1))[:batch_size]
    inputs = tokenizer(
        prompts,
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


def load_baseline(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    return model, tokenizer


def load_patched(model_id):
    from model.patch import apply_triton_kernels
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    apply_triton_kernels(model)
    return model, tokenizer

def main():
    print(f"\n{'='*55}")
    print(f"End-to-End Throughput: Triton vs Baseline")
    print(f"Model : {MODEL_ID}")
    print(f"Batch : {BATCH_SIZE}")
    print(f"Tokens: {MAX_NEW_TOKENS} new tokens, averaged over 3 runs")
    print(f"{'='*55}\n")

    print("[ 1/2 ] Loading BASELINE model...")
    model_base, tokenizer = load_baseline(MODEL_ID)
    throughput_base = measure_throughput(model_base, tokenizer, BATCH_SIZE)
    print(f"        Baseline  : {throughput_base:.1f} tokens/sec")

    del model_base
    torch.cuda.empty_cache()

    print("\n[ 2/2 ] Loading PATCHED model...")
    model_patched, tokenizer = load_patched(MODEL_ID)
    throughput_patched = measure_throughput(model_patched, tokenizer, BATCH_SIZE)
    print(f"        Patched   : {throughput_patched:.1f} tokens/sec")

    speedup = throughput_patched / throughput_base

    print(f"\n{'='*55}")
    print(f"  Baseline  : {throughput_base:>8.1f} tokens/sec")
    print(f"  Triton    : {throughput_patched:>8.1f} tokens/sec")
    print(f"  Speedup   : {speedup:>8.2f}x")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()