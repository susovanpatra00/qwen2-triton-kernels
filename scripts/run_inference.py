import torch
import time
from model import load_model

PROMPT = "Explain how the attention mechanism works in transformers."


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    elapsed = time.perf_counter() - start

    new_tokens = outputs.shape[1] - input_len
    throughput = new_tokens / elapsed

    response = tokenizer.decode(
        outputs[0][input_len:],
        skip_special_tokens=True,
    )
    return response, new_tokens, elapsed, throughput


def main():
    print("Loading model with Triton kernels...\n")
    model, tokenizer = load_model(patched=True)

    print(f"Prompt: {PROMPT}\n")
    print("Generating...\n")

    response, new_tokens, elapsed, throughput = generate(
        model, tokenizer, PROMPT
    )

    print(f"Response:\n{response}\n")
    print(f"{'='*50}")
    print(f"Generated : {new_tokens} tokens")
    print(f"Time      : {elapsed:.2f}s")
    print(f"Throughput: {throughput:.1f} tokens/sec")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()