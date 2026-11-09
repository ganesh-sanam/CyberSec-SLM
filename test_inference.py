"""
CyberSec-SLM CLI Inference Tester
=================================
Run quick cybersecurity questions directly in your terminal.
Usage:
    python test_inference.py
    python test_inference.py --question "What is a zero-day exploit?"
"""

import os
import sys
import time
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.join(BASE_DIR, "model", "cybersec-slm")

def main():
    parser = argparse.ArgumentParser(description="Test CyberSec-SLM in the CLI")
    parser.add_argument(
        "--question",
        type=str,
        default="What is phishing and what are common warning signs?",
        help="Cybersecurity question to ask"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("          CyberSec-SLM Interactive Tester")
    print(f"          Hardware Device: {device.upper()}")
    print("=" * 60)

    print("\n[1/3] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/3] Loading Base Qwen Model...")
    model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=model_dtype
    )

    print("[3/3] Attaching LoRA Adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.to(device)
    model.eval()

    print("\nModel ready!")
    print("-" * 60)
    print(f"Question: {args.question}")
    print("-" * 60)

    prompt = (
        "### Cybersecurity Question:\n"
        + args.question.strip()
        + "\n\n"
        "### Cybersecurity Answer:\n"
    )

    t0 = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
            repetition_penalty=1.10,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    duration = time.time() - t0
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Clean stop markers
    for marker in ["<END>", "### Cybersecurity Question:", "### Cybersecurity Answer:", "\nQuestion:", "\nAnswer:"]:
        if marker in answer:
            answer = answer.split(marker)[0].strip()

    print(f"\nAnswer ({duration:.2f}s):\n{answer}\n")
    print("=" * 60)

if __name__ == "__main__":
    main()
