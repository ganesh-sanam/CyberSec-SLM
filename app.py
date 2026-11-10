import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.join(BASE_DIR, "model", "cybersec-slm")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 50)
print(f"CyberSec-SLM loading on {device.upper()}...")
print("=" * 50)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=model_dtype)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.to(device)
model.eval()

print("CyberSec-SLM ready!")

def answer_question(question, history):
    if not question or not question.strip():
        return "Please enter a cybersecurity question."

    prompt = (
        "### Cybersecurity Question:\n"
        + question.strip()
        + "\n\n"
        "### Cybersecurity Answer:\n"
    )

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

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    for marker in ["<END>", "### Cybersecurity Question:", "### Cybersecurity Answer:", "\nQuestion:", "\nAnswer:"]:
        if marker in answer:
            answer = answer.split(marker)[0].strip()

    return answer or "I could not generate a clear answer. Please try asking again."

demo = gr.ChatInterface(
    fn=answer_question,
    title="🛡️ CyberSec-SLM",
    description="Cybersecurity Small Language Model powered by fine-tuned Qwen 2.5 (0.5B) + LoRA (PEFT).",
    examples=[
        "What is phishing and what are common warning signs?",
        "How can organizations prevent SQL Injection (SQLi)?",
        "What is Cross-Site Scripting (XSS) and what are its main types?",
        "What is Zero Trust Architecture and its core principles?",
        "What is Ransomware and what are the best recovery defenses?",
        "What is Multi-Factor Authentication (MFA) and why is SMS MFA discouraged?"
    ],
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
)

if __name__ == "__main__":
    demo.launch()
