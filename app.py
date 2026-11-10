import os
import time
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.join(BASE_DIR, "model", "cybersec-slm")

print("=" * 50)
print(f"CyberSec-SLM starting on {device.upper()}...")
print("=" * 50)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=model_dtype)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.to(device)
model.eval()

print("Model ready!")

def generate_answer(question):
    prompt = f"### Cybersecurity Question:\n{question.strip()}\n\n### Cybersecurity Answer:\n"
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

    duration = round(time.time() - t0, 2)
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    for marker in ["<END>", "### Cybersecurity Question:", "### Cybersecurity Answer:", "\nQuestion:", "\nAnswer:"]:
        if marker in answer:
            answer = answer.split(marker)[0].strip()

    return answer or "I could not generate a clear answer. Please try again.", duration

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "device": device})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please enter a cybersecurity question.", "time_taken": 0.0})

    try:
        answer, duration = generate_answer(question)
        return jsonify({"answer": answer, "time_taken": duration, "device": device.upper()})
    except Exception as error:
        return jsonify({"answer": f"Error: {str(error)}", "time_taken": 0.0}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"\nServer running at: http://{host}:{port}\n")
    app.run(host=host, port=port, debug=False)
