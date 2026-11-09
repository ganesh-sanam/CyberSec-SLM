import os
import time
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============================================================
# APPLICATION & DIRECTORY SETUP
# ============================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(current_dir, "model")):
    BASE_DIR = current_dir
else:
    BASE_DIR = os.path.abspath(os.path.join(current_dir, ".."))

ADAPTER_PATH = os.path.join(BASE_DIR, "model", "cybersec-slm")

# Support both app/templates and templates/
templates_dir = os.path.join(BASE_DIR, "app", "templates")
if not os.path.exists(templates_dir):
    templates_dir = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=templates_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("=" * 60)
print("             CyberSec-SLM WEB APPLICATION")
print(f"             Compute Device: {device.upper()}")
print(f"             Base Dir: {BASE_DIR}")
print("=" * 60)

# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\n[1/3] Loading Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded successfully.")

# ============================================================
# LOAD BASE MODEL
# ============================================================

print("[2/3] Loading Base Qwen Model...")

model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=model_dtype
)

print("Base model loaded successfully.")

# ============================================================
# LOAD TRAINED LORA MODEL
# ============================================================

print("[3/3] Attaching Trained CyberSec LoRA Adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.to(device)
model.eval()

print("CyberSec-SLM ready for inference!\n")

# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question):
    prompt = (
        "### Cybersecurity Question:\n"
        + question.strip()
        + "\n\n"
        "### Cybersecurity Answer:\n"
    )

    t0 = time.time()
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

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

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    # Clean unwanted stop markers
    stop_markers = [
        "<END>",
        "### Cybersecurity Question:",
        "### Cybersecurity Answer:",
        "<question>",
        "<answer>",
        "\nQuestion:",
        "\nAnswer:"
    ]

    for marker in stop_markers:
        if marker in answer:
            answer = answer.split(marker)[0].strip()

    if not answer:
        answer = (
            "I could not generate a clear answer. "
            "Please try asking the cybersecurity question again."
        )

    return answer, duration

# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model": MODEL_NAME,
        "device": device
    })

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({
            "answer": "Please enter a cybersecurity question.",
            "time_taken": 0.0
        })

    try:
        answer, duration = generate_answer(question)
        return jsonify({
            "answer": answer,
            "time_taken": duration,
            "device": device.upper()
        })
    except Exception as error:
        return jsonify({
            "answer": f"An error occurred while generating the answer: {str(error)}",
            "time_taken": 0.0
        }), 500

# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")

    print("=" * 60)
    print(f"Server accessible at: http://{host}:{port}")
    print("Press CTRL+C to terminate.")
    print("=" * 60 + "\n")

    app.run(
        host=host,
        port=port,
        debug=False
    )
