---
title: CyberSec-SLM
emoji: 🛡️
colorFrom: blue
colorTo: cyan
sdk: gradio
app_file: app.py
pinned: false
---

# 🛡️ CyberSec-SLM: Cybersecurity Small Language Model

An edge-deployable, domain-adapted **Small Language Model (SLM)** chatbot engineered for cybersecurity defense, threat analysis, and secure code guidance.

---

## 🌟 Highlights

* **Model Foundation**: [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (0.5B parameters)
* **Adaptation Technique**: Low-Rank Adaptation (**LoRA**) via Hugging Face PEFT ($r=4, \alpha=8$)
* **Target Attention Projections**: `q_proj`, `k_proj`, `v_proj`, `o_proj`
* **Resource Footprint**: ~1 GB RAM footprint; runs smoothly on standard CPUs as well as CUDA GPUs.
* **Interface**: Interactive Gradio Chat Interface.

---

## 🚀 How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Gradio app
python app.py
```
Your terminal will display a local link (e.g. `http://127.0.0.1:7860`). Open it in your browser!

---

## 📁 Project Structure

```
CyberSec-SLM/
│
├── model/
│   └── cybersec-slm/          # Fine-tuned LoRA weights & tokenizer configs
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── chat_template.jinja
│       ├── tokenizer.json
│       └── tokenizer_config.json
│
├── app.py                     # Gradio chat application & model loader
├── requirements.txt           # Python library dependencies
├── README.md                  # Hugging Face Spaces metadata & docs
└── .gitignore                 # Git ignore rules
```
