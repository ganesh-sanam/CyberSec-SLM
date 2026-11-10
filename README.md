---
title: CyberSec-SLM
emoji: 🛡️
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 7860
pinned: false
---

# 🛡️ CyberSec-SLM

Cybersecurity Small Language Model web application powered by fine-tuned **Qwen2.5-0.5B-Instruct** using LoRA (PEFT).

## 🚀 How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
python app.py
```
Open **`http://localhost:7860`** in your browser.

## 🐳 Docker / Hugging Face Spaces

This repository is ready to deploy directly to **Hugging Face Spaces** (Docker SDK) or run with standard Docker:

```bash
docker build -t cybersec-slm .
docker run -p 7860:7860 cybersec-slm
```
