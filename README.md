---
title: CyberSec-SLM
emoji: 🛡️
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 7860
pinned: false
---

# 🛡️ CyberSec-SLM: Cybersecurity Small Language Model

An edge-deployable, domain-adapted **Small Language Model (SLM)** and responsive web assistant engineered for cybersecurity defense, threat analysis, and secure code guidance.

---

## 🌟 Highlights

* **Model Foundation**: [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (0.5B parameters)
* **Adaptation Technique**: Low-Rank Adaptation (**LoRA**) via Hugging Face PEFT ($r=4, \alpha=8$)
* **Target Attention Projections**: `q_proj`, `k_proj`, `v_proj`, `o_proj`
* **Resource Footprint**: ~1 GB RAM footprint; runs smoothly on standard CPUs as well as CUDA GPUs.
* **Full-Stack Implementation**: Flask REST API backend + responsive dark-mode web dashboard.

---

## 📁 Repository Structure

```
CyberSec-SLM/
│
├── app/
│   ├── app.py                     # Flask inference API & model server
│   └── templates/
│       └── index.html             # Responsive dark-mode dashboard
│
├── dataset/
│   └── cybersecurity_qa.jsonl     # Domain-specific Q&A training dataset
│
├── training/
│   └── train_lora.py              # Full LoRA fine-tuning pipeline script
│
├── model/
│   ├── cybersec-slm/              # Production LoRA adapter & tokenizer configs
│   └── cybersec-slm-test/         # Experimental checkpoint
│
├── test_inference.py              # Terminal CLI inference test script
├── requirements.txt               # Python package dependencies
├── Dockerfile                     # Multi-stage production container
├── docker-compose.yml             # Container orchestration
├── run.bat                        # Windows 1-click double-click launcher
├── run.ps1                        # PowerShell launcher
└── README.md                      # Documentation & HF Space metadata
```

---

## 🚀 Quickstart: Running Locally

### 1. Using the 1-Click Launcher (Windows)
Simply double-click **`run.bat`**. It will start the server and automatically open **`http://localhost:5000`** in your browser.

### 2. Manual Terminal Start
```bash
# Install dependencies
pip install -r requirements.txt

# Start Web Server
python app/app.py
```
Visit **`http://localhost:5000`** in your browser.

### 3. Quick CLI Test
Test the model directly from your terminal without opening a browser:
```bash
python test_inference.py --question "What is a Zero-Day vulnerability?"
```

---

## 🐳 Docker Deployment

To build and run with Docker:
```bash
# Using Docker Compose
docker-compose up --build

# Or standard Docker
docker build -t cybersec-slm .
docker run -p 5000:7860 cybersec-slm
```
Navigate to **`http://localhost:5000`**.

---

## ☁️ Free Cloud Hosting Options

### Option A: Hugging Face Spaces (Recommended for Portfolios)
1. Create a free account at [huggingface.co](https://huggingface.co).
2. Click **New Space** &rarr; Name it `cybersec-slm` &rarr; Select **Docker**.
3. Push this repository to your Space. The included `README.md` frontmatter and `Dockerfile` will automatically build and host the model on **16 GB RAM / 2 vCPUs** for free!

### Option B: Instant Public URL via Cloudflare Tunnel
Share your local running instance with an interviewer anywhere in the world:
```bash
# Terminal 1: Run the app
python app/app.py

# Terminal 2: Expose via Cloudflare Tunnel
npx untun@latest tunnel --port 5000
```
Copy the generated `https://...trycloudflare.com` URL.

---

## 🧠 Model Training & Dataset

* **Dataset**: Located in [`dataset/cybersecurity_qa.jsonl`](dataset/cybersecurity_qa.jsonl), containing structured cybersecurity vulnerability, mitigation, and cryptographic concept pairs.
* **Fine-Tuning Script**: Located in [`training/train_lora.py`](training/train_lora.py). Run `python training/train_lora.py` to retrain or fine-tune further on custom threat intelligence feeds.

---

## 💼 Internship & Interview Highlights

* **Edge vs. Cloud**: Explains how 0.5B SLMs enable on-premise, zero-data-leakage security triage without costly 70B parameter API dependencies.
* **Parameter Efficiency**: Demonstrates working knowledge of PEFT/LoRA, freezing 99%+ of base weights while training rank-4 adapters.
* **End-to-End Delivery**: Shows mastery across the entire ML lifecycle: dataset creation, fine-tuning, REST API engineering, containerization, and frontend UX.
