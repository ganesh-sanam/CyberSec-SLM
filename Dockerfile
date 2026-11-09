# ============================================================
# CyberSec-SLM Hugging Face Spaces & Production Dockerfile
# Fully compliant with Hugging Face non-root UID 1000 specs
# ============================================================

FROM python:3.10-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install basic system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user with UID 1000 (Required by Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Switch to non-root user
USER user

# Set user environment paths
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app \
    PORT=7860 \
    HOST=0.0.0.0

# Set working directory inside user home
WORKDIR $HOME/app

# Copy requirements file with user ownership
COPY --chown=user requirements.txt $HOME/app/requirements.txt

# Install packages as the user into /home/user/.local
# 1. Upgrade pip
# 2. Install lightweight CPU PyTorch (~180MB vs 2.5GB CUDA) for ultra-fast build
# 3. Install remaining requirements (flask, transformers, peft, etc.)
RUN pip install --no-cache-dir --user --upgrade pip && \
    pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --user -r $HOME/app/requirements.txt

# Copy all application files with user ownership
COPY --chown=user . $HOME/app

# Expose standard Hugging Face Spaces port
EXPOSE 7860

# Start server
CMD ["python", "app.py"]
