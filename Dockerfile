FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOST=0.0.0.0

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt .

RUN pip install --no-cache-dir --user --upgrade pip && \
    pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

CMD ["python", "app.py"]
