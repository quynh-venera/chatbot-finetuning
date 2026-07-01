# Dockerfile
#
# Builds the Venera AI chatbot FastAPI inference server.
# The GGUF model is downloaded from HuggingFace at container startup
# (not baked in), keeping the image small (~100MB vs ~1GB).
#
# Build:
#   docker build -t venera-chatbot:latest .
#
# Run locally (model will be downloaded on first start):
#   docker run -p 8000:8000 \
#     -e HF_TOKEN=hf_... \
#     -e HF_REPO_ID=your-username/venera-chatbot-model \
#     venera-chatbot:latest
#
# Run locally with cached model (skips download):
#   docker run -p 8000:8000 \
#     -e HF_TOKEN=hf_... \
#     -e HF_REPO_ID=your-username/venera-chatbot-model \
#     -v /path/to/cached/model.gguf:/app/model.gguf \
#     venera-chatbot:latest

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY serve/requirements-serve.txt ./requirements-serve.txt
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY serve/ ./serve/

RUN useradd -m -u 1000 venera && chown -R venera:venera /app
USER venera

ENV MODEL_CACHE_PATH=/app/model.gguf
ENV N_CTX=4096
ENV N_THREADS=4
ENV MAX_TOKENS=512

EXPOSE 8000

CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
