"""
serve/app.py

FastAPI inference server for the Venera AI fine-tuned GGUF model.

Loads the GGUF model via llama-cpp-python at startup. The model is
downloaded from a private HuggingFace repo on first run and cached
locally at MODEL_CACHE_PATH so subsequent restarts skip the download.

Environment variables (set in .env locally, Kubernetes Secret in GKE):
    HF_TOKEN            HuggingFace token with read access to the repo
    HF_REPO_ID          HuggingFace repo ID, e.g. "username/venera-chatbot-model"
    HF_FILENAME         GGUF filename in the repo (default: model-q4.gguf)
    MODEL_CACHE_PATH    Local path to cache the downloaded model (default: /app/model.gguf)
    N_CTX               Context length (default: 4096)
    N_THREADS           CPU threads for inference (default: 4)
    MAX_TOKENS          Max tokens to generate per response (default: 512)
"""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel

# --- Configuration -----------------------------------------------------------

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
HF_FILENAME = os.environ.get("HF_FILENAME", "model-q4.gguf")
MODEL_CACHE_PATH = os.environ.get("MODEL_CACHE_PATH", "/app/model.gguf")

N_CTX = int(os.environ.get("N_CTX", "4096"))
N_THREADS = int(os.environ.get("N_THREADS", "4"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))

SYSTEM_PROMPT = (
    "You are a helpful assistant for Venera AI, a health technology company. "
    "Answer questions accurately and concisely based on Venera AI's products, "
    "features, and documentation. If you don't know the answer, say so clearly. "
    "You do not provide medical advice or diagnoses — only information about how "
    "Venera AI's platform works."
)

# --- Prometheus metrics -------------------------------------------------------
# Scraped by the in-cluster Prometheus (see k8s/monitoring/) via the /metrics
# endpoint below. Pod annotations (prometheus.io/scrape=true) tell Prometheus'
# kubernetes_sd_configs to find this pod automatically — no manual target list.

REQUEST_COUNT = Counter(
    "chatbot_requests_total",
    "Total HTTP requests received",
    ["endpoint", "method", "status"],
)
REQUEST_LATENCY = Histogram(
    "chatbot_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60),
)
INFERENCE_ERRORS = Counter(
    "chatbot_inference_errors_total",
    "Total inference errors raised by the model",
)
TOKENS_GENERATED = Histogram(
    "chatbot_tokens_generated",
    "Tokens generated per /chat response",
    buckets=(8, 16, 32, 64, 128, 256, 512, 1024),
)
REQUESTS_IN_PROGRESS = Gauge(
    "chatbot_requests_in_progress",
    "Number of /chat requests currently being served",
)
MODEL_LOADED = Gauge(
    "chatbot_model_loaded",
    "1 if the GGUF model is loaded and ready, 0 otherwise",
)
MODEL_LOAD_SECONDS = Gauge(
    "chatbot_model_load_seconds",
    "How long the last model load (download + init) took, in seconds",
)

# --- Model loading -----------------------------------------------------------

llm: Llama | None = None


def _ensure_model() -> str:
    """
    Return the local path to the GGUF model, downloading from HuggingFace
    if not already cached. Skips download if the file already exists at
    MODEL_CACHE_PATH (e.g. on a pod restart with a persistent volume, or
    in a Docker image that baked the model in).
    """
    dest = Path(MODEL_CACHE_PATH)

    if dest.exists():
        print(f"[serve] Model already at {dest}, skipping download.")
        return str(dest)

    if not HF_REPO_ID:
        raise RuntimeError(
            "HF_REPO_ID environment variable is required when the model "
            "is not already present at MODEL_CACHE_PATH."
        )

    print(f"[serve] Downloading {HF_FILENAME} from {HF_REPO_ID}...")
    t0 = time.time()

    dest.parent.mkdir(parents=True, exist_ok=True)

    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        token=HF_TOKEN or None,
        local_dir=str(dest.parent),
        local_dir_use_symlinks=False,
    )

    # hf_hub_download saves to a subdirectory by default; move to MODEL_CACHE_PATH.
    downloaded_path = Path(downloaded)
    if downloaded_path != dest:
        downloaded_path.rename(dest)

    elapsed = time.time() - t0
    size_mb = dest.stat().st_size / 1024 / 1024
    print(f"[serve] Downloaded {size_mb:.1f} MB in {elapsed:.1f}s → {dest}")
    return str(dest)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Download (if needed) and load the model at startup; release at shutdown."""
    global llm

    model_path = _ensure_model()

    print(f"[serve] Loading model from {model_path}...")
    t0 = time.time()
    llm = Llama(
        model_path=model_path,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        verbose=False,
    )
    load_seconds = time.time() - t0
    MODEL_LOAD_SECONDS.set(load_seconds)
    MODEL_LOADED.set(1)
    print(f"[serve] Model loaded in {load_seconds:.1f}s")
    yield
    llm = None
    MODEL_LOADED.set(0)
    print("[serve] Model unloaded.")


# --- FastAPI app -------------------------------------------------------------

app = FastAPI(
    title="Venera AI Chatbot API",
    description="Fine-tuned Qwen2.5-1.5B-Instruct model serving Venera AI product Q&A.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    """Record request count and latency for every endpoint except /metrics
    itself (scraping /metrics shouldn't inflate its own counters)."""
    if request.url.path == "/metrics":
        return await call_next(request)

    start = time.time()
    try:
        response = await call_next(request)
    except Exception:
        REQUEST_COUNT.labels(endpoint=request.url.path, method=request.method, status="500").inc()
        raise
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(time.time() - start)
    REQUEST_COUNT.labels(
        endpoint=request.url.path, method=request.method, status=str(response.status_code)
    ).inc()
    return response


# --- Request/response models -------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    max_tokens: int | None = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    model: str
    tokens_used: int


# --- Endpoints ---------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness probe — returns 200 if the server is up."""
    return {"status": "ok", "model_loaded": llm is not None}


@app.get("/ready")
def ready():
    """Readiness probe — returns 200 only when the model is fully loaded."""
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not yet loaded.")
    return {"status": "ready"}


@app.get("/metrics")
def metrics():
    """Prometheus scrape target. Kept unauthenticated — the k8s Service is
    ClusterIP-only, so this is only reachable from inside the cluster."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Accept a question, run inference via the fine-tuned GGUF model,
    and return the answer.
    """
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    # ChatML format matching the fine-tuning template.
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{request.question.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    max_tok = request.max_tokens or MAX_TOKENS

    REQUESTS_IN_PROGRESS.inc()
    try:
        output = llm(
            prompt,
            max_tokens=max_tok,
            stop=["<|im_end|>", "<|im_start|>"],
            echo=False,
        )
    except Exception as e:
        INFERENCE_ERRORS.inc()
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    finally:
        REQUESTS_IN_PROGRESS.dec()

    answer = output["choices"][0]["text"].strip()
    tokens_used = output["usage"]["total_tokens"]
    TOKENS_GENERATED.observe(tokens_used)

    return ChatResponse(
        question=request.question,
        answer=answer,
        model=HF_FILENAME,
        tokens_used=tokens_used,
    )
# Updated
