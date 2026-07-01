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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
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
    print(f"[serve] Model loaded in {time.time() - t0:.1f}s")
    yield
    llm = None
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

    try:
        output = llm(
            prompt,
            max_tokens=max_tok,
            stop=["<|im_end|>", "<|im_start|>"],
            echo=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    answer = output["choices"][0]["text"].strip()
    tokens_used = output["usage"]["total_tokens"]

    return ChatResponse(
        question=request.question,
        answer=answer,
        model=HF_FILENAME,
        tokens_used=tokens_used,
    )
