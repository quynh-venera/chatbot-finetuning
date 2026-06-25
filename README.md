# chatbot-finetuning

Data acquisition, processing, and fine-tuning pipeline for the **Venera AI** chatbot (Venera AI is a health technology company). Built in 9 stages, from data acquisition through automated retraining.

> Companion repo: `Venera-AI/venera-chatbot`.

## Status

| # | Stage | Status |
|---|---|---|
| 1 | Data acquisition (crawl + synthetic + DVC versioning) | Done |
| 2 | Data processing (clean, chunk, chat template, split) | Done |
| 3 | Fine-tuning (Unsloth Studio GUI, Qwen2.5-1.5B) | Next |
| 4 | Evaluation (MCQ benchmark, retrieval metrics) | Pending |
| 5 | Optimization (quantization via llama.cpp → GGUF) | Pending |
| 6 | Deployment (CI/CD, Docker, Azure VM, GKE, Kubernetes) | Pending |
| 7 | Monitoring & stress testing | Pending |
| 8 | Agentic AI layer (RAG, search tool, MCP) | Pending |
| 9 | Automated retraining loop | Pending |

## Setup

```bash
# Conda env (Miniforge, native arm64 on Apple Silicon)
conda create -n venera python=3.11
conda activate venera
pip install -r requirements.txt

cp .env.example .env
# then fill in JINA_API_KEY, OPENAI_API_KEY, etc.
```

## Stage 1 + 2: run the full acquisition + processing batch

```bash
python -m src.acquisition.run_batch
```

This will:
1. Discover URLs on `venerian.space` (no sitemap — uses link-discovery BFS fallback)
2. Fetch each page's content via the Jina Reader API
3. Generate synthetic Q&A pairs via GPT-4o
4. Clean, chunk, format, deduplicate, and split into train/val/test JSONL

Outputs land in `data/raw/` and `data/processed/`.

## Data versioning (DVC)

```bash
dvc repro          # re-run the pipeline if deps changed
dvc commit -f      # commit pipeline outputs (needed, not plain `dvc add`)
git add -A && git commit -m "..."
git tag data-v1.0
git push && git push --tags
dvc push           # currently blocked — Azure connection string not yet configured
```

See the handoff doc for the exact steps to finish the Azure DVC remote setup.

## Project structure

```
chatbot-finetuning/
├── .env.example
├── .gitignore
├── .dvc/config
├── dvc.yaml
├── requirements.txt
├── logs/
├── data/
│   ├── manifest.json
│   ├── raw/{crawled,synthetic}/
│   └── processed/{train,val,test}.jsonl, all_with_meta.jsonl, stats.json
├── src/acquisition/
│   ├── crawler.py
│   ├── link_discovery.py
│   ├── synthetic.py
│   ├── processor.py
│   ├── merge.py          (legacy, unused — kept for reference)
│   └── run_batch.py
└── .github/workflows/data_acquisition.yml
```
