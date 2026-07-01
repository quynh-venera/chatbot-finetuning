# chatbot-finetuning

Data acquisition, processing, and fine-tuning pipeline for the **Venera AI** chatbot (Venera AI is a health technology company). Built in 9 stages, from data acquisition through automated retraining.

> Companion repo: `Venera-AI/venera-chatbot`.

## Status

| # | Stage | Status |
|---|---|---|
| 1 | Data acquisition (crawl + synthetic + DVC versioning) | Done |
| 2 | Data processing (clean, chunk, chat template, split) | Done |
| 2.5 | Continued pre-training (CPT) data prep — *prototype/learning addition, not in original 9-stage plan* | Done |
| 3 | Fine-tuning (Unsloth Studio GUI, Qwen2.5-1.5B) — CPT then LoRA instruction tuning | Next |
| 4 | Evaluation (MCQ benchmark, 11/13 raw, 11/12 adjusted) | Done |
| 5 | Optimization (GGUF Q4_K_M export via Unsloth Studio) | Done |
| 6 | Deployment (FastAPI + Docker + GKE + GitHub Actions CI/CD) | Done |
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

## Stage 2.5: Continued pre-training (CPT) data prep

This is an addition beyond the original 9-stage plan, included to prototype the
full ML engineering process (CPT + instruction fine-tuning) rather than only
instruction fine-tuning alone. It is **not** required for a working chatbot —
Stage 3's LoRA fine-tuning on `train.jsonl`/`val.jsonl` works fine on its own.

CPT trains a model on raw, unlabeled text so it absorbs domain phrasing/knowledge,
as distinct from Stage 2's labeled Q&A pairs which teach specific behavior. See
[Unsloth's CPT docs](https://unsloth.ai/docs/basics/continued-pretraining) for
the underlying technique.

```bash
python -m src.pretraining.cpt_processor
```

Reads the latest `data/raw/crawled/crawl_*.jsonl`, cleans and chunks each page's
raw content (same chunking approach as Stage 2), and writes:

- `data/processed/cpt/cpt_corpus.jsonl` — `{"text": "..."}` records, the format
  Unsloth's text-completion/CPT notebooks expect
- `data/processed/cpt/cpt_stats.json` — page/chunk/token counts

**Caveat:** our corpus (~26 crawled pages) is small relative to typical CPT
corpora (hundreds of thousands of tokens or more). Treat this stage as a
pipeline prototype/learning exercise, not a production domain adaptation.

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
├── preview_data.py
├── logs/
├── data/
│   ├── manifest.json
│   ├── raw/{crawled,synthetic}/
│   └── processed/
│       ├── {train,val,test}.jsonl, all_with_meta.jsonl, stats.json
│       └── cpt/cpt_corpus.jsonl, cpt_stats.json   (Stage 2.5)
├── src/
│   ├── acquisition/
│   │   ├── crawler.py
│   │   ├── link_discovery.py
│   │   ├── synthetic.py
│   │   ├── processor.py
│   │   ├── merge.py          (legacy, unused — kept for reference)
│   │   └── run_batch.py
│   └── pretraining/
│       └── cpt_processor.py  (Stage 2.5 — CPT data prep)
└── .github/workflows/data_acquisition.yml
```
