"""
cpt_processor.py

Stage 2.5: Continued Pre-training (CPT) data prep.

Continued pre-training is a different technique from the Stage 2
instruction-tuning pipeline (processor.py): instead of labeled
chat-message Q&A triples, CPT trains a model on raw, unlabeled text so
it absorbs domain knowledge and phrasing — in this case, Venera AI's
own site content — before Stage 3's LoRA instruction fine-tuning
teaches it specific Q&A *behavior* on top of that.

Per Unsloth's docs (https://unsloth.ai/docs/basics/continued-pretraining),
CPT is normally used to steer a model toward a new domain or language
using large amounts of raw text. Our corpus here (26 crawled pages) is
small relative to typical CPT corpora, so this is explicitly a
*prototype/learning exercise* stage — useful for exercising the full
ML engineering pipeline end-to-end, not intended to produce a
production-grade domain adaptation on its own.

Input:  data/raw/crawled/crawl_<date>.jsonl  (Stage 1 output — raw
        Markdown page content, untouched by synthetic Q&A generation)
Output: data/processed/cpt/cpt_corpus.jsonl  — chunked raw-text records
        in the {"text": "..."} format Unsloth's text-completion /
        CPT notebooks expect.
"""

import datetime
import json
from pathlib import Path

import tiktoken

RAW_CRAWLED_DIR = Path("data/raw/crawled")
CPT_OUTPUT_DIR = Path("data/processed/cpt")

# Reuse the same chunk sizing as processor.py's Stage 2 chunker, since
# the underlying constraint (model context window) is the same. CPT
# chunks don't need the overlap that helps Q&A retrieval-style chunking,
# but a little overlap doesn't hurt and keeps the two pipelines aligned.
CHUNK_TOKEN_SIZE = 512
CHUNK_OVERLAP_TOKENS = 64

# Mirrors processor.py's MIN_ANSWER_CHARS junk-filtering logic, applied
# here to whole pages instead of individual Q&A answers — same problem
# (40-char 404 pages), same fix.
MIN_PAGE_CHARS = 100

# URL patterns to exclude from the CPT corpus:
#   - /vi/ locale pages: Vietnamese-locale content. Including these
#     would teach the model Vietnamese phrasing as part of "Venera AI's
#     domain language" without that being a deliberate choice — exclude
#     until a multilingual strategy is decided on purpose.
#   - /privacy, /terms: generic legal/ToS boilerplate that's nearly
#     identical across most company websites. Low-value CPT signal —
#     doesn't teach anything distinctively "Venera AI."
#
# Matched against URL path *segments*, not raw substrings — e.g. "/vi"
# must be a path segment, not just any place "vi" appears, and a bare
# locale root like "https://venerian.space/vi" (no trailing slash or
# further path) must match just as reliably as "/vi/posts/...".
EXCLUDED_PATH_SEGMENTS = {"vi", "privacy", "terms"}

_encoder = tiktoken.get_encoding("cl100k_base")


def _latest_crawl_file() -> Path:
    files = sorted(RAW_CRAWLED_DIR.glob("crawl_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No crawl_*.jsonl files found in {RAW_CRAWLED_DIR}")
    return files[-1]


def _is_excluded_url(url: str) -> bool:
    """True if any path segment of the URL matches an excluded segment
    (e.g. 'vi', 'privacy', 'terms'), regardless of trailing slashes or
    where in the path it appears."""
    from urllib.parse import urlparse
    path = urlparse(url).path
    segments = [seg for seg in path.split("/") if seg]
    return any(seg.lower() in EXCLUDED_PATH_SEGMENTS for seg in segments)


def _clean_text(text: str) -> str:
    """Same whitespace cleanup as processor.py — collapse blank lines,
    strip trailing whitespace per line."""
    text = text.replace("\r\n", "\n")
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _chunk_text(text: str, size: int = CHUNK_TOKEN_SIZE, overlap: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    """Token-count chunking with overlap. Identical approach to
    processor.py's chunker — kept here as its own copy rather than a
    shared import, since CPT and Stage 2 are deliberately separate
    pipelines that may diverge later (e.g. different chunk sizes)."""
    tokens = _encoder.encode(text)
    if len(tokens) <= size:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(_encoder.decode(chunk_tokens))
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


def run_cpt_processing() -> dict:
    """
    Read the latest crawl file, clean + chunk each page's raw content,
    and write a CPT-formatted JSONL corpus: one {"text": "..."} record
    per chunk. Returns a stats dict.
    """
    crawl_file = _latest_crawl_file()
    CPT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    today = datetime.date.today().isoformat()
    out_path = CPT_OUTPUT_DIR / "cpt_corpus.jsonl"

    pages_read = 0
    pages_filtered_url = 0
    pages_filtered_min_chars = 0
    chunks_written = 0
    total_tokens = 0

    with crawl_file.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            pages_read += 1

            url = record.get("url", "")
            if _is_excluded_url(url):
                pages_filtered_url += 1
                continue

            cleaned = _clean_text(record.get("content", ""))

            if len(cleaned) < MIN_PAGE_CHARS:
                pages_filtered_min_chars += 1
                continue

            for chunk in _chunk_text(cleaned):
                fout.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")
                chunks_written += 1
                total_tokens += len(_encoder.encode(chunk))

    stats = {
        "date": today,
        "source_crawl_file": crawl_file.name,
        "pages_read": pages_read,
        "pages_filtered_url_pattern": pages_filtered_url,
        "excluded_path_segments": sorted(EXCLUDED_PATH_SEGMENTS),
        "pages_filtered_min_chars": pages_filtered_min_chars,
        "min_page_chars_threshold": MIN_PAGE_CHARS,
        "chunks_written": chunks_written,
        "approx_total_tokens": total_tokens,
    }

    stats_path = CPT_OUTPUT_DIR / "cpt_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"[cpt_processor] {stats}")
    print(f"[cpt_processor] Output: {out_path}")
    print(f"[cpt_processor] Stats: {stats_path}")

    if chunks_written < 50:
        print(
            "[cpt_processor] NOTE: this is a small corpus for continued "
            "pre-training (typical CPT corpora are much larger — hundreds "
            "of thousands of tokens or more). Treat this as a pipeline "
            "prototype/learning exercise, not a production domain adaptation."
        )

    return stats


if __name__ == "__main__":
    run_cpt_processing()
