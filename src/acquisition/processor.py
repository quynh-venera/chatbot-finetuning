"""
processor.py

Stage 2: data processing. Takes the raw crawled pages and raw synthetic
Q&A pairs, cleans/chunks the crawl content, formats everything into
chat-template JSONL records, deduplicates, and writes a stratified
80/10/10 train/val/test split plus stats.json.
"""

import datetime
import json
import random
from pathlib import Path

import tiktoken

RAW_CRAWLED_DIR = Path("data/raw/crawled")
RAW_SYNTHETIC_DIR = Path("data/raw/synthetic")
PROCESSED_DIR = Path("data/processed")

# Pages that return ~40-char 404 bodies were slipping through the old
# threshold of 30. Raised to 100 to filter that junk out.
MIN_ANSWER_CHARS = 100

CHUNK_TOKEN_SIZE = 512
CHUNK_OVERLAP_TOKENS = 64

SYSTEM_PROMPT = (
    "You are a helpful assistant for Venera AI, a health technology company. "
    "Answer questions accurately and concisely based on Venera AI's products, "
    "features, and documentation. If you don't know the answer, say so clearly. "
    "You do not provide medical advice or diagnoses — only information about how "
    "Venera AI's platform works."
)

SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train, val, test
RANDOM_SEED = 42

_encoder = tiktoken.get_encoding("cl100k_base")


def _latest_file(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    return files[-1]


def _clean_text(text: str) -> str:
    """Basic whitespace/markup cleanup on crawled page content."""
    text = text.replace("\r\n", "\n")
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _chunk_text(text: str, size: int = CHUNK_TOKEN_SIZE, overlap: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    """Sentence-aware-ish chunking by token count with overlap, for any
    crawl content used directly (synthetic Q&A pairs are already short
    and don't need chunking)."""
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


def _load_synthetic_pairs() -> list[dict]:
    path = _latest_file(RAW_SYNTHETIC_DIR, "synthetic_*.jsonl")
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def _to_chat_record(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def _dedup(records: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for r in records:
        key = (r["messages"][1]["content"].strip().lower(),
               r["messages"][2]["content"].strip().lower())
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


def _stratified_split(records: list[dict], ratios=SPLIT_RATIOS, seed=RANDOM_SEED):
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def run_processing() -> dict:
    """
    Full Stage 2 pipeline: clean -> chunk (informational, for any
    long-form content) -> format -> dedup -> split. Writes
    train/val/test.jsonl, all_with_meta.jsonl, and stats.json.
    Returns the stats dict.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_pairs = _load_synthetic_pairs()

    filtered_out = 0
    all_records = []
    all_with_meta = []

    for pair in raw_pairs:
        question = _clean_text(pair.get("question", "")).strip()
        answer = _clean_text(pair.get("answer", "")).strip()

        if len(answer) < MIN_ANSWER_CHARS:
            filtered_out += 1
            continue
        if not question or not answer:
            filtered_out += 1
            continue

        record = _to_chat_record(question, answer)
        all_records.append(record)
        all_with_meta.append({**record, "source_url": pair.get("source_url", "")})

    deduped = _dedup(all_records)
    duplicates_removed = len(all_records) - len(deduped)

    train, val, test = _stratified_split(deduped)

    today = datetime.date.today().isoformat()

    def _write_jsonl(path: Path, records: list[dict]):
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _write_jsonl(PROCESSED_DIR / "train.jsonl", train)
    _write_jsonl(PROCESSED_DIR / "val.jsonl", val)
    _write_jsonl(PROCESSED_DIR / "test.jsonl", test)
    _write_jsonl(PROCESSED_DIR / "all_with_meta.jsonl", all_with_meta)

    stats = {
        "date": today,
        "raw_pairs": len(raw_pairs),
        "filtered_out_min_answer_chars": filtered_out,
        "min_answer_chars_threshold": MIN_ANSWER_CHARS,
        "duplicates_removed": duplicates_removed,
        "total": len(deduped),
        "train": len(train),
        "val": len(val),
        "test": len(test),
    }
    (PROCESSED_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"[processor] {stats}")
    return stats


if __name__ == "__main__":
    run_processing()
