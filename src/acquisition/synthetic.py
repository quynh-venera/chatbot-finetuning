"""
synthetic.py

Stage 1: generate synthetic question/answer pairs from crawled page
content using an LLM. Currently OpenAI-only — Gemini support exists
but is gated behind USE_GEMINI so the google.generativeai package is
never imported or called while it's off (avoids dead-key warnings and
any chance of an accidental call against an unfunded Gemini key).
"""

import datetime
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --- Toggle ---------------------------------------------------------
# Flip to True only once a funded GEMINI_API_KEY / billing is confirmed.
USE_GEMINI = False

if USE_GEMINI:
    import google.generativeai as genai  # noqa: F401  (intentionally lazy/gated)
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

MAX_QUESTIONS_PER_PAGE = 5
MODEL_NAME = "gpt-4o"

SYSTEM_PROMPT = (
    "You are helping build a training dataset for a customer-support chatbot "
    "for Venera AI, a health technology company. Given the page content below, "
    "write up to {n} natural question-and-answer pairs a real user might ask "
    "about Venera AI's product, features, or documentation, answerable strictly "
    "from the given content. Do not invent medical advice or diagnoses — Venera AI "
    "provides information about how its platform works, not clinical guidance. "
    "Return ONLY a JSON array of objects with \"question\" and \"answer\" keys, "
    "no preamble, no markdown fences."
)

INPUT_DIR = Path("data/raw/crawled")
OUTPUT_DIR = Path("data/raw/synthetic")


def _latest_crawl_file() -> Path:
    files = sorted(INPUT_DIR.glob("crawl_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No crawl_*.jsonl files found in {INPUT_DIR}")
    return files[-1]


def generate_pairs_for_page(url: str, content: str) -> list[dict]:
    """Call GPT-4o to generate up to MAX_QUESTIONS_PER_PAGE Q&A pairs for one page."""
    prompt = SYSTEM_PROMPT.format(n=MAX_QUESTIONS_PER_PAGE)
    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content[:8000]},  # guard against huge pages
            ],
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()
        pairs = json.loads(raw)
        for p in pairs:
            p["source_url"] = url
        return pairs[:MAX_QUESTIONS_PER_PAGE]
    except (json.JSONDecodeError, Exception) as e:  # noqa: BLE001
        print(f"[synthetic] Failed to generate/parse pairs for {url}: {e}")
        return []


def run_synthetic() -> Path:
    """
    Read the latest crawl file, generate Q&A pairs per page, and write
    them to data/raw/synthetic/synthetic_<date>.jsonl.
    """
    crawl_file = _latest_crawl_file()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    today = datetime.date.today().isoformat()
    out_path = OUTPUT_DIR / f"synthetic_{today}.jsonl"

    total_pairs = 0
    with crawl_file.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            record = json.loads(line)
            pairs = generate_pairs_for_page(record["url"], record["content"])
            for pair in pairs:
                fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
                total_pairs += 1

    print(f"[synthetic] Generated {total_pairs} Q&A pairs from {crawl_file.name}.")
    print(f"[synthetic] Output: {out_path}")
    print(f"[synthetic] USE_GEMINI={USE_GEMINI} (OpenAI-only run)" if not USE_GEMINI
          else "[synthetic] USE_GEMINI=True")
    return out_path


if __name__ == "__main__":
    run_synthetic()
