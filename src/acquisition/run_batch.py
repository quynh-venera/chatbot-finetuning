"""
run_batch.py

Master runner for Stage 1 + Stage 2: crawl -> generate synthetic Q&A ->
process/split. Run as: python -m src.acquisition.run_batch
"""

import os
import sys

from dotenv import load_dotenv

from src.acquisition import crawler, synthetic, processor

load_dotenv()


def check_env() -> bool:
    """
    Validate required environment variables.
    VENERA_SITEMAP_URL is intentionally optional — an empty value
    triggers the link-discovery fallback in crawler.get_all_urls().
    """
    required = ["JINA_API_KEY", "OPENAI_API_KEY"]
    missing = [var for var in required if not os.environ.get(var)]

    if missing:
        print(f"[run_batch] Missing required env vars: {missing}")
        return False

    sitemap = os.environ.get("VENERA_SITEMAP_URL", "")
    if not sitemap:
        print("[run_batch] VENERA_SITEMAP_URL is empty — link-discovery "
              "fallback will be used (this is expected for venerian.space).")

    return True


def main():
    if not check_env():
        print("[run_batch] Aborting — fix environment variables and retry.")
        sys.exit(1)

    print("[run_batch] Stage 1a: crawling...")
    crawler.run_crawl()

    print("[run_batch] Stage 1b: generating synthetic Q&A pairs...")
    synthetic.run_synthetic()

    print("[run_batch] Stage 2: processing (clean/chunk/format/dedup/split)...")
    stats = processor.run_processing()

    # NOTE: previously this read stats['total_pairs'], which only existed
    # in the old merge.py's return dict. processor.run_processing() returns
    # 'total' instead — fixed here to match.
    print(
        f"[run_batch] Done. {stats['total']} total examples "
        f"({stats['train']} train / {stats['val']} val / {stats['test']} test)."
    )


if __name__ == "__main__":
    main()
