"""
crawler.py

Stage 1 crawler. Discovers URLs (sitemap if available, otherwise falls
back to link_discovery.py's BFS crawler), then fetches each page's
content via the Jina Reader API and writes results to
data/raw/crawled/crawl_<date>.jsonl.
"""

import datetime
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

from src.acquisition.link_discovery import discover_urls

load_dotenv()

JINA_API_KEY = os.environ.get("JINA_API_KEY", "")
VENERA_SITEMAP_URL = os.environ.get("VENERA_SITEMAP_URL", "").strip()

JINA_BASE = "https://r.jina.ai/"

OUTPUT_DIR = Path("data/raw/crawled")
MANIFEST_PATH = Path("data/manifest.json")


def _try_sitemap(sitemap_url: str) -> list[str]:
    """Attempt to fetch and parse a sitemap.xml. Returns [] on any failure."""
    if not sitemap_url:
        return []
    try:
        resp = httpx.get(sitemap_url, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError:
        return []

    # Minimal sitemap <loc> extraction — good enough for standard sitemaps.
    import re

    locs = re.findall(r"<loc>(.*?)</loc>", resp.text)
    return [loc.strip() for loc in locs if loc.strip()]


def get_all_urls() -> list[str]:
    """
    Try the sitemap first. If VENERA_SITEMAP_URL is empty, unreachable,
    or returns zero URLs, fall back to link-discovery BFS crawling.
    """
    sitemap_urls = _try_sitemap(VENERA_SITEMAP_URL)
    if sitemap_urls:
        print(f"[crawler] Using sitemap: {len(sitemap_urls)} URLs found.")
        return sitemap_urls

    print("[crawler] No usable sitemap (empty/unreachable/zero URLs). "
          "Falling back to link discovery.")
    return discover_urls()


def fetch_via_jina(url: str, client: httpx.Client) -> str | None:
    """Fetch a page's readable content through the Jina Reader API."""
    jina_url = f"{JINA_BASE}{url}"
    headers = {"Authorization": f"Bearer {JINA_API_KEY}"} if JINA_API_KEY else {}
    try:
        resp = client.get(jina_url, headers=headers, timeout=30.0)
        resp.raise_for_status()
        return resp.text
    except httpx.HTTPError as e:
        print(f"[crawler] Failed to fetch {url} via Jina: {e}")
        return None


def run_crawl() -> Path:
    """
    Discover URLs, fetch each via Jina, and write a JSONL file of
    {url, content} records. Returns the path to the written file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)

    urls = get_all_urls()
    today = datetime.date.today().isoformat()
    out_path = OUTPUT_DIR / f"crawl_{today}.jsonl"

    manifest = {}
    records_written = 0
    failures = 0

    with httpx.Client() as client, out_path.open("w", encoding="utf-8") as f:
        for url in urls:
            content = fetch_via_jina(url, client)
            if content is None:
                failures += 1
                continue

            record = {"url": url, "content": content}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_written += 1

            import hashlib
            manifest[url] = hashlib.sha256(content.encode("utf-8")).hexdigest()

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[crawler] Done. {records_written} pages written, {failures} failures.")
    print(f"[crawler] Output: {out_path}")
    print(f"[crawler] Manifest: {MANIFEST_PATH}")
    return out_path


if __name__ == "__main__":
    run_crawl()
