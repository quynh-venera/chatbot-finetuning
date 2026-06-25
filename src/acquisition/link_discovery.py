"""
link_discovery.py

Fallback link-discovery crawler used when a site has no sitemap.xml
(this is the case for venerian.space).

Strategy: breadth-first search starting from the site root, following
<a href="..."> links found in each page's HTML. Depth-limited and
page-capped to keep the crawl bounded. Domain-scoped via an explicit
allowlist (for legitimate extra subdomains, e.g. platform.venerian.space)
and a blocklist (for subdomains that require auth and have no public
content, e.g. app.venerian.space).
"""

import re
from collections import deque
from urllib.parse import urljoin, urlparse

import httpx

# --- Configuration -----------------------------------------------------

ROOT_DOMAIN = "venerian.space"

# Subdomains that are safe and useful to crawl in addition to the root.
ALLOWED_EXTRA_DOMAINS = {
    "platform.venerian.space",  # hosts /whitepaper
}

# Subdomains that are known to require auth / have no public content.
BLOCKED_DOMAINS = {
    "app.venerian.space",  # logged-in product, nothing public to crawl
}

MAX_DEPTH = 3
MAX_PAGES = 300

# Static asset path fragments / extensions to skip — these are never
# useful "content" pages and just waste crawl budget / pollute results.
STATIC_ASSET_PATTERNS = [
    r"/_next/static/",
    r"/assets/",
    r"\.(?:css|js|png|jpg|jpeg|gif|svg|webp|ico|woff2?|ttf|eot|map)(?:\?.*)?$",
]

# Malformed URLs that come from unresolved JS template literals in HTML,
# e.g. href="${baseUrl}/foo" leaking into static markup.
TEMPLATE_LITERAL_PATTERN = re.compile(r"\$\{.*?\}")

HEADERS = {
    "User-Agent": "VeneraBot/1.0 (+https://venerian.space; data acquisition for internal chatbot)"
}


def _is_static_asset(url: str) -> bool:
    return any(re.search(pat, url, re.IGNORECASE) for pat in STATIC_ASSET_PATTERNS)


def _is_malformed(url: str) -> bool:
    return bool(TEMPLATE_LITERAL_PATTERN.search(url))


def _domain_allowed(netloc: str) -> bool:
    netloc = netloc.lower()
    if netloc in BLOCKED_DOMAINS:
        return False
    if netloc == ROOT_DOMAIN or netloc == f"www.{ROOT_DOMAIN}":
        return True
    if netloc in ALLOWED_EXTRA_DOMAINS:
        return True
    return False


def _normalize_url(url: str) -> str:
    """Strip fragments and trailing slashes for dedup purposes."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _extract_links(html: str, base_url: str) -> list[str]:
    links = []
    for match in re.finditer(r'href=["\']([^"\']+)["\']', html, re.IGNORECASE):
        href = match.group(1).strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        if _is_malformed(href):
            continue
        absolute = urljoin(base_url, href)
        if _is_static_asset(absolute):
            continue
        parsed = urlparse(absolute)
        if parsed.scheme not in ("http", "https"):
            continue
        if not _domain_allowed(parsed.netloc):
            continue
        links.append(_normalize_url(absolute))
    return links


def discover_urls(seed_url: str = f"https://{ROOT_DOMAIN}/en") -> list[str]:
    """
    BFS crawl starting from seed_url, returning a deduplicated list of
    discovered page URLs across the root domain plus any allowlisted
    extra domains, respecting MAX_DEPTH and MAX_PAGES.
    """
    seen: set[str] = {_normalize_url(seed_url)}
    queue: deque[tuple[str, int]] = deque([(seed_url, 0)])
    discovered: list[str] = []

    with httpx.Client(headers=HEADERS, timeout=15.0, follow_redirects=True) as client:
        while queue and len(discovered) < MAX_PAGES:
            url, depth = queue.popleft()
            try:
                resp = client.get(url)
                resp.raise_for_status()
            except httpx.HTTPError:
                continue

            discovered.append(url)

            if depth >= MAX_DEPTH:
                continue

            for link in _extract_links(resp.text, url):
                if link not in seen:
                    seen.add(link)
                    queue.append((link, depth + 1))

    return discovered


if __name__ == "__main__":
    urls = discover_urls()
    print(f"Discovered {len(urls)} URLs:")
    for u in urls:
        print(f"  {u}")
