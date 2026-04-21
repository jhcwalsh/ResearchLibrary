"""
Find papers in Zotero that have a URL or DOI but no PDF attachment,
then attempt to download the PDF and attach it.

Strategies tried (in order):
  1. Direct URL — if the URL already points to a .pdf file
  2. Unpaywall API — finds open-access PDFs by DOI (free, no key needed)
  3. DOI landing page scrape — follows doi.org redirect and looks for a PDF link
  4. Original URL scrape — fetches the URL and looks for a PDF link
"""
import re
import tempfile
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

from .zotero_client import get_client, get_attachment_key, get_all_papers

UNPAYWALL_EMAIL = "research@example.com"
REQUEST_TIMEOUT = 20
RETRY_DELAY = 1.0
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ResearchLibraryBot/1.0; +research)"
}


def _lazy_imports():
    """Import requests/bs4 only when actually needed so the base install is light."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise ImportError(
            "download_attachments requires the 'download' extras: "
            "pip install -e \".[download]\""
        ) from e
    return requests, BeautifulSoup


def _is_pdf_url(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


def _looks_like_pdf(response) -> bool:
    ct = response.headers.get("Content-Type", "")
    return "application/pdf" in ct or "octet-stream" in ct


def _extract_pdf_links(html: str, base_url: str) -> list[str]:
    _, BeautifulSoup = _lazy_imports()
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for tag in soup.find_all(["a", "iframe", "embed"], href=True):
        href = tag.get("href") or tag.get("src") or ""
        if ".pdf" in href.lower():
            links.append(urljoin(base_url, href))
    for tag in soup.find_all("meta", attrs={"name": re.compile("citation_pdf_url", re.I)}):
        content = tag.get("content", "")
        if content:
            links.append(content)
    return links


def try_direct_url(url: str) -> bytes | None:
    requests, _ = _lazy_imports()
    if not _is_pdf_url(url):
        return None
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if r.ok and _looks_like_pdf(r):
            return r.content
    except Exception:
        pass
    return None


def try_unpaywall(doi: str) -> bytes | None:
    requests, _ = _lazy_imports()
    if not doi:
        return None
    doi = doi.strip()
    api_url = f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}"
    try:
        r = requests.get(api_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if not r.ok:
            return None
        data = r.json()
        best = data.get("best_oa_location") or {}
        pdf_url = best.get("url_for_pdf")
        if not pdf_url:
            for loc in data.get("oa_locations", []):
                if loc.get("url_for_pdf"):
                    pdf_url = loc["url_for_pdf"]
                    break
        if not pdf_url:
            return None
        pdf_r = requests.get(pdf_url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if pdf_r.ok and _looks_like_pdf(pdf_r):
            return pdf_r.content
    except Exception:
        pass
    return None


def try_scrape_for_pdf(url: str) -> bytes | None:
    requests, _ = _lazy_imports()
    if not url:
        return None
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if not r.ok:
            return None
        if _looks_like_pdf(r):
            return r.content
        pdf_links = _extract_pdf_links(r.text, r.url)
        for link in pdf_links[:3]:
            try:
                pdf_r = requests.get(link, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
                if pdf_r.ok and _looks_like_pdf(pdf_r):
                    return pdf_r.content
            except Exception:
                continue
    except Exception:
        pass
    return None


def fetch_pdf(doi: str, url: str) -> tuple[bytes | None, str]:
    if url:
        pdf = try_direct_url(url)
        if pdf:
            return pdf, "direct_url"
    if doi:
        pdf = try_unpaywall(doi)
        if pdf:
            return pdf, "unpaywall"
    if doi:
        pdf = try_scrape_for_pdf(f"https://doi.org/{doi}")
        if pdf:
            return pdf, "doi_scrape"
    if url and not _is_pdf_url(url):
        pdf = try_scrape_for_pdf(url)
        if pdf:
            return pdf, "url_scrape"
    return None, "failed"


def attach_pdf_to_item(zot, parent_key: str, title: str, pdf_bytes: bytes) -> bool:
    safe_title = re.sub(r'[\\/:*?"<>|]', "_", title)[:80]
    with tempfile.NamedTemporaryFile(suffix=".pdf", prefix=safe_title + "_", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)
    try:
        result = zot.attachment_simple([str(tmp_path)], parentid=parent_key)
        if result.get("successful"):
            return True
        return False
    except Exception:
        return False
    finally:
        tmp_path.unlink(missing_ok=True)


def download_missing(dry_run: bool = True, limit: int | None = None,
                     progress=None) -> dict:
    """Find papers without PDFs and try to download them. Returns summary dict."""
    def step(msg):
        if progress:
            progress(msg)

    zot = get_client()
    step("Fetching all library items...")
    papers = get_all_papers()
    candidates = [p for p in papers if (p.get("doi") or p.get("url"))]
    step(f"{len(candidates)} papers have a DOI or URL; checking for PDFs...")

    needs_pdf = []
    for i, p in enumerate(candidates):
        if get_attachment_key(p["key"]) is None:
            needs_pdf.append(p)
        time.sleep(0.05)
        if (i + 1) % 50 == 0:
            step(f"  checked {i+1}/{len(candidates)}")

    if limit is not None:
        needs_pdf = needs_pdf[:limit]

    step(f"{len(needs_pdf)} papers need a PDF attachment"
         + (" (dry-run; nothing will be uploaded)" if dry_run else ""))

    success = failed = 0
    for p in needs_pdf:
        label = f"{p['title'][:65]}{' (' + p['year'] + ')' if p['year'] else ''}"
        if dry_run:
            step(f"  [DRY] {label}")
            continue
        pdf_bytes, strategy = fetch_pdf(p.get("doi", ""), p.get("url", ""))
        if pdf_bytes is None:
            step(f"  FAIL {label}")
            failed += 1
        elif attach_pdf_to_item(zot, p["key"], p["title"], pdf_bytes):
            step(f"  OK   {label}  via {strategy}")
            success += 1
        else:
            step(f"  FAIL upload  {label}")
            failed += 1
        time.sleep(RETRY_DELAY)

    return {
        "candidates": len(candidates),
        "needed_pdf": len(needs_pdf),
        "attached": success,
        "failed": failed,
        "dry_run": dry_run,
    }
