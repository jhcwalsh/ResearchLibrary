"""
Find papers in Zotero that have a URL or DOI but no PDF attachment,
then attempt to download the PDF and attach it.

Strategies tried (in order):
  1. Direct URL — if the URL already points to a .pdf file
  2. Unpaywall API — finds open-access PDFs by DOI (free, no key needed)
  3. DOI landing page scrape — follows doi.org redirect and looks for a PDF link
  4. Original URL scrape — fetches the URL and looks for a PDF link

Requirements:
    pip install requests beautifulsoup4

Usage:
    python download_attachments.py              # dry-run: show what would be downloaded
    python download_attachments.py --download   # actually download and attach
    python download_attachments.py --limit 20   # process at most 20 papers
    python download_attachments.py --download --limit 5
"""
import argparse
import re
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

from dotenv import load_dotenv

load_dotenv(override=True)

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Missing dependencies. Run:  pip install requests beautifulsoup4")
    sys.exit(1)

from src.research_library.zotero_client import get_client, get_attachment_key

# ── Config ─────────────────────────────────────────────────────────────────────
UNPAYWALL_EMAIL = "research@example.com"   # Unpaywall requires a contact email
REQUEST_TIMEOUT = 20
RETRY_DELAY = 1.0   # seconds between Zotero API calls

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ResearchLibraryBot/1.0; +research)"
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_pdf_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(".pdf")


def _looks_like_pdf(response: requests.Response) -> bool:
    ct = response.headers.get("Content-Type", "")
    return "application/pdf" in ct or "octet-stream" in ct


def _extract_pdf_links(html: str, base_url: str) -> list[str]:
    """Return all href/src values that look like PDF links."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for tag in soup.find_all(["a", "iframe", "embed"], href=True):
        href = tag.get("href") or tag.get("src") or ""
        if ".pdf" in href.lower():
            links.append(urljoin(base_url, href))
    # also check meta refresh / og:url style tags
    for tag in soup.find_all("meta", attrs={"name": re.compile("citation_pdf_url", re.I)}):
        content = tag.get("content", "")
        if content:
            links.append(content)
    return links


def try_direct_url(url: str) -> bytes | None:
    """Download directly if URL is a PDF."""
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
    """Use Unpaywall to find an open-access PDF by DOI."""
    if not doi:
        return None
    doi = doi.strip()
    api_url = f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}"
    try:
        r = requests.get(api_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if not r.ok:
            return None
        data = r.json()
        pdf_url = None
        # Prefer best_oa_location
        best = data.get("best_oa_location") or {}
        pdf_url = best.get("url_for_pdf")
        if not pdf_url:
            # Try all OA locations
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
    """Fetch the page at url and try to find a PDF link to download."""
    if not url:
        return None
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if not r.ok:
            return None
        if _looks_like_pdf(r):
            return r.content
        pdf_links = _extract_pdf_links(r.text, r.url)
        for link in pdf_links[:3]:  # try first 3 candidates
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
    """
    Try all strategies to fetch a PDF.
    Returns (pdf_bytes, strategy_name) or (None, "failed").
    """
    # 1. Direct URL
    if url:
        pdf = try_direct_url(url)
        if pdf:
            return pdf, "direct_url"

    # 2. Unpaywall (needs DOI)
    if doi:
        pdf = try_unpaywall(doi)
        if pdf:
            return pdf, "unpaywall"

    # 3. DOI landing page scrape
    if doi:
        doi_url = f"https://doi.org/{doi}"
        pdf = try_scrape_for_pdf(doi_url)
        if pdf:
            return pdf, "doi_scrape"

    # 4. URL scrape
    if url and not _is_pdf_url(url):
        pdf = try_scrape_for_pdf(url)
        if pdf:
            return pdf, "url_scrape"

    return None, "failed"


def attach_pdf_to_item(zot, parent_key: str, title: str, pdf_bytes: bytes) -> bool:
    """Upload pdf_bytes as a child PDF attachment of parent_key."""
    # Write to a temp file — pyzotero needs a file path
    safe_title = re.sub(r'[\\/:*?"<>|]', "_", title)[:80]
    with tempfile.NamedTemporaryFile(suffix=".pdf", prefix=safe_title + "_",
                                    delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    try:
        result = zot.attachment_simple([str(tmp_path)], parentid=parent_key)
        # result is a dict with keys: successful, unchanged, failed
        if result.get("successful"):
            return True
        if result.get("failed"):
            print(f"    Zotero upload failed: {result['failed']}")
        return False
    except Exception as e:
        print(f"    Upload error: {e}")
        return False
    finally:
        tmp_path.unlink(missing_ok=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download missing PDFs and attach to Zotero items.")
    parser.add_argument("--download", action="store_true",
                        help="Actually download and attach (default is dry-run)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of papers to process")
    args = parser.parse_args()

    dry_run = not args.download
    if dry_run:
        print("DRY RUN — use --download to actually attach PDFs\n")

    zot = get_client()
    print("Fetching all library items...")

    from src.research_library.zotero_client import get_all_papers
    papers = get_all_papers()
    print(f"  {len(papers)} unique items in library\n")

    # Find papers with a URL or DOI but no PDF attachment
    candidates = [p for p in papers if (p.get("doi") or p.get("url"))]
    print(f"  {len(candidates)} papers have a DOI or URL")

    print("  Checking for existing PDF attachments (this may take a moment)...")
    needs_pdf = []
    for i, p in enumerate(candidates):
        att_key = get_attachment_key(p["key"])
        if att_key is None:
            needs_pdf.append(p)
        time.sleep(0.05)  # gentle rate limiting
        if (i + 1) % 50 == 0:
            print(f"    Checked {i+1}/{len(candidates)}...")

    print(f"\n  {len(needs_pdf)} papers need a PDF attachment\n")

    if args.limit:
        needs_pdf = needs_pdf[:args.limit]
        print(f"  Processing first {len(needs_pdf)} (--limit {args.limit})\n")

    # ── Results counters ───────────────────────────────────────────────────────
    success = 0
    failed = 0
    skipped = 0

    for p in needs_pdf:
        title_short = p["title"][:65]
        year = f" ({p['year']})" if p["year"] else ""
        print(f"[{'DRY' if dry_run else '...'}] {title_short}{year}")

        if dry_run:
            doi = p.get("doi", "")
            url = p.get("url", "")
            print(f"       doi={doi or '-'}  url={url[:60] if url else '-'}")
            skipped += 1
            continue

        pdf_bytes, strategy = fetch_pdf(p.get("doi", ""), p.get("url", ""))

        if pdf_bytes is None:
            print(f"  FAIL  no PDF found  ({p.get('doi') or p.get('url', '')[:60]})")
            failed += 1
        else:
            ok = attach_pdf_to_item(zot, p["key"], p["title"], pdf_bytes)
            if ok:
                print(f"  OK    attached via {strategy}  ({len(pdf_bytes)//1024} KB)")
                success += 1
            else:
                print(f"  FAIL  download OK ({strategy}) but upload failed")
                failed += 1

        time.sleep(RETRY_DELAY)

    print("\n" + "=" * 60)
    if dry_run:
        print(f"Dry run complete: {len(needs_pdf)} papers would be processed.")
        print("Run with --download to fetch and attach PDFs.")
    else:
        print(f"Done: {success} attached, {failed} failed out of {len(needs_pdf)} attempted.")


if __name__ == "__main__":
    main()
