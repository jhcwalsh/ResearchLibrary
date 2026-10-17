"""Where new papers come from: arXiv's Atom API, SSRN's listing feed and OpenAlex.

Every source is parsed into the same candidate dict:
    id, source, title, authors ("Last, First"), year, published (ISO date),
    abstract, doi, url, pdf_url, arxiv_id, ssrn_id, venue, extra
with "" for missing text fields. `id` is set later by merge_candidates.
"""
import re
import xml.etree.ElementTree as ET
from datetime import datetime

from ..zotero_client import _normalize_doi

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"

_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^\s?#]+?)(?:v\d+)?(?:\.pdf)?$", re.IGNORECASE)


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def last_first(name: str) -> str:
    """'Jane Q. Doe' -> 'Doe, Jane Q.'; names already in 'Last, First' form pass through."""
    name = _clean(name)
    if "," in name or " " not in name:
        return name
    first, last = name.rsplit(" ", 1)
    return f"{last}, {first}"


def _blank_candidate(source: str) -> dict:
    return {"id": "", "source": source, "title": "", "authors": [], "year": "", "published": "",
            "abstract": "", "doi": "", "url": "", "pdf_url": "", "arxiv_id": "", "ssrn_id": "",
            "venue": "", "extra": {}}


def arxiv_id_from_url(url: str) -> str:
    m = _ARXIV_ID_RE.search(url or "")
    return m.group(1) if m else ""


# ── arXiv ─────────────────────────────────────────────────────────────────────

def parse_arxiv_atom(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    out = []
    for entry in root.findall(f"{ATOM}entry"):
        c = _blank_candidate("arxiv")
        c["url"] = _clean(entry.findtext(f"{ATOM}id"))
        c["arxiv_id"] = arxiv_id_from_url(c["url"])
        c["title"] = _clean(entry.findtext(f"{ATOM}title"))
        c["abstract"] = _clean(entry.findtext(f"{ATOM}summary"))
        c["authors"] = [last_first(a.findtext(f"{ATOM}name") or "") for a in entry.findall(f"{ATOM}author")]
        published = _clean(entry.findtext(f"{ATOM}published"))
        c["published"] = published[:10]
        c["year"] = published[:4]
        doi = _clean(entry.findtext(f"{ARXIV}doi"))
        c["doi"] = doi or (f"10.48550/arXiv.{c['arxiv_id']}" if c["arxiv_id"] else "")
        for link in entry.findall(f"{ATOM}link"):
            if link.get("title") == "pdf":
                c["pdf_url"] = link.get("href", "")
        primary = entry.find(f"{ARXIV}primary_category")
        term = primary.get("term", "") if primary is not None else ""
        c["venue"] = f"arXiv {term}".strip()
        out.append(c)
    return out


# ── SSRN ──────────────────────────────────────────────────────────────────────

def _ssrn_date(text: str) -> str:
    """'16 Oct 2026' -> '2026-10-16'; '' when it cannot be parsed."""
    try:
        return datetime.strptime((text or "").strip(), "%d %b %Y").date().isoformat()
    except ValueError:
        return ""


def parse_ssrn_page(data: dict) -> list[dict]:
    out = []
    for p in data.get("papers") or []:
        published = _ssrn_date(p.get("approved_date", ""))
        if not published:
            continue
        c = _blank_candidate("ssrn")
        c["ssrn_id"] = str(p.get("id", ""))
        c["title"] = _clean(p.get("title"))
        c["authors"] = [f"{_clean(a.get('last_name'))}, {_clean(a.get('first_name'))}".strip(", ")
                        for a in p.get("authors") or []]
        c["published"] = published
        c["year"] = published[:4]
        c["url"] = p.get("url") or f"https://papers.ssrn.com/sol3/papers.cfm?abstract_id={c['ssrn_id']}"
        c["venue"] = "SSRN"
        affiliations = p.get("affiliations") or []
        if isinstance(affiliations, str):
            affiliations = [a.strip() for a in affiliations.split(",") if a.strip()]
        c["extra"] = {"affiliations": affiliations, "pages": p.get("page_count"), "downloads": p.get("downloads")}
        out.append(c)
    return out


# ── OpenAlex ──────────────────────────────────────────────────────────────────

def rebuild_abstract(inverted: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; put the words back in order."""
    if not inverted:
        return ""
    positions = sorted((pos, word) for word, poss in inverted.items() for pos in poss)
    return " ".join(word for _, word in positions)


def parse_openalex_work(work: dict) -> dict:
    c = _blank_candidate("openalex")
    c["doi"] = _normalize_doi(work.get("doi") or "")
    c["title"] = _clean(work.get("title"))
    c["abstract"] = rebuild_abstract(work.get("abstract_inverted_index"))
    c["authors"] = [last_first(((a or {}).get("author") or {}).get("display_name") or "")
                    for a in work.get("authorships") or []]
    c["authors"] = [a for a in c["authors"] if a]
    published = work.get("publication_date") or ""
    c["published"] = published[:10]
    c["year"] = published[:4]
    loc = work.get("primary_location") or {}
    landing = loc.get("landing_page_url") or ""
    c["url"] = landing or (f"https://doi.org/{c['doi']}" if c["doi"] else "")
    c["pdf_url"] = ((work.get("open_access") or {}).get("oa_url")) or loc.get("pdf_url") or ""
    c["venue"] = ((loc.get("source") or {}).get("display_name")) or ""
    c["arxiv_id"] = arxiv_id_from_url(landing)
    return c
