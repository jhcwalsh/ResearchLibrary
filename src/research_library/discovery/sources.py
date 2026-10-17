"""Where new papers come from: arXiv's Atom API, SSRN's listing feed and OpenAlex.

Every source is parsed into the same candidate dict:
    id, source, title, authors ("Last, First"), year, published (ISO date),
    abstract, doi, url, pdf_url, arxiv_id, ssrn_id, venue, extra
with "" for missing text fields. `id` is set later by merge_candidates.
"""
import html as _html
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime
from urllib.parse import urlencode

from ..analysis import _title_key
from ..zotero_client import _normalize_doi
from . import store

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


# ── Fetching ──────────────────────────────────────────────────────────────────

# SSRN's feed answers browser-like agents; a bare library string risks a 403.
USER_AGENT = "Mozilla/5.0 (compatible; research-library/0.1; +https://github.com/jhcwalsh)"
ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_TIMEOUT = 60
ARXIV_ATTEMPTS = 3
ARXIV_FIREHOSE = "cat:q-fin.*"
ARXIV_KEYWORD_CATS = "(cat:q-fin.* OR cat:econ.* OR cat:stat.ML OR cat:cs.LG)"
SSRN_URL = "https://api.ssrn.com/content/v1/bindings/{binding}/papers?index={index}&count={count}&sort=0"
OPENALEX_WORKS = "https://api.openalex.org/works"
OPENALEX_SELECT = "id,doi,title,publication_date,primary_location,type,authorships,abstract_inverted_index,open_access"


def _http_get(url: str, headers: dict | None = None, timeout: int = 30) -> bytes:
    """The one place network requests happen; tests replace it."""
    import requests
    base = {"User-Agent": USER_AGENT, "Accept": "application/json, application/atom+xml, text/html;q=0.9, */*;q=0.8"}
    resp = requests.get(url, headers={**base, **(headers or {})}, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def _arxiv_stamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M")


def arxiv_query_url(search_query: str, start: int, max_results: int) -> str:
    params = {"search_query": search_query, "start": start, "max_results": max_results,
              "sortBy": "submittedDate", "sortOrder": "descending"}
    return ARXIV_API + "?" + urlencode(params, safe=':*[]()')


def _arxiv_get(url: str, pause: float, sleep) -> list[dict]:
    """arXiv stalls or rate-limits now and then; wait and try again before giving up."""
    for attempt in range(ARXIV_ATTEMPTS):
        try:
            return parse_arxiv_atom(_http_get(url, timeout=ARXIV_TIMEOUT).decode("utf-8"))
        except Exception:
            if attempt == ARXIV_ATTEMPTS - 1:
                raise
            sleep(max(pause, 1.0) * 2 * (attempt + 1))
    return []


def fetch_arxiv(window_from: datetime, window_to: datetime, topic_queries: dict[str, list[str]],
                pause: float = 3.0, sleep=time.sleep) -> list[dict]:
    """Everything in q-fin for the window, plus one keyword search per topic in the wider categories."""
    date_clause = f"submittedDate:[{_arxiv_stamp(window_from)} TO {_arxiv_stamp(window_to)}]"
    out: list[dict] = []
    start, page_size = 0, 200
    while True:
        page = _arxiv_get(arxiv_query_url(f"{ARXIV_FIREHOSE} AND {date_clause}", start, page_size), pause, sleep)
        out.extend(page)
        if len(page) < page_size:
            break
        start += page_size
        sleep(pause)
    for topic, queries in topic_queries.items():
        phrases = " OR ".join(f'all:"{q}"' for q in queries if q.strip())
        if not phrases:
            continue
        sleep(pause)
        query = f"{ARXIV_KEYWORD_CATS} AND ({phrases}) AND {date_clause}"
        out.extend(_arxiv_get(arxiv_query_url(query, 0, 100), pause, sleep))
    return out


def fetch_ssrn(bindings: list[int], since: date, max_pages: int = 5, sleep=time.sleep) -> list[dict]:
    """Newest-first pages of each binding until the approved date falls before `since`."""
    since_iso = since.isoformat()
    out: list[dict] = []
    count = 200
    for binding in bindings:
        index = 0
        for page_no in range(max_pages):
            if page_no:
                sleep(1.0)
            data = json.loads(_http_get(SSRN_URL.format(binding=binding, index=index, count=count)))
            page = parse_ssrn_page(data)
            out.extend(c for c in page if c["published"] >= since_iso)
            raw = data.get("papers") or []
            if len(raw) < count or (page and page[-1]["published"] < since_iso):
                break
            index += count
    return out


def fetch_openalex(topic_queries: dict[str, list[str]], since: date, mailto: str) -> list[dict]:
    """One keyword search per topic over titles and abstracts published since `since`."""
    out: list[dict] = []
    for topic, queries in topic_queries.items():
        phrases = " OR ".join(f'"{q}"' for q in queries if q.strip())
        if not phrases:
            continue
        params = {"filter": f"title_and_abstract.search:({phrases}),from_publication_date:{since.isoformat()}",
                  "per_page": 50, "select": OPENALEX_SELECT, "mailto": mailto}
        data = json.loads(_http_get(OPENALEX_WORKS + "?" + urlencode(params)))
        out.extend(parse_openalex_work(w) for w in data.get("results") or [])
    return out


# ── Merging and the owned check ───────────────────────────────────────────────

def _match_keys(c: dict) -> list[str]:
    keys = []
    doi = _normalize_doi(c.get("doi") or "").lower()
    if doi:
        keys.append("doi:" + doi)
    if c.get("arxiv_id"):
        keys.append("arxiv:" + c["arxiv_id"])
    if c.get("ssrn_id"):
        keys.append("ssrn:" + str(c["ssrn_id"]))
    tk = _title_key(c.get("title") or "")
    if tk:
        keys.append("title:" + tk)
    return keys


def _merge_two(a: dict, b: dict) -> dict:
    """Keep the one with an abstract; fill its empty fields from the other."""
    base, other = (a, b) if (a.get("abstract") or not b.get("abstract")) else (b, a)
    merged = dict(base)
    for k, v in other.items():
        if k == "extra":
            merged["extra"] = {**(other.get("extra") or {}), **(base.get("extra") or {})}
        elif not merged.get(k) and v:
            merged[k] = v
    return merged


def merge_candidates(cands: list[dict]) -> list[dict]:
    """Collapse duplicates across sources by DOI, arXiv id, SSRN id and title; set ids."""
    groups: list[dict] = []
    by_key: dict[str, int] = {}
    for c in cands:
        keys = _match_keys(c)
        gi = next((by_key[k] for k in keys if k in by_key), None)
        if gi is None:
            gi = len(groups)
            groups.append(dict(c))
        else:
            groups[gi] = _merge_two(groups[gi], c)
        for k in keys:
            by_key[k] = gi
    for g in groups:
        g["id"] = store.candidate_id(g)
    return groups


def drop_owned(cands: list[dict], index: dict, ledger: dict) -> list[dict]:
    """Remove papers the library already has (by DOI or title) or James has already seen."""
    dois, titles = set(), set()
    for cat in (index.get("categories") or {}).values():
        for p in cat.get("papers") or []:
            doi = _normalize_doi(p.get("doi") or "").lower()
            if doi:
                dois.add(doi)
            tk = _title_key(p.get("title") or "")
            if tk:
                titles.add(tk)
    out = []
    for c in cands:
        doi = _normalize_doi(c.get("doi") or "").lower()
        if (doi and doi in dois) or _title_key(c.get("title") or "") in titles or c.get("id") in ledger:
            continue
        out.append(c)
    return out


# ── Looking up one reference (consider_paper) ─────────────────────────────────

CROSSREF_WORKS = "https://api.crossref.org/works/"
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>]+)", re.IGNORECASE)
_ARXIV_BARE_RE = re.compile(r"^(?:arxiv:)?(\d{4}\.\d{4,5})(?:v\d+)?$", re.IGNORECASE)
_SSRN_RE = re.compile(r"(?:ssrn\.com/.*?abstract(?:_id)?=(\d+))|(?:^ssrn:(\d+)$)", re.IGNORECASE)
_META_RE = re.compile(r'<meta\s+[^>]*?(?:name|property)=["\']([^"\']+)["\'][^>]*?content=["\']([^"\']*)["\']', re.IGNORECASE)
_META_RE_REV = re.compile(r'<meta\s+[^>]*?content=["\']([^"\']*)["\'][^>]*?(?:name|property)=["\']([^"\']+)["\']', re.IGNORECASE)


def _lookup_arxiv(arxiv_id: str) -> dict:
    url = ARXIV_API + "?" + urlencode({"id_list": arxiv_id, "max_results": 1})
    cands = _arxiv_get(url, pause=3.0, sleep=time.sleep)
    for c in cands:
        if c["arxiv_id"] == arxiv_id:
            return c
    if cands:
        return cands[0]
    raise LookupError(f"arXiv has no paper with id {arxiv_id}.")


def _parse_crossref(message: dict) -> dict:
    c = _blank_candidate("crossref")
    c["doi"] = _normalize_doi(message.get("DOI") or "")
    c["title"] = _clean((message.get("title") or [""])[0])
    c["authors"] = [f"{_clean(a.get('family'))}, {_clean(a.get('given'))}".strip(", ")
                    for a in message.get("author") or [] if a.get("family") or a.get("given")]
    c["abstract"] = _clean(re.sub(r"<[^>]+>", " ", message.get("abstract") or ""))
    parts = ((message.get("issued") or {}).get("date-parts") or [[]])[0]
    if parts:
        c["year"] = str(parts[0])
        c["published"] = "-".join(f"{p:02d}" if i else str(p) for i, p in enumerate(parts))
    c["url"] = message.get("URL") or (f"https://doi.org/{c['doi']}" if c["doi"] else "")
    c["venue"] = _clean((message.get("container-title") or [""])[0])
    return c


def _lookup_doi(doi: str) -> dict:
    mailto = os.environ.get("UNPAYWALL_EMAIL", "")
    try:
        url = f"{OPENALEX_WORKS}/https://doi.org/{doi}" + (f"?{urlencode({'mailto': mailto})}" if mailto else "")
        return parse_openalex_work(json.loads(_http_get(url)))
    except Exception:
        pass
    try:
        data = json.loads(_http_get(CROSSREF_WORKS + doi))
        return _parse_crossref(data["message"])
    except Exception as exc:
        raise LookupError(f"Neither OpenAlex nor Crossref knows DOI {doi} ({exc}).") from exc


def _lookup_ssrn(ssrn_id: str, bindings: list[int], max_pages: int = 3) -> dict:
    for binding in bindings:
        for page_no in range(max_pages):
            data = json.loads(_http_get(SSRN_URL.format(binding=binding, index=page_no * 200, count=200)))
            for c in parse_ssrn_page(data):
                if c["ssrn_id"] == ssrn_id:
                    return c
            if len(data.get("papers") or []) < 200:
                break
    raise LookupError(f"SSRN paper {ssrn_id} is not in the recent feed; paste its title or DOI instead.")


def _page_meta(html_text: str) -> dict[str, list[str]]:
    meta: dict[str, list[str]] = {}
    for name, content in _META_RE.findall(html_text) + [(n, c) for c, n in _META_RE_REV.findall(html_text)]:
        meta.setdefault(name.lower(), []).append(_html.unescape(content))
    return meta


def _lookup_page(url: str) -> dict:
    meta = _page_meta(_http_get(url).decode("utf-8", "replace"))
    first = lambda *names: next((meta[n][0] for n in names if meta.get(n)), "")  # noqa: E731
    doi = _normalize_doi(first("citation_doi", "dc.identifier"))
    cand = None
    if doi and _DOI_RE.match(doi):
        try:
            cand = _lookup_doi(doi)
        except LookupError:
            cand = None
    if cand is None:
        cand = _blank_candidate("page")
    cand["title"] = cand["title"] or _clean(first("citation_title", "dc.title", "og:title"))
    if not cand["title"]:
        raise LookupError("That page carries no citation metadata; paste the paper's DOI or title instead.")
    cand["authors"] = cand["authors"] or [last_first(a) for a in meta.get("citation_author", [])]
    cand["abstract"] = cand["abstract"] or _clean(first("citation_abstract", "dc.description", "description", "og:description"))
    cand["doi"] = cand["doi"] or doi
    cand["url"] = cand["url"] or url
    cand["pdf_url"] = cand["pdf_url"] or first("citation_pdf_url")
    published = first("citation_publication_date", "citation_date", "citation_online_date").replace("/", "-")
    cand["published"] = cand["published"] or published[:10]
    cand["year"] = cand["year"] or published[:4]
    cand["venue"] = cand["venue"] or _clean(first("citation_journal_title", "citation_publisher", "og:site_name"))
    return cand


def lookup_reference(reference: str, bindings: list[int]) -> dict:
    """A candidate for an arXiv id/URL, DOI, SSRN URL or a paper's web page. Raises LookupError."""
    ref = (reference or "").strip()
    if not ref:
        raise LookupError("Give me a DOI, an arXiv id or URL, an SSRN URL, or the paper's web page.")
    arxiv_id = arxiv_id_from_url(ref)
    bare = _ARXIV_BARE_RE.match(ref)
    if bare:
        arxiv_id = bare.group(1)
    if arxiv_id:
        cand = _lookup_arxiv(arxiv_id)
    else:
        ssrn = _SSRN_RE.search(ref)
        doi = _DOI_RE.search(ref)
        if ssrn:
            cand = _lookup_ssrn(ssrn.group(1) or ssrn.group(2), bindings)
        elif doi:
            cand = _lookup_doi(doi.group(1).rstrip(".,;)"))
        elif ref.lower().startswith("http"):
            cand = _lookup_page(ref)
        else:
            raise LookupError("I need a DOI, an arXiv id or URL, an SSRN URL, or a paper's web page URL.")
    cand["source"] = "sent"
    cand["extra"] = {**(cand.get("extra") or {}), "reference": reference}
    return cand
