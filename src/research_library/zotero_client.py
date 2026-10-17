import os
import re
from dotenv import load_dotenv

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

load_dotenv(override=True)

_client = None


def get_client():
    """Lazily construct a pyzotero client. Importing pyzotero is deferred so
    modules that only need helpers (e.g. _format_item) can be loaded without
    network-layer dependencies."""
    global _client
    if _client is None:
        from pyzotero import zotero
        api_key = os.environ["ZOTERO_API_KEY"]
        library_id = os.environ["ZOTERO_LIBRARY_ID"]
        library_type = os.getenv("ZOTERO_LIBRARY_TYPE", "user")
        _client = zotero.Zotero(library_id, library_type, api_key)
    return _client


def _format_item(item: dict) -> dict:
    """Return a clean, readable representation of a Zotero item."""
    data = item.get("data", {})
    creators = data.get("creators", [])
    authors = [
        f"{c.get('lastName', '')}, {c.get('firstName', '')}".strip(", ")
        for c in creators
        if c.get("creatorType") == "author"
    ]
    return {
        "key": data.get("key", ""),
        "title": data.get("title") or "(no title)",
        "authors": authors,
        "year": (re.search(r"\b(19|20)\d{2}\b", data.get("date", "") or "") or [""])[0] if data.get("date") else "",
        "item_type": data.get("itemType", ""),
        "abstract": data.get("abstractNote", ""),
        "publication": data.get("publicationTitle", "") or data.get("bookTitle", ""),
        "doi": data.get("DOI", ""),
        "url": data.get("url", ""),
        "tags": [t["tag"] for t in data.get("tags", [])],
        "collections": data.get("collections", []),
    }


def search_papers(query: str, limit: int = 20) -> list[dict]:
    zot = get_client()
    items = zot.items(q=query, limit=limit, itemType="-attachment || note")
    return [_format_item(i) for i in items]


def get_paper(key: str) -> dict | None:
    zot = get_client()
    try:
        item = zot.item(key)
        return _format_item(item)
    except Exception:
        return None


def list_collections() -> list[dict]:
    zot = get_client()
    cols = zot.collections()
    return [
        {
            "key": c["data"]["key"],
            "name": c["data"]["name"],
            "parent": c["data"].get("parentCollection", ""),
            "num_items": c["meta"].get("numItems", 0),
        }
        for c in cols
    ]


def get_collection_papers(collection_name: str, limit: int = 50) -> list[dict]:
    zot = get_client()
    cols = zot.collections()
    match = next(
        (c for c in cols if c["data"]["name"].lower() == collection_name.lower()),
        None,
    )
    if not match:
        # Partial match fallback
        match = next(
            (c for c in cols if collection_name.lower() in c["data"]["name"].lower()),
            None,
        )
    if not match:
        return []
    items = zot.collection_items(match["data"]["key"], limit=limit, itemType="-attachment || note")
    return [_format_item(i) for i in items]


def get_recent_papers(limit: int = 20) -> list[dict]:
    zot = get_client()
    items = zot.items(limit=limit, sort="dateAdded", direction="desc", itemType="-attachment || note")
    return [_format_item(i) for i in items]


def get_annotations(item_key: str) -> list[dict]:
    zot = get_client()
    children = zot.children(item_key)
    notes = []
    for child in children:
        data = child.get("data", {})
        if data.get("itemType") in ("note", "annotation"):
            notes.append({
                "type": data.get("itemType"),
                "note": data.get("note", ""),
                "comment": data.get("annotationComment", ""),
                "text": data.get("annotationText", ""),
                "color": data.get("annotationColor", ""),
            })
    return notes


def get_attachment_key(item_key: str) -> str | None:
    zot = get_client()
    children = zot.children(item_key)
    attachment = next(
        (c for c in children if c["data"].get("contentType") == "application/pdf"),
        None,
    )
    return attachment["data"]["key"] if attachment else None


def get_fulltext(item_key: str, char_limit: int = 40000) -> dict:
    try:
        att_key = get_attachment_key(item_key)
        if not att_key:
            return {"content": None, "source": "none", "truncated": False, "error": "no_attachment"}
        zot = get_client()
        ft = zot.fulltext_item(att_key)
        content = ft.get("content", "")
        if not content or len(content) < 50:
            return {"content": None, "source": "none", "truncated": False, "error": "not_indexed"}
        truncated = len(content) > char_limit
        return {
            "content": content[:char_limit],
            "source": "zotero_index",
            "truncated": truncated,
            "total_chars": len(content),
            "indexed_pages": ft.get("indexedPages"),
            "total_pages": ft.get("totalPages"),
            "error": None,
        }
    except Exception as e:
        return {"content": None, "source": "none", "truncated": False, "error": f"api_error: {e}"}


def get_fulltext_batch(
    papers: list[dict],
    max_papers: int = 5,
    char_limit: int = 3000,
    max_attempts: int | None = None,
) -> list[dict]:
    """Fetch full text for up to max_papers papers that actually have it.

    Papers are tried in order, and ones with no indexed text are skipped over
    rather than using up the budget: keyword results are often headed by older
    items with no PDF attached, so taking the first max_papers blindly could
    return nothing usable even when later papers were fully available.

    Each attempt costs two API calls, so at most max_attempts papers are tried
    (default 5x max_papers). Papers that were tried carry a "fulltext" key
    whether or not text was found; papers never reached do not. Input order is
    preserved either way.
    """
    if max_attempts is None:
        max_attempts = max_papers * 5

    result: list[dict] = []
    found = attempts = 0
    for i, p in enumerate(papers):
        if found >= max_papers or attempts >= max_attempts:
            result.extend(papers[i:])  # untried papers pass through untouched
            break
        attempts += 1
        try:
            ft = get_fulltext(p["key"], char_limit=char_limit)
        except Exception as e:
            ft = {"content": None, "source": "none", "truncated": False, "error": str(e)}
        if ft.get("content"):
            found += 1
        result.append({**p, "fulltext": ft})
    return result


def extract_pdf_text(item_key: str, char_limit: int = 40000) -> dict:
    if not PYMUPDF_AVAILABLE:
        return {"content": None, "source": "none", "truncated": False, "error": "pymupdf_not_installed"}
    try:
        att_key = get_attachment_key(item_key)
        if not att_key:
            return {"content": None, "source": "none", "truncated": False, "error": "no_attachment"}
        zot = get_client()
        pdf_bytes = zot.file(att_key)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        truncated = len(text) > char_limit
        return {
            "content": text[:char_limit],
            "source": "pdf_extract",
            "truncated": truncated,
            "total_chars": len(text),
            "error": None,
        }
    except Exception as e:
        return {"content": None, "source": "none", "truncated": False, "error": f"extract_error: {e}"}


def get_tags(limit: int = 200) -> list[str]:
    zot = get_client()
    tags = zot.tags(limit=limit)
    return sorted(tags)


def get_all_papers() -> list[dict]:
    """Fetch every non-attachment item in the library, handling pagination."""
    zot = get_client()
    all_items = zot.everything(zot.items(itemType="-attachment || note"))
    return [_format_item(i) for i in all_items]


def get_papers_by_tag(tag: str, limit: int = 50) -> list[dict]:
    zot = get_client()
    items = zot.items(tag=tag, limit=limit, itemType="-attachment || note")
    return [_format_item(i) for i in items]


# ── Write helpers ────────────────────────────────────────────────────────────

def _normalize_doi(doi: str | None) -> str:
    if not doi:
        return ""
    return re.sub(r"^(https?://(dx\.)?doi\.org/)", "", doi.strip(), flags=re.IGNORECASE)


def _parse_creators(authors: list[str]) -> list[dict]:
    """Convert ['Last, First'] or ['Full Name'] entries → pyzotero creator dicts."""
    creators = []
    for a in authors:
        if not a or not a.strip():
            continue
        if "," in a:
            last, first = [p.strip() for p in a.split(",", 1)]
            creators.append({"creatorType": "author", "lastName": last, "firstName": first})
        else:
            creators.append({"creatorType": "author", "name": a.strip()})
    return creators


def _doi_key(doi: str | None) -> str:
    """Lookup key for DOI matching. DOIs are case-insensitive, but the stored
    value keeps the caller's casing — so this is only ever used for comparison,
    never for what gets written to Zotero."""
    return _normalize_doi(doi).lower()


# Cached {doi_key: raw_item} for the whole library. See _build_doi_index.
_doi_index: dict[str, dict] | None = None


def _reset_doi_index() -> None:
    global _doi_index
    _doi_index = None


def _build_doi_index() -> dict[str, dict]:
    """Map every DOI in the library to its item.

    The Zotero API has no server-side DOI lookup: `q` only indexes titles,
    creator names, year and attachment full text, in every qmode. A q-search for
    a DOI therefore returns nothing even when the item is present, so the whole
    library has to be scanned. That scan is slow (~20s for ~750 items), hence
    the process-lifetime cache, kept current by _remember_item as items are created.
    """
    zot = get_client()
    index: dict[str, dict] = {}
    for item in zot.everything(zot.items(itemType="-attachment || note")):
        key = _doi_key(item.get("data", {}).get("DOI", ""))
        if key:
            index.setdefault(key, item)  # first match wins; library may hold dupes
    return index


def _get_doi_index(refresh: bool = False) -> dict[str, dict]:
    global _doi_index
    if _doi_index is None or refresh:
        _doi_index = _build_doi_index()
    return _doi_index


def _remember_item(item: dict) -> None:
    """Record a newly created item so later lookups in the same process see it."""
    if _doi_index is None:
        return
    key = _doi_key(item.get("data", {}).get("DOI", ""))
    if key:
        _doi_index.setdefault(key, item)


def find_by_doi(doi: str, refresh: bool = False) -> dict | None:
    """Return the existing Zotero item whose DOI matches, or None.

    First call builds a full-library DOI index and is slow; later calls are free.
    """
    key = _doi_key(doi)
    if not key:
        return None
    return _get_doi_index(refresh=refresh).get(key)


def create_paper(
    *,
    title: str,
    authors: list[str],
    item_type: str = "journalArticle",
    year: str | None = None,
    doi: str | None = None,
    journal: str | None = None,
    volume: str | None = None,
    issue: str | None = None,
    pages: str | None = None,
    url: str | None = None,
    abstract: str | None = None,
    tags: list[str] | None = None,
    collection_keys: list[str] | None = None,
    force: bool = False,
    repository: str | None = None,
    archive_id: str | None = None,
) -> dict:
    """Create a new paper. By default returns the existing item if a paper with
    the same DOI is already in the library; pass force=True to override.
    `repository` and `archive_id` apply to preprints (e.g. "arXiv", "arXiv:2610.00001")."""
    if doi and not force:
        existing = find_by_doi(doi)
        if existing:
            return {**_format_item(existing), "duplicate": True}

    payload = _build_item_payload(
        item_type=item_type,
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        journal=journal,
        volume=volume,
        issue=issue,
        pages=pages,
        url=url,
        abstract=abstract,
        tags=tags,
        collection_keys=collection_keys,
        repository=repository,
        archive_id=archive_id,
    )
    zot = get_client()
    resp = zot.create_items([payload])
    if resp.get("failed"):
        return {"error": resp["failed"]}
    created = resp["successful"]["0"]
    _remember_item(created)
    return {**_format_item(created), "duplicate": False}


# Item-type-specific field maps. Zotero rejects fields not valid for the type.
_BOOK_LIKE_TYPES = {"book", "manuscript", "thesis", "report"}


def _build_item_payload(
    *,
    item_type: str,
    title: str,
    authors: list[str],
    year: str | None,
    doi: str | None,
    journal: str | None,
    volume: str | None,
    issue: str | None,
    pages: str | None,
    url: str | None,
    abstract: str | None,
    tags: list[str] | None,
    collection_keys: list[str] | None,
    repository: str | None = None,
    archive_id: str | None = None,
) -> dict:
    """Build a Zotero create-items payload that uses fields valid for the item type."""
    payload: dict = {
        "itemType": item_type,
        "title": title,
        "creators": _parse_creators(authors),
        "date": str(year) if year else "",
        "url": url or "",
        "abstractNote": abstract or "",
        "tags": [{"tag": t} for t in (tags or [])],
        "collections": collection_keys or [],
    }
    # DOI is valid for journal/preprint/conference/etc. but not for plain books.
    if item_type not in _BOOK_LIKE_TYPES:
        payload["DOI"] = _normalize_doi(doi)
    elif doi:
        # Stash DOI in extra so it survives for book/thesis/report types
        payload["extra"] = f"DOI: {_normalize_doi(doi)}"

    if item_type == "preprint":
        # Preprints have no journal fields; Zotero rejects them. The venue is the repository.
        payload["repository"] = repository or journal or ""
        payload["archiveID"] = archive_id or ""
    elif item_type in _BOOK_LIKE_TYPES:
        # For books, the `journal` kwarg holds the publisher name
        if journal:
            payload["publisher"] = journal
        if pages:
            payload["numPages"] = pages
        # volume/issue not applicable to plain books
    else:
        payload["publicationTitle"] = journal or ""
        payload["volume"] = str(volume) if volume else ""
        payload["issue"] = str(issue) if issue else ""
        payload["pages"] = pages or ""

    return payload


def create_papers_batch(papers: list[dict], force: bool = False) -> dict:
    """Create multiple papers in a single batched API call (up to 50 per batch).
    Each dict in `papers` mirrors create_paper's kwargs (title, authors, year,
    doi, journal, volume, issue, pages, url, abstract, tags, collection_keys).
    Returns {created: [items], duplicates: [items], failed: [{input, error}]}.
    """
    if not papers:
        return {"created": [], "duplicates": [], "failed": []}

    created: list[dict] = []
    duplicates: list[dict] = []
    failed: list[dict] = []
    payloads: list[dict] = []
    inputs: list[dict] = []  # parallel list for error reporting

    seen_in_batch: dict[str, dict] = {}  # doi_key -> the payload already queued
    for p in papers:
        doi = p.get("doi")
        if doi and not force:
            existing = find_by_doi(doi)
            if existing:
                duplicates.append(_format_item(existing))
                continue
            # Two entries in the same batch can share a DOI; neither is in the
            # library yet, so only this check stops both from being created.
            key = _doi_key(doi)
            if key in seen_in_batch:
                duplicates.append({"title": p.get("title", ""), "doi": doi,
                                   "duplicate_of_batch_entry": seen_in_batch[key].get("title", "")})
                continue
            seen_in_batch[key] = p
        payloads.append(_build_item_payload(
            item_type=p.get("item_type", "journalArticle"),
            title=p["title"],
            authors=p.get("authors", []),
            year=p.get("year"),
            doi=doi,
            journal=p.get("journal"),
            volume=p.get("volume"),
            issue=p.get("issue"),
            pages=p.get("pages"),
            url=p.get("url"),
            abstract=p.get("abstract"),
            tags=p.get("tags"),
            collection_keys=p.get("collection_keys"),
        ))
        inputs.append(p)

    zot = get_client()
    for start in range(0, len(payloads), 50):
        chunk = payloads[start:start + 50]
        in_chunk = inputs[start:start + 50]
        resp = zot.create_items(chunk)
        for idx_str, item in (resp.get("successful") or {}).items():
            _remember_item(item)
            created.append(_format_item(item))
        for idx_str, err in (resp.get("failed") or {}).items():
            failed.append({"input": in_chunk[int(idx_str)], "error": err})

    return {"created": created, "duplicates": duplicates, "failed": failed}


def add_to_collection(item_key: str, collection_key: str) -> dict:
    zot = get_client()
    item = zot.item(item_key)
    existing = item["data"].get("collections", [])
    if collection_key in existing:
        return {"ok": True, "already_member": True, "collections": existing}
    item["data"]["collections"] = existing + [collection_key]
    zot.update_item(item)
    return {"ok": True, "already_member": False, "collections": item["data"]["collections"]}


def remove_from_collection(item_key: str, collection_key: str) -> dict:
    zot = get_client()
    item = zot.item(item_key)
    existing = item["data"].get("collections", [])
    if collection_key not in existing:
        return {"ok": True, "was_member": False, "collections": existing}
    item["data"]["collections"] = [c for c in existing if c != collection_key]
    zot.update_item(item)
    return {"ok": True, "was_member": True, "collections": item["data"]["collections"]}


def add_tag(item_key: str, tag: str) -> dict:
    zot = get_client()
    item = zot.item(item_key)
    existing = {t["tag"] for t in item["data"].get("tags", [])}
    if tag in existing:
        return {"ok": True, "already_tagged": True}
    item["data"]["tags"] = item["data"].get("tags", []) + [{"tag": tag}]
    zot.update_item(item)
    return {"ok": True, "already_tagged": False}


def remove_tag(item_key: str, tag: str) -> dict:
    zot = get_client()
    item = zot.item(item_key)
    existing = item["data"].get("tags", [])
    new = [t for t in existing if t.get("tag") != tag]
    if len(new) == len(existing):
        return {"ok": True, "was_tagged": False}
    item["data"]["tags"] = new
    zot.update_item(item)
    return {"ok": True, "was_tagged": True}


def update_paper(item_key: str, **fields) -> dict:
    """Update fields on an existing item.
    Supported keys: title, year, doi, journal, volume, issue, pages, url, abstract.
    """
    zot = get_client()
    item = zot.item(item_key)
    data = item["data"]
    field_map = {
        "title": "title",
        "year": "date",
        "doi": "DOI",
        "journal": "publicationTitle",
        "volume": "volume",
        "issue": "issue",
        "pages": "pages",
        "url": "url",
        "abstract": "abstractNote",
    }
    for k, v in fields.items():
        if k in field_map:
            data[field_map[k]] = "" if v is None else str(v)
    zot.update_item(item)
    return _format_item(item)


def attach_pdf(item_key: str, pdf_path: str) -> dict:
    """Attach a local PDF as a child of an existing item.

    Not pyzotero's attachment_simple: that puts the whole path in the item's
    `filename`, and Zotero rejects it ("Stored-file filename cannot contain a
    directory path"). The name and its directory have to travel separately.
    """
    if not os.path.isfile(pdf_path):
        return {"error": f"File not found: {pdf_path}"}

    directory, filename = os.path.split(os.path.abspath(pdf_path))
    zot = get_client()
    template = zot.item_template("attachment", linkmode="imported_file")
    template["title"] = filename
    template["filename"] = filename
    template["contentType"] = "application/pdf"

    result = zot.upload_attachments([template], parentid=item_key, basedir=directory)

    # pyzotero reports rejections in a list instead of raising, so an unchecked
    # return value reads as success even when nothing was uploaded.
    if result.get("failure"):
        return {"error": "Zotero rejected the attachment", "details": result["failure"]}
    unchanged = bool(result.get("unchanged"))
    return {
        "ok": True,
        "unchanged": unchanged,  # file with identical contents was already attached
        "filename": filename,
        "attachment_key": template.get("key", ""),
        "parent": item_key,
    }


def create_collection(name: str, parent_key: str | None = None) -> dict:
    """Create a new collection (folder), optionally as a child of an existing one."""
    payload = [{"name": name}]
    if parent_key:
        payload[0]["parentCollection"] = parent_key
    zot = get_client()
    resp = zot.create_collections(payload)
    if resp.get("failed"):
        return {"error": resp["failed"]}
    return {"key": resp["success"]["0"], "name": name, "parent": parent_key}
