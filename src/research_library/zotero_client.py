import os
import re
from dotenv import load_dotenv
from pyzotero import zotero

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

load_dotenv(override=True)

_client: zotero.Zotero | None = None


def get_client() -> zotero.Zotero:
    global _client
    if _client is None:
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


def get_fulltext_batch(papers: list[dict], max_papers: int = 5, char_limit: int = 3000) -> list[dict]:
    result = []
    for p in papers[:max_papers]:
        try:
            ft = get_fulltext(p["key"], char_limit=char_limit)
        except Exception as e:
            ft = {"content": None, "source": "none", "truncated": False, "error": str(e)}
        result.append({**p, "fulltext": ft})
    # Papers beyond max_papers get no fulltext key
    result.extend(papers[max_papers:])
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
