"""Acting on candidates: keep (the only Zotero write), dismiss, and consider a sent link.

Keep files the paper in Zotero, tries for a PDF, and folds the paper into the
cached index and embeddings straight away, so semantic search and the Contents
page see it without a rebuild. The topic's summary caption on the Contents page
will say it predates the latest papers until `research-admin contents` runs.
"""
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import embeddings, zotero_client
from ..analysis import CACHE_PATH as INDEX_CACHE_PATH
from . import run, scoring, sources, store, summaries


# ── Index and embedding updates ───────────────────────────────────────────────

def append_to_index(paper: dict, topic: str, index_path: Path | None = None) -> str:
    """Put a kept paper at the front of its topic in library_index.json.

    Returns "updated", "exists" (key already there), "topic_missing" or "no_index".
    """
    path = Path(index_path) if index_path else INDEX_CACHE_PATH
    if not path.exists():
        return "no_index"
    index = json.loads(path.read_text(encoding="utf-8"))
    cat = index.get("categories", {}).get(topic)
    if cat is None:
        return "topic_missing"
    if any(p.get("key") == paper.get("key") for c in index["categories"].values() for p in c.get("papers", [])):
        return "exists"
    record = {k: paper.get(k, "") for k in ("key", "title", "authors", "year", "abstract", "doi", "url")}
    cat["papers"].insert(0, record)
    index["unique_papers"] = int(index.get("unique_papers", 0)) + 1
    store.atomic_write_text(path, json.dumps(index, indent=2))
    return "updated"


def append_embedding(key: str, vector: list[float], topic: str, emb_path: Path | None = None) -> bool:
    """Append one paper vector to embeddings.npz. False if the file is missing or the vector empty."""
    path = Path(emb_path) if emb_path else embeddings.EMBEDDINGS_PATH
    if not path.exists() or not vector:
        return False
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in ("keys", "vectors", "paper_cats", "cat_names", "cat_vectors")}
    if key in arrays["keys"].tolist():
        return True
    row = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(
        tmp,
        keys=np.append(arrays["keys"], key),
        vectors=np.vstack([arrays["vectors"].astype(np.float32), row]),
        paper_cats=np.append(arrays["paper_cats"], topic),
        cat_names=arrays["cat_names"],
        cat_vectors=arrays["cat_vectors"],
    )
    os.replace(tmp, path)
    return True


def _fold_into_index(cand: dict, key: str, index_path: Path | None, emb_path: Path | None) -> tuple[str, str]:
    """Index and embedding update after the Zotero write; never raises."""
    try:
        outcome = append_to_index({**cand, "key": key}, cand.get("topic") or "", index_path)
        if outcome in ("topic_missing", "no_index"):
            return outcome, ""
        has_vector = append_embedding(key, cand.get("vector") or [], cand.get("topic") or "", emb_path)
        return ("updated" if has_vector else "updated_no_vector"), ""
    except Exception as exc:
        return "failed", f"index update failed: {exc}"


# ── Keep ──────────────────────────────────────────────────────────────────────

def _is_preprint(c: dict) -> bool:
    return c.get("source") in ("arxiv", "ssrn") or bool(c.get("arxiv_id")) or bool(c.get("ssrn_id")) or c.get("venue") == "SSRN"


def _repository(c: dict) -> str | None:
    if c.get("arxiv_id"):
        return "arXiv"
    if c.get("ssrn_id") or c.get("venue") == "SSRN":
        return "SSRN"
    return c.get("venue") or None


def _archive_id(c: dict) -> str | None:
    if c.get("arxiv_id"):
        return f"arXiv:{c['arxiv_id']}"
    if c.get("ssrn_id"):
        return f"SSRN:{c['ssrn_id']}"
    return None


def _result(cid: str, title: str, status: str, **extra) -> dict:
    base = {"id": cid, "title": title, "status": status, "zotero_key": None, "pdf": "skipped", "index": "skipped", "message": ""}
    base.update(extra)
    return base


def keep_candidates(ids: list[str], attach_pdf: bool = True, root: Path | None = None, index_path: Path | None = None,
                    emb_path: Path | None = None, zc=None, fetch_pdf=None, attach=None) -> list[dict]:
    """File each candidate in Zotero, try for a PDF, update the cached index and embeddings."""
    zc = zc or zotero_client
    if attach_pdf and (fetch_pdf is None or attach is None):
        from .. import downloads
        fetch_pdf = fetch_pdf or downloads.fetch_pdf
        attach = attach or downloads.attach_pdf_to_item

    results = []
    for cid in ids:
        ledger = store.load_ledger(root)
        entry = ledger.get(cid)
        cand = store.find_candidate(cid, root) if entry else None
        if not entry or not cand:
            results.append(_result(cid, "", "unknown", message="No candidate with that id."))
            continue
        if entry.get("status") == "kept":
            results.append(_result(cid, cand["title"], "already_kept", zotero_key=entry.get("zotero_key"),
                                   message="Already kept."))
            continue

        preprint = _is_preprint(cand)
        try:
            created = zc.create_paper(
                title=cand["title"], authors=cand.get("authors") or [],
                item_type="preprint" if preprint else "journalArticle",
                year=cand.get("year") or None, doi=cand.get("doi") or None, url=cand.get("url") or None,
                abstract=cand.get("abstract") or None,
                journal=None if preprint else (cand.get("venue") or None),
                tags=[cand.get("topic") or "unsorted", "discovered"],
                repository=_repository(cand) if preprint else None,
                archive_id=_archive_id(cand) if preprint else None,
            )
        except Exception as exc:
            results.append(_result(cid, cand["title"], "error", message=f"Zotero rejected the item: {exc}"))
            continue
        if not created or created.get("error") or not created.get("key"):
            results.append(_result(cid, cand["title"], "error", message=f"Zotero rejected the item: {created}"))
            continue

        key = created["key"]
        status = "already_owned" if created.get("duplicate") else "kept"
        # The Zotero write is the irreversible step: record it before anything else can fail.
        store.set_status(cid, "kept", root, zotero_key=key)
        pdf = "skipped"
        if attach_pdf:
            try:
                data, _how = fetch_pdf(cand.get("doi") or "", cand.get("pdf_url") or cand.get("url") or "")
                if data:
                    pdf = "attached" if attach(None, key, cand["title"], data) else "failed"
                else:
                    pdf = "not_found"
            except Exception:
                pdf = "failed"

        index_status, note = _fold_into_index(cand, key, index_path, emb_path)
        messages = [m for m in (
            "Already in Zotero (same DOI); folded into the index." if status == "already_owned" else "", note) if m]
        results.append(_result(cid, cand["title"], status, zotero_key=key, pdf=pdf, index=index_status,
                               message=" ".join(messages)))
    return results


# ── Dismiss ───────────────────────────────────────────────────────────────────

def dismiss_candidates(ids: list[str] | None = None, all_pending: bool = False, root: Path | None = None) -> list[dict]:
    if all_pending:
        ids = [c["id"] for c in store.pending_candidates(root)]
    results = []
    for cid in ids or []:
        entry = store.load_ledger(root).get(cid)
        if not entry:
            results.append({"id": cid, "title": "", "status": "unknown"})
        elif entry.get("status") != "pending":
            results.append({"id": cid, "title": entry.get("title", ""), "status": "not_pending"})
        else:
            store.set_status(cid, "dismissed", root)
            results.append({"id": cid, "title": entry.get("title", ""), "status": "dismissed"})
    return results


# ── Consider a link someone sent ──────────────────────────────────────────────

def consider_reference(reference: str, client=None, root: Path | None = None, index_path: Path | None = None,
                       library: dict | None = None, embed=None, bindings: list[int] | None = None,
                       now: datetime | None = None) -> dict:
    """Score and summarise one paper the way the daily run would, and park it in today's inbox."""
    now = now or datetime.now()
    index = run._load_index(index_path)
    follow = store.load_follow(root)
    cand = sources.lookup_reference(reference, bindings or follow.get("ssrn_bindings") or [])
    cand["source"] = "sent"
    cand = sources.merge_candidates([cand])[0]

    ledger = store.load_ledger(root)
    known = ledger.get(cand["id"])
    if known and known.get("status") == "kept":
        return {"owned": True, "title": cand["title"], "doi": cand.get("doi", ""), "zotero_key": known.get("zotero_key")}
    if known and known.get("status") == "pending":
        existing = store.find_candidate(cand["id"], root)
        if existing:
            return existing
    if not sources.drop_owned([cand], index, {}):
        return {"owned": True, "title": cand["title"], "doi": cand.get("doi", "")}

    library = library or embeddings.load_embeddings()
    if library is None:
        raise FileNotFoundError(f"No embeddings at {embeddings.EMBEDDINGS_PATH}. Run `research-admin embed` first.")
    all_topics = sorted(set(library["paper_cats"]))
    vectors = scoring.embed_candidates([cand], embed=embed)
    scoring.score_candidates([cand], vectors, library, all_topics)

    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    papers_by_key = {p["key"]: p for c in index["categories"].values() for p in c["papers"]}
    nearest = [papers_by_key[n["key"]] for n in cand.get("nearest", []) if n["key"] in papers_by_key]
    try:
        cand.update(summaries.summarize_candidate(client, cand, nearest))
    except Exception as exc:
        cand.update(run._failed_summary(exc))

    today = now.date().isoformat()
    digest = store.load_digest(today, root) or {
        "date": today, "generated_at": now.isoformat(timespec="seconds"), "window": {},
        "stats": {"fetched": {}, "new": 0, "scored": 0, "shown": 0}, "candidates": [], "sent": []}
    digest.setdefault("sent", []).append(cand)
    store.save_digest(digest, root)
    store.record_pending(digest, root)
    if known:  # previously dismissed or expired: back into the inbox, pointing at today's copy
        store.set_status(cand["id"], "pending", root, digest=today)
    return cand
