"""Tests for keeping, dismissing and considering discovery candidates."""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from research_library.discovery import keep, sources, store

FIX = Path(__file__).parent / "fixtures"


def _index(tmp_path, topics=("Big",)):
    index = {"generated_at": "x", "unique_papers": 1, "categories": {t: {"summary": "s", "subcategories": {}, "papers": [
        {"key": "B1", "title": "Owned", "authors": ["Doe, J"], "year": "2020", "abstract": "a", "doi": "10.9/owned"}]} for t in topics}}
    ip = tmp_path / "library_index.json"
    ip.write_text(json.dumps(index), encoding="utf-8")
    ep = tmp_path / "embeddings.npz"
    np.savez(ep, keys=np.array(["B1"]), vectors=np.eye(4, dtype=np.float32)[:1], paper_cats=np.array(["Big"]),
             cat_names=np.array(["Big"]), cat_vectors=np.eye(4, dtype=np.float32)[:1])
    return ip, ep


def _digest(root, cands):
    d = {"date": "2026-10-16", "generated_at": "x", "window": {}, "stats": {}, "candidates": cands, "sent": []}
    store.save_digest(d, root)
    store.record_pending(d, root)
    return d


def _cand(title, **kw):
    c = {"source": "arxiv", "title": title, "authors": ["Doe, Jane"], "year": "2026", "published": "2026-10-15", "abstract": "abs",
         "doi": "10.48550/arXiv.2610.1", "url": "https://arxiv.org/abs/2610.1", "pdf_url": "https://arxiv.org/pdf/2610.1",
         "arxiv_id": "2610.1", "ssrn_id": "", "venue": "arXiv q-fin.PM", "extra": {},
         "topic": "Big", "fit": 0.7, "nearest": [], "vector": [0, 1, 0, 0]}
    c.update(kw)
    c["id"] = store.candidate_id(c)
    return c


class FakeZc:
    def __init__(self, dup=False):
        self.calls = []
        self.dup = dup

    def create_paper(self, **kw):
        self.calls.append(kw)
        return {"key": "NEWKEY", "duplicate": self.dup}


def test_keep_creates_preprint_attaches_pdf_and_updates_index(tmp_path):
    ip, ep = _index(tmp_path)
    root = tmp_path / "disc"
    c = _cand("New A")
    _digest(root, [c])
    zc = FakeZc()
    out = keep.keep_candidates([c["id"]], root=root, index_path=ip, emb_path=ep, zc=zc,
                               fetch_pdf=lambda doi, url: (b"%PDF-1.4 x", "direct_url"), attach=lambda z, key, title, data: True)
    assert out[0]["status"] == "kept" and out[0]["zotero_key"] == "NEWKEY" and out[0]["pdf"] == "attached" and out[0]["index"] == "updated"
    kw = zc.calls[0]
    assert kw["item_type"] == "preprint" and kw["repository"] == "arXiv" and kw["archive_id"] == "arXiv:2610.1" and kw["tags"] == ["Big", "discovered"]
    index = json.loads(ip.read_text(encoding="utf-8"))
    assert index["categories"]["Big"]["papers"][0] == {"key": "NEWKEY", "title": "New A", "authors": ["Doe, Jane"], "year": "2026",
                                                        "abstract": "abs", "doi": "10.48550/arXiv.2610.1", "url": "https://arxiv.org/abs/2610.1"}
    assert index["unique_papers"] == 2
    emb = np.load(ep)
    assert list(emb["keys"]) == ["B1", "NEWKEY"] and emb["vectors"].shape == (2, 4) and list(emb["paper_cats"]) == ["Big", "Big"]
    assert store.load_ledger(root)[c["id"]]["status"] == "kept"


def test_keep_unknown_and_already_kept_do_not_touch_zotero(tmp_path):
    ip, ep = _index(tmp_path)
    root = tmp_path / "disc"
    c = _cand("New A")
    _digest(root, [c])
    store.set_status(c["id"], "kept", root, zotero_key="K")
    zc = FakeZc()
    out = keep.keep_candidates(["nope", c["id"]], root=root, index_path=ip, emb_path=ep, zc=zc, fetch_pdf=None, attach=None)
    assert [r["status"] for r in out] == ["unknown", "already_kept"] and zc.calls == []


def test_keep_with_missing_topic_still_adds_to_zotero(tmp_path):
    ip, ep = _index(tmp_path, topics=("Renamed",))
    root = tmp_path / "disc"
    c = _cand("New A")
    _digest(root, [c])
    zc = FakeZc()
    out = keep.keep_candidates([c["id"]], attach_pdf=False, root=root, index_path=ip, emb_path=ep, zc=zc)
    assert out[0]["status"] == "kept" and out[0]["index"] == "topic_missing" and out[0]["pdf"] == "skipped" and len(zc.calls) == 1


def test_keep_reports_duplicate_and_pdf_not_found(tmp_path):
    ip, ep = _index(tmp_path)
    root = tmp_path / "disc"
    c = _cand("New A")
    _digest(root, [c])
    out = keep.keep_candidates([c["id"]], root=root, index_path=ip, emb_path=ep, zc=FakeZc(dup=True),
                               fetch_pdf=lambda doi, url: (None, "failed"), attach=None)
    assert out[0]["status"] == "already_owned" and out[0]["pdf"] == "not_found" and store.load_ledger(root)[c["id"]]["zotero_key"] == "NEWKEY"


def test_dismiss(tmp_path):
    root = tmp_path / "disc"
    a, b = _cand("A"), _cand("B", arxiv_id="2610.2", doi="10.48550/arXiv.2610.2")
    _digest(root, [a, b])
    assert [r["status"] for r in keep.dismiss_candidates([a["id"], "zzz"], root=root)] == ["dismissed", "unknown"]
    assert [r["id"] for r in keep.dismiss_candidates(all_pending=True, root=root)] == [b["id"]]
    assert store.pending_candidates(root) == []


def test_lookup_reference_routes(monkeypatch):
    seen = []

    def fake(url, headers=None, timeout=30):
        seen.append(url)
        if "export.arxiv.org" in url:
            return (FIX / "arxiv_sample.xml").read_bytes()
        if "api.openalex.org/works/https://doi.org/" in url:
            return json.dumps(json.loads((FIX / "openalex_sample.json").read_text(encoding="utf-8"))["results"][0]).encode()
        raise AssertionError(url)

    monkeypatch.setattr(sources, "_http_get", fake)
    assert sources.lookup_reference("https://arxiv.org/abs/2610.22144v1", [203])["arxiv_id"] == "2610.22144" and "id_list=2610.22144" in seen[-1]
    assert sources.lookup_reference("10.2139/ssrn.1234", [203])["doi"] == "10.2139/ssrn.1234"
    with pytest.raises(LookupError):
        sources.lookup_reference("just some words", [203])


def test_consider_reference_scores_summarises_and_stores_under_sent(tmp_path, monkeypatch):
    ip, ep = _index(tmp_path)
    root = tmp_path / "disc"
    monkeypatch.setattr(sources, "lookup_reference", lambda ref, bindings: {**_cand("Sent paper", source="sent", topic=None), "id": ""})
    monkeypatch.setattr(keep.summaries, "summarize_candidate",
                        lambda client, cand, nearest: {"summary": "S", "why_it_fits": "W", "recommendation": "maybe", "reason": "R"})
    library = {"keys": ["B1"], "vectors": np.eye(4, dtype=np.float32)[:1], "paper_cats": ["Big"], "cat_names": ["Big"],
               "cat_vectors": np.eye(4, dtype=np.float32)[:1]}
    out = keep.consider_reference("https://arxiv.org/abs/2610.1", client=object(), root=root, index_path=ip, library=library,
                                  embed=lambda texts: np.array([[1, 0, 0, 0]] * len(texts), dtype=np.float32), now=datetime(2026, 10, 16, 9, 0))
    assert out["topic"] == "Big" and out["summary"] == "S" and out["source"] == "sent"
    digest = store.load_digest("2026-10-16", root)
    assert digest["sent"][0]["title"] == "Sent paper" and digest["candidates"] == []
    assert store.load_ledger(root)[out["id"]]["status"] == "pending"


# ── Review fix pass ───────────────────────────────────────────────────────────

def _raise(exc):
    raise exc


def test_keep_survives_an_index_update_failure_and_marks_kept(tmp_path, monkeypatch):
    ip, ep = _index(tmp_path)
    root = tmp_path / "disc"
    c = _cand("New A")
    _digest(root, [c])
    zc = FakeZc()
    monkeypatch.setattr(keep, "append_embedding", lambda *a, **k: _raise(RuntimeError("bad npz")))
    out = keep.keep_candidates([c["id"]], attach_pdf=False, root=root, index_path=ip, emb_path=ep, zc=zc)
    assert out[0]["status"] == "kept" and out[0]["index"] == "failed" and "bad npz" in out[0]["message"]
    assert store.load_ledger(root)[c["id"]]["status"] == "kept"
    out2 = keep.keep_candidates([c["id"]], attach_pdf=False, root=root, index_path=ip, emb_path=ep, zc=zc)
    assert out2[0]["status"] == "already_kept" and len(zc.calls) == 1


def test_append_to_index_is_idempotent_and_a_missing_vector_file_is_reported(tmp_path):
    ip, ep = _index(tmp_path)
    paper = {"key": "NEWKEY", "title": "T", "authors": [], "year": "", "abstract": "", "doi": "", "url": ""}
    assert keep.append_to_index(paper, "Big", ip) == "updated"
    assert keep.append_to_index(paper, "Big", ip) == "exists"
    assert keep.append_to_index(paper, "Nope", ip) == "topic_missing"
    assert keep.append_to_index(paper, "Big", tmp_path / "no_index.json") == "no_index"
    assert len(json.loads(ip.read_text(encoding="utf-8"))["categories"]["Big"]["papers"]) == 2
    root = tmp_path / "disc"
    c = _cand("New B", arxiv_id="2610.9", doi="10.48550/arXiv.2610.9")
    _digest(root, [c])
    out = keep.keep_candidates([c["id"]], attach_pdf=False, root=root, index_path=ip, emb_path=tmp_path / "missing.npz", zc=FakeZc())
    assert out[0]["index"] == "updated_no_vector"


def test_consider_reference_on_a_dismissed_paper_makes_it_pending_again(tmp_path, monkeypatch):
    ip, ep = _index(tmp_path)
    root = tmp_path / "disc"
    prior = _cand("Sent paper", source="sent")
    _digest(root, [prior])
    store.set_status(prior["id"], "dismissed", root)
    monkeypatch.setattr(sources, "lookup_reference", lambda ref, bindings: {**_cand("Sent paper", source="sent", topic=None), "id": ""})
    monkeypatch.setattr(keep.summaries, "summarize_candidate",
                        lambda client, cand, nearest: {"summary": "S", "why_it_fits": "W", "recommendation": "maybe", "reason": "R"})
    library = {"keys": ["B1"], "vectors": np.eye(4, dtype=np.float32)[:1], "paper_cats": ["Big"], "cat_names": ["Big"],
               "cat_vectors": np.eye(4, dtype=np.float32)[:1]}
    out = keep.consider_reference("https://arxiv.org/abs/2610.1", client=object(), root=root, index_path=ip, library=library,
                                  embed=lambda texts: np.array([[1, 0, 0, 0]] * len(texts), dtype=np.float32), now=datetime(2026, 10, 17, 9, 0))
    assert store.load_ledger(root)[out["id"]]["status"] == "pending"
    assert out["id"] in {c["id"] for c in store.pending_candidates(root)}


def test_lookup_page_reads_citation_meta_and_strips_a_doi_prefix(monkeypatch):
    html = ('<html><head><meta name="citation_title" content="A Page Paper">'
            '<meta name="citation_author" content="Jane Doe"><meta name="citation_author" content="Rick Roe">'
            '<meta name="citation_doi" content="doi:10.1234/abc.5"><meta name="citation_abstract" content="Abs.">'
            '<meta name="citation_publication_date" content="2026/10/01"></head><body></body></html>')

    def fake(url, headers=None, timeout=30):
        if "api.openalex.org" in url or "crossref" in url:
            raise RuntimeError("down")
        return html.encode("utf-8")

    monkeypatch.setattr(sources, "_http_get", fake)
    c = sources.lookup_reference("https://publisher.example/paper", [203])
    assert c["doi"] == "10.1234/abc.5" and c["title"] == "A Page Paper"
    assert c["authors"] == ["Doe, Jane", "Roe, Rick"] and c["published"] == "2026-10-01" and c["abstract"] == "Abs."
