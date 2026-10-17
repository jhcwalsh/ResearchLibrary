"""Tests for the discovery store: follow settings, digests, the ledger and the inbox."""
from datetime import date

from research_library.discovery import store


def _cand(cid_src, title="A Paper", **kw):
    c = {"source": "arxiv", "title": title, "authors": ["Doe, Jane"], "year": "2026", "published": "2026-10-15",
         "abstract": "", "doi": "", "url": "", "pdf_url": "", "arxiv_id": "", "ssrn_id": "", "venue": "", "extra": {}}
    c.update(kw)
    c["id"] = store.candidate_id(c)
    return c


def test_candidate_id_prefers_doi_then_arxiv_then_ssrn_then_title():
    a = store.candidate_id({"doi": "https://doi.org/10.1/X", "arxiv_id": "1", "title": "t"})
    assert a == store.candidate_id({"doi": "10.1/x", "title": "other"}) and len(a) == 12
    assert store.candidate_id({"doi": "", "arxiv_id": "2610.1", "ssrn_id": "9"}) == store.candidate_id({"arxiv_id": "2610.1"})
    assert store.candidate_id({"ssrn_id": "9"}) != store.candidate_id({"title": "9"})
    assert store.candidate_id({"title": "Deep Hedging!"}) == store.candidate_id({"title": "deep hedging"})


def test_follow_defaults_and_roundtrip(tmp_path):
    f = store.load_follow(tmp_path)
    assert f["ssrn_bindings"] == [203] and f["min_fit"] == 0.62 and f["per_topic"] == 3 and f["max_total"] == 15
    f["unfollowed"] = ["ESG"]
    store.save_follow(f, tmp_path)
    assert store.load_follow(tmp_path)["unfollowed"] == ["ESG"]
    assert store.followed_topics(f, ["PE", "ESG", "Risk"]) == ["PE", "Risk"]


def test_digest_ledger_pending_and_find(tmp_path):
    d1 = {"date": "2026-10-15", "generated_at": "x", "window": {}, "stats": {},
          "candidates": [_cand(1, "One", arxiv_id="1")], "sent": []}
    d2 = {"date": "2026-10-16", "generated_at": "x", "window": {}, "stats": {},
          "candidates": [_cand(2, "Two", arxiv_id="2")], "sent": [_cand(3, "Three", ssrn_id="3")]}
    for d in (d1, d2):
        store.save_digest(d, tmp_path)
        store.record_pending(d, tmp_path)
    assert store.digest_dates(tmp_path) == ["2026-10-16", "2026-10-15"]
    pend = store.pending_candidates(tmp_path)
    assert [c["title"] for c in pend] == ["Two", "Three", "One"] and pend[0]["digest"] == "2026-10-16"
    one = d1["candidates"][0]["id"]
    store.set_status(one, "kept", tmp_path, zotero_key="ZK")
    assert store.load_ledger(tmp_path)[one]["zotero_key"] == "ZK"
    assert [c["title"] for c in store.pending_candidates(tmp_path)] == ["Two", "Three"]
    assert store.find_candidate(one, tmp_path)["title"] == "One"
    assert store.find_candidate("nope", tmp_path) is None


def test_expire_pending(tmp_path):
    d = {"date": "2026-09-01", "generated_at": "x", "window": {}, "stats": {},
         "candidates": [_cand(1, "Old", arxiv_id="1")], "sent": []}
    store.save_digest(d, tmp_path)
    store.record_pending(d, tmp_path)
    assert store.expire_pending(date(2026, 10, 16), 14, tmp_path) == 1
    assert store.load_ledger(tmp_path)[d["candidates"][0]["id"]]["status"] == "expired"
    assert store.pending_candidates(tmp_path) == []
