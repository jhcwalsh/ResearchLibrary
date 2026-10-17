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
    assert f["ssrn_bindings"] == [203] and f["min_fit"] == 0.50 and f["min_fit_title_only"] == 0.76 and f["per_topic"] == 3 and f["max_total"] == 15
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


# ── Review fix pass ───────────────────────────────────────────────────────────

import json  # noqa: E402
import pytest  # noqa: E402


def _raise(exc):
    raise exc


def test_pending_candidates_reads_only_digests_with_pending_entries(tmp_path, monkeypatch):
    d1 = {"date": "2026-10-15", "generated_at": "x", "window": {}, "stats": {}, "candidates": [_cand(1, "One", arxiv_id="1")], "sent": []}
    d2 = {"date": "2026-10-16", "generated_at": "x", "window": {}, "stats": {}, "candidates": [_cand(2, "Two", arxiv_id="2")], "sent": []}
    for d in (d1, d2):
        store.save_digest(d, tmp_path)
        store.record_pending(d, tmp_path)
    store.set_status(d1["candidates"][0]["id"], "kept", tmp_path)
    loads = []
    orig = store.load_digest
    monkeypatch.setattr(store, "load_digest", lambda d, root=None: loads.append(d) or orig(d, root))
    assert [c["title"] for c in store.pending_candidates(tmp_path)] == ["Two"]
    assert loads == ["2026-10-16"]


def test_save_follow_keeps_only_explicit_settings(tmp_path, monkeypatch):
    f = store.load_follow(tmp_path)
    f["unfollowed"] = ["ESG"]
    store.save_follow(f, tmp_path)
    raw = json.loads((tmp_path / "follow.json").read_text(encoding="utf-8"))
    assert "min_fit" not in raw and raw["unfollowed"] == ["ESG"]
    monkeypatch.setitem(store.DEFAULTS, "min_fit", 0.42)
    assert store.load_follow(tmp_path)["min_fit"] == 0.42
    f["min_fit"] = 0.6
    store.save_follow(f, tmp_path)
    assert store.load_follow(tmp_path)["min_fit"] == 0.6


def test_write_json_leaves_the_old_file_intact_if_the_final_step_fails(tmp_path, monkeypatch):
    store.save_ledger({"a": 1}, tmp_path)
    monkeypatch.setattr(store.os, "replace", lambda *a, **k: _raise(OSError("disk full")))
    with pytest.raises(OSError):
        store.save_ledger({"b": 2}, tmp_path)
    assert store.load_ledger(tmp_path) == {"a": 1}
    assert not list(tmp_path.glob("*.tmp"))
