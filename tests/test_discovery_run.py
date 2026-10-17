"""Tests for the daily discovery run."""
import json
from datetime import datetime

import numpy as np

from research_library.discovery import run, store


def test_compute_window():
    now = datetime(2026, 10, 16, 6, 30)
    assert run.compute_window(None, now, None) == (datetime(2026, 10, 14, 6, 30), now)
    assert run.compute_window("2026-10-15T20:00:00", now, None)[0] == datetime(2026, 10, 15, 6, 30)
    assert run.compute_window("2026-09-01T06:30:00", now, None)[0] == datetime(2026, 10, 9, 6, 30)
    assert run.compute_window(None, now, 3)[0] == datetime(2026, 10, 13, 6, 30)


def _setup(tmp_path):
    index = {"generated_at": "x", "unique_papers": 2, "categories": {
        "Big": {"summary": "s", "subcategories": {}, "papers": [
            {"key": "B1", "title": "Owned", "authors": ["Doe, J"], "year": "2020", "abstract": "a", "doi": "10.9/owned"}]},
        "Small": {"summary": "s", "subcategories": {}, "papers": [
            {"key": "S1", "title": "Small one", "authors": ["Roe, R"], "year": "2021", "abstract": "b", "doi": ""}]}}}
    ip = tmp_path / "library_index.json"
    ip.write_text(json.dumps(index), encoding="utf-8")
    v = np.eye(4, dtype=np.float32)[:2]
    library = {"keys": ["B1", "S1"], "vectors": v, "paper_cats": ["Big", "Small"], "cat_names": ["Big", "Small"], "cat_vectors": v}
    return ip, library


def _cand(title, **kw):
    c = {"source": "arxiv", "title": title, "authors": ["Doe, Jane"], "year": "2026", "published": "2026-10-15", "abstract": "abs",
         "doi": "", "url": "u", "pdf_url": "", "arxiv_id": title, "ssrn_id": "", "venue": "arXiv q-fin.PM", "extra": {}}
    c.update(kw)
    return c


def _raise(exc):
    raise exc


def test_discover_writes_digest_pending_and_last_run(tmp_path, monkeypatch):
    ip, library = _setup(tmp_path)
    root = tmp_path / "disc"
    fetchers = {"arxiv": lambda **kw: [_cand("New A"), _cand("Owned", doi="10.9/owned")],
                "ssrn": lambda **kw: [_cand("New S", source="ssrn", abstract="", ssrn_id="7", arxiv_id="")],
                "openalex": lambda **kw: _raise(RuntimeError("down"))}
    monkeypatch.setattr(run.summaries, "ensure_queries", lambda *a, **k: {"Big": ["q"], "Small": ["q"]})
    calls = []

    def fake_summary(client, cand, nearest):
        calls.append(cand["title"])
        if cand["title"] == "New S":
            raise RuntimeError("boom")
        return {"summary": "S", "why_it_fits": "W", "recommendation": "keep", "reason": "R"}

    monkeypatch.setattr(run.summaries, "summarize_candidate", fake_summary)
    embed = lambda texts: np.array([[1, 0.2, 0, 0] if "New A" in t else [0.2, 1, 0, 0] for t in texts], dtype=np.float32)  # noqa: E731
    monkeypatch.setenv("UNPAYWALL_EMAIL", "me@example.org")
    out = run.discover(progress=lambda m: None, client=object(), now=datetime(2026, 10, 16, 6, 30), root=root,
                       index_path=ip, library=library, embed=embed, fetchers=fetchers)
    assert out["fetched"] == {"arxiv": 2, "ssrn": 1, "openalex": 0} and out["new"] == 2 and out["shown"] == 2
    assert out["by_topic"] == {"Big": 1, "Small": 1} and out["digest"] == "2026-10-16"
    digest = store.load_digest("2026-10-16", root)
    titles = {c["title"]: c for c in digest["candidates"]}
    assert titles["New A"]["topic"] == "Big" and titles["New A"]["summary"] == "S" and len(titles["New A"]["vector"]) == 4
    assert titles["New S"]["summary"] is None and titles["New S"]["reason"].startswith("summary failed")
    assert all(e["status"] == "pending" for e in store.load_ledger(root).values()) and len(store.load_ledger(root)) == 2
    assert store.load_follow(root)["last_run"] == "2026-10-16T06:30:00"


def test_dry_run_writes_nothing_and_skips_summaries(tmp_path, monkeypatch):
    ip, library = _setup(tmp_path)
    root = tmp_path / "disc"
    monkeypatch.setattr(run.summaries, "ensure_queries", lambda *a, **k: {"Big": ["q"], "Small": ["q"]})
    monkeypatch.setattr(run.summaries, "summarize_candidate", lambda *a, **k: _raise(AssertionError("should not summarise")))
    fetchers = {"arxiv": lambda **kw: [_cand("New A")], "ssrn": lambda **kw: [], "openalex": lambda **kw: []}
    out = run.discover(dry_run=True, progress=lambda m: None, client=object(), now=datetime(2026, 10, 16, 6, 30), root=root,
                       index_path=ip, library=library,
                       embed=lambda texts: np.array([[1, 0, 0, 0]] * len(texts), dtype=np.float32), fetchers=fetchers)
    assert out["shown"] == 1 and out["digest"] is None and out["selected"][0]["title"] == "New A"
    assert not root.exists() or not list(root.glob("*.json"))
