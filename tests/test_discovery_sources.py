"""Tests for discovery sources: arXiv, SSRN and OpenAlex parsing and fetching."""
import json
from pathlib import Path

from research_library.discovery import sources

FIX = Path(__file__).parent / "fixtures"


# ── Parsers ───────────────────────────────────────────────────────────────────

def test_last_first():
    assert sources.last_first("Jane Q. Doe") == "Doe, Jane Q."
    assert sources.last_first("Plato") == "Plato"
    assert sources.last_first(" Doe, Jane ") == "Doe, Jane"


def test_parse_arxiv_atom():
    cands = sources.parse_arxiv_atom((FIX / "arxiv_sample.xml").read_text(encoding="utf-8"))
    assert len(cands) == 2
    c = cands[0]
    assert c["source"] == "arxiv" and c["arxiv_id"] == "2610.22144"
    assert c["doi"] == "10.48550/arXiv.2610.22144" and c["pdf_url"] == "https://arxiv.org/pdf/2610.22144v1"
    assert c["authors"] == ["Lovelace, Ada", "Babbage, Charles"]
    assert c["published"] == "2026-10-15" and c["year"] == "2026" and c["venue"] == "arXiv q-fin.PM"
    assert "  " not in c["title"] and "\n" not in c["title"]
    assert cands[1]["doi"] == "10.1000/abc"


def test_parse_ssrn_page_skips_bad_dates():
    cands = sources.parse_ssrn_page(json.loads((FIX / "ssrn_sample.json").read_text(encoding="utf-8")))
    assert [c["ssrn_id"] for c in cands] == ["7620718", "7619999"]
    c = cands[0]
    assert c["source"] == "ssrn" and c["published"] == "2026-10-16" and c["year"] == "2026"
    assert c["abstract"] == "" and c["venue"] == "SSRN" and c["extra"]["pages"] == 52 and c["extra"]["downloads"] == 11
    assert c["authors"][0].count(",") == 1


def test_openalex_abstract_and_work():
    assert sources.rebuild_abstract({"We": [0], "study": [1], "it.": [2]}) == "We study it."
    assert sources.rebuild_abstract(None) == ""
    works = json.loads((FIX / "openalex_sample.json").read_text(encoding="utf-8"))["results"]
    a, b = (sources.parse_openalex_work(w) for w in works)
    assert a["source"] == "openalex" and a["doi"] == "10.2139/ssrn.1234"
    assert a["abstract"] == "We study private equity cash flows." and a["venue"] == "SSRN Electronic Journal"
    assert a["pdf_url"].startswith("http") and len(a["authors"]) == 2 and "," in a["authors"][0]
    assert b["arxiv_id"] == "2610.00002" and b["abstract"] == ""


# ── Fetching, merging, owned check ────────────────────────────────────────────

from datetime import date, datetime  # noqa: E402


def _fake_http(monkeypatch, routes):
    calls = []

    def fake(url, headers=None, timeout=30):
        calls.append(url)
        for prefix, body in routes.items():
            if url.startswith(prefix):
                return body if isinstance(body, bytes) else body.encode("utf-8")
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(sources, "_http_get", fake)
    return calls


def test_fetch_arxiv_builds_firehose_and_topic_queries(monkeypatch):
    calls = _fake_http(monkeypatch, {"http://export.arxiv.org/api/query": (FIX / "arxiv_sample.xml").read_text(encoding="utf-8")})
    out = sources.fetch_arxiv(datetime(2026, 10, 14, 6, 30), datetime(2026, 10, 16, 6, 30),
                              {"PE": ["private equity", "buyout returns"]}, pause=0, sleep=lambda s: None)
    assert len(calls) == 2
    assert "cat:q-fin.*" in calls[0] and "submittedDate" in calls[0] and "202610140630" in calls[0]
    assert "all:%22private+equity%22" in calls[1] or 'all:"private equity"' in calls[1]
    assert len(out) == 4  # both fixture entries, twice; merge happens later


def test_fetch_ssrn_pages_until_window_edge(monkeypatch):
    page = (FIX / "ssrn_sample.json").read_text(encoding="utf-8")
    calls = _fake_http(monkeypatch, {"https://api.ssrn.com/content/v1/bindings/203/papers": page})
    out = sources.fetch_ssrn([203], since=date(2026, 10, 10), sleep=lambda s: None)
    assert len(calls) == 1 and "index=0" in calls[0] and "count=200" in calls[0]
    assert [c["ssrn_id"] for c in out] == ["7620718"]


def test_fetch_openalex_query_shape(monkeypatch):
    calls = _fake_http(monkeypatch, {"https://api.openalex.org/works": (FIX / "openalex_sample.json").read_text(encoding="utf-8")})
    out = sources.fetch_openalex({"PE": ["private equity"]}, since=date(2026, 10, 14), mailto="me@example.org")
    assert len(out) == 2 and "mailto=me%40example.org" in calls[0]
    assert "from_publication_date%3A2026-10-14" in calls[0] and "title_and_abstract.search" in calls[0]


def test_merge_and_drop_owned():
    a = {"source": "arxiv", "title": "Deep Hedging", "doi": "10.1/x", "abstract": "", "arxiv_id": "1", "ssrn_id": "",
         "authors": [], "year": "", "published": "", "url": "", "pdf_url": "u", "venue": "", "extra": {}}
    b = {**a, "source": "openalex", "abstract": "Full abstract", "pdf_url": ""}
    c = {**a, "doi": "", "arxiv_id": "", "ssrn_id": "9", "title": "Deep Hedging!"}
    d = {**a, "doi": "10.9/owned", "arxiv_id": "", "title": "Owned Paper"}
    merged = sources.merge_candidates([a, b, c, d])
    assert len(merged) == 2
    kept = next(m for m in merged if m["doi"] == "10.1/x")
    assert kept["abstract"] == "Full abstract" and kept["pdf_url"] == "u"
    index = {"categories": {"T": {"papers": [{"key": "K1", "title": "Owned paper", "doi": ""}]}}}
    out = sources.drop_owned(merged, index, ledger={})
    assert [m["doi"] for m in out] == ["10.1/x"]
    assert sources.drop_owned(out, {"categories": {}}, ledger={out[0]["id"]: {"status": "dismissed"}}) == []
