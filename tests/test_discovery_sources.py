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
