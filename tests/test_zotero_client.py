"""
Tests for zotero_client.py.

Unit tests mock the pyzotero client; integration tests hit the live Zotero API.
Run integration tests with: pytest -m integration
Run unit tests only with:   pytest -m "not integration"
"""
import pytest
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, "src")

from research_library.zotero_client import (
    _format_item,
    get_fulltext_batch,
    PYMUPDF_AVAILABLE,
)
from tests.conftest import KNOWN_PAPER_KEY, KNOWN_PAPER_TITLE


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_raw_item(key="ABC123", title="Test Paper", item_type="journalArticle",
                   date="2023-01-01", abstract="Test abstract.", doi="10.1/test",
                   authors=None, tags=None):
    return {
        "data": {
            "key": key,
            "title": title,
            "itemType": item_type,
            "date": date,
            "abstractNote": abstract,
            "DOI": doi,
            "url": "",
            "publicationTitle": "Test Journal",
            "creators": authors or [
                {"creatorType": "author", "lastName": "Smith", "firstName": "John"},
                {"creatorType": "author", "lastName": "Jones", "firstName": "Alice"},
                {"creatorType": "editor", "lastName": "Brown", "firstName": "Bob"},
            ],
            "tags": tags or [{"tag": "finance"}, {"tag": "optimization"}],
            "collections": ["COL1"],
        }
    }


# ── Unit: _format_item ────────────────────────────────────────────────────────

class TestFormatItem:
    def test_basic_fields(self):
        raw = _make_raw_item()
        result = _format_item(raw)
        assert result["key"] == "ABC123"
        assert result["title"] == "Test Paper"
        assert result["year"] == "2023"
        assert result["abstract"] == "Test abstract."
        assert result["doi"] == "10.1/test"
        assert result["publication"] == "Test Journal"

    def test_authors_only_includes_authors_not_editors(self):
        raw = _make_raw_item()
        result = _format_item(raw)
        assert result["authors"] == ["Smith, John", "Jones, Alice"]
        assert len(result["authors"]) == 2  # editor excluded

    def test_tags_extracted(self):
        raw = _make_raw_item(tags=[{"tag": "macro"}, {"tag": "regime"}])
        result = _format_item(raw)
        assert result["tags"] == ["macro", "regime"]

    def test_missing_date_returns_empty_year(self):
        raw = _make_raw_item(date="")
        raw["data"]["date"] = ""
        result = _format_item(raw)
        assert result["year"] == ""

    def test_no_title_returns_placeholder(self):
        raw = _make_raw_item()
        raw["data"]["title"] = ""
        result = _format_item(raw)
        assert result["title"] == "(no title)"

    def test_empty_item_does_not_raise(self):
        result = _format_item({"data": {}})
        assert result["key"] == ""
        assert result["title"] == "(no title)"
        assert result["authors"] == []


# ── Unit: get_fulltext_batch ──────────────────────────────────────────────────

class TestGetFulltextBatch:
    def _make_paper(self, key):
        return {"key": key, "title": f"Paper {key}", "authors": [], "abstract": ""}

    def test_batch_adds_fulltext_key_to_papers(self):
        papers = [self._make_paper("K1"), self._make_paper("K2")]
        fake_ft = {"content": "full text here", "source": "zotero_index",
                   "truncated": False, "error": None, "total_chars": 14}
        with patch("research_library.zotero_client.get_fulltext", return_value=fake_ft):
            result = get_fulltext_batch(papers, max_papers=2, char_limit=1000)
        assert len(result) == 2
        assert result[0]["fulltext"]["content"] == "full text here"
        assert result[1]["fulltext"]["content"] == "full text here"

    def test_batch_respects_max_papers(self):
        papers = [self._make_paper(f"K{i}") for i in range(5)]
        fake_ft = {"content": "text", "source": "zotero_index",
                   "truncated": False, "error": None, "total_chars": 4}
        with patch("research_library.zotero_client.get_fulltext", return_value=fake_ft) as mock_ft:
            result = get_fulltext_batch(papers, max_papers=2, char_limit=100)
        assert mock_ft.call_count == 2
        # Papers beyond max don't get a fulltext key
        for p in result[2:]:
            assert "fulltext" not in p

    def test_batch_handles_individual_failures_gracefully(self):
        papers = [self._make_paper("K1"), self._make_paper("K2")]
        def side_effect(key, char_limit):
            if key == "K1":
                raise Exception("API timeout")
            return {"content": "text", "source": "zotero_index",
                    "truncated": False, "error": None, "total_chars": 4}
        with patch("research_library.zotero_client.get_fulltext", side_effect=side_effect):
            result = get_fulltext_batch(papers, max_papers=2)
        # K1 failed but K2 should still have content
        assert result[0]["fulltext"]["error"] is not None
        assert result[1]["fulltext"]["content"] == "text"


# ── Unit: get_fulltext error paths ────────────────────────────────────────────

class TestGetFulltextErrorPaths:
    def test_no_attachment_returns_error(self):
        from research_library.zotero_client import get_fulltext
        with patch("research_library.zotero_client.get_attachment_key", return_value=None):
            result = get_fulltext("ANYKEY")
        assert result["content"] is None
        assert result["error"] == "no_attachment"

    def test_empty_content_returns_not_indexed(self):
        from research_library.zotero_client import get_fulltext
        mock_zot = MagicMock()
        mock_zot.fulltext_item.return_value = {"content": ""}
        with patch("research_library.zotero_client.get_attachment_key", return_value="ATTKEY"), \
             patch("research_library.zotero_client.get_client", return_value=mock_zot):
            result = get_fulltext("ANYKEY")
        assert result["error"] == "not_indexed"

    def test_content_is_truncated_when_over_limit(self):
        from research_library.zotero_client import get_fulltext
        long_text = "x" * 10000
        mock_zot = MagicMock()
        mock_zot.fulltext_item.return_value = {"content": long_text}
        with patch("research_library.zotero_client.get_attachment_key", return_value="ATTKEY"), \
             patch("research_library.zotero_client.get_client", return_value=mock_zot):
            result = get_fulltext("ANYKEY", char_limit=500)
        assert result["truncated"] is True
        assert len(result["content"]) == 500
        assert result["total_chars"] == 10000


# ── Integration: live Zotero API ──────────────────────────────────────────────

@pytest.mark.integration
class TestZoteroClientIntegration:
    def test_client_connects(self):
        from research_library.zotero_client import get_client
        zot = get_client()
        assert zot is not None

    def test_search_papers_returns_results(self):
        from research_library.zotero_client import search_papers
        results = search_papers("asset allocation", limit=5)
        assert len(results) > 0
        assert all("title" in p for p in results)
        assert all("authors" in p for p in results)
        assert all("key" in p for p in results)

    def test_search_returns_no_attachments(self):
        from research_library.zotero_client import search_papers
        results = search_papers("portfolio", limit=10)
        for p in results:
            assert p["item_type"] != "attachment"

    def test_get_paper_known_key(self):
        from research_library.zotero_client import get_paper
        paper = get_paper(KNOWN_PAPER_KEY)
        assert paper is not None
        assert KNOWN_PAPER_TITLE in paper["title"]

    def test_get_paper_invalid_key_returns_none(self):
        from research_library.zotero_client import get_paper
        result = get_paper("XXXXXXXX")
        assert result is None

    def test_list_collections_returns_list(self):
        from research_library.zotero_client import list_collections
        cols = list_collections()
        assert isinstance(cols, list)
        assert len(cols) > 0
        assert all("name" in c for c in cols)
        assert all("key" in c for c in cols)

    def test_get_recent_papers(self):
        from research_library.zotero_client import get_recent_papers
        papers = get_recent_papers(limit=5)
        assert len(papers) == 5
        assert all("title" in p for p in papers)

    def test_get_tags_returns_sorted_list(self):
        from research_library.zotero_client import get_tags
        tags = get_tags()
        assert isinstance(tags, list)
        assert tags == sorted(tags)

    def test_get_attachment_key_for_known_paper(self):
        from research_library.zotero_client import get_attachment_key
        att_key = get_attachment_key(KNOWN_PAPER_KEY)
        assert att_key is not None
        assert isinstance(att_key, str)
        assert len(att_key) == 8  # Zotero keys are 8 chars

    def test_get_fulltext_known_paper(self):
        from research_library.zotero_client import get_fulltext
        result = get_fulltext(KNOWN_PAPER_KEY)
        assert result["error"] is None
        assert result["content"] is not None
        assert len(result["content"]) > 100
        assert result["source"] == "zotero_index"

    def test_get_fulltext_char_limit_respected(self):
        from research_library.zotero_client import get_fulltext
        result = get_fulltext(KNOWN_PAPER_KEY, char_limit=500)
        assert len(result["content"]) == 500
        assert result["truncated"] is True

    def test_get_fulltext_batch_integration(self):
        from research_library.zotero_client import search_papers, get_fulltext_batch
        papers = search_papers("optimization", limit=3)
        batch = get_fulltext_batch(papers, max_papers=2, char_limit=200)
        papers_with_ft = [p for p in batch if p.get("fulltext", {}).get("content")]
        assert len(papers_with_ft) >= 1
        assert all(len(p["fulltext"]["content"]) <= 200 for p in papers_with_ft)

    def test_pymupdf_flag_is_bool(self):
        assert isinstance(PYMUPDF_AVAILABLE, bool)
