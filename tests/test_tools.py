"""Tests for tools.py — MCP tool definitions and handler dispatch."""
import json
import pytest
from unittest.mock import patch, AsyncMock

import sys
sys.path.insert(0, "src")

from research_library.tools import TOOLS, handle_tool


# ── Tool registry ─────────────────────────────────────────────────────────────

class TestToolRegistry:
    EXPECTED_TOOLS = {
        "search_papers", "get_paper", "list_collections", "get_collection_papers",
        "get_recent_papers", "get_annotations", "get_tags", "get_papers_by_tag",
        "get_fulltext", "get_fulltext_batch",
        "semantic_search", "list_categories", "get_category_papers", "ask_library",
        "get_nearest_papers",
        "add_paper", "add_papers_batch", "add_to_collection", "remove_from_collection",
        "add_tag", "remove_tag", "update_paper", "attach_pdf", "create_collection",
        "get_new_papers", "keep_new_papers", "dismiss_new_papers", "consider_paper",
        "get_followed_topics", "set_followed_topics",
    }

    def test_all_expected_tools_registered(self):
        names = {t.name for t in TOOLS}
        assert names == self.EXPECTED_TOOLS

    def test_all_tools_have_descriptions(self):
        for tool in TOOLS:
            assert tool.description, f"{tool.name} has no description"

    def test_required_fields_defined(self):
        required_map = {
            "search_papers": ["query"],
            "get_paper": ["key"],
            "get_collection_papers": ["collection_name"],
            "get_annotations": ["key"],
            "get_papers_by_tag": ["tag"],
            "get_fulltext": ["key"],
            "get_fulltext_batch": ["keys"],
            "semantic_search": ["query"],
            "get_category_papers": ["category"],
            "ask_library": ["question"],
            "add_paper": ["title", "authors"],
            "add_papers_batch": ["papers"],
            "add_to_collection": ["item_key", "collection_key"],
            "remove_from_collection": ["item_key", "collection_key"],
            "add_tag": ["item_key", "tag"],
            "remove_tag": ["item_key", "tag"],
            "update_paper": ["item_key"],
            "attach_pdf": ["item_key", "pdf_path"],
            "create_collection": ["name"],
        }
        tool_dict = {t.name: t for t in TOOLS}
        for tool_name, required in required_map.items():
            schema = tool_dict[tool_name].inputSchema
            assert schema.get("required") == required, \
                f"{tool_name}: expected required={required}, got {schema.get('required')}"

    def test_optional_tools_have_no_required(self):
        tool_dict = {t.name: t for t in TOOLS}
        for name in ("list_collections", "get_tags", "get_recent_papers", "list_categories"):
            schema = tool_dict[name].inputSchema
            assert "required" not in schema or schema["required"] == []


# ── Tool handler dispatch ─────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestHandleTool:
    async def test_search_papers_returns_json(self):
        fake = [{"key": "K1", "title": "Paper A", "authors": []}]
        with patch("research_library.tools.zc.search_papers", return_value=fake):
            result = await handle_tool("search_papers", {"query": "test", "limit": 5})
        data = json.loads(result[0].text)
        assert data[0]["key"] == "K1"

    async def test_search_papers_empty_returns_message(self):
        with patch("research_library.tools.zc.search_papers", return_value=[]):
            result = await handle_tool("search_papers", {"query": "xyzzy"})
        assert "No papers found" in result[0].text

    async def test_get_paper_found(self):
        fake = {"key": "K1", "title": "Found Paper"}
        with patch("research_library.tools.zc.get_paper", return_value=fake):
            result = await handle_tool("get_paper", {"key": "K1"})
        data = json.loads(result[0].text)
        assert data["title"] == "Found Paper"

    async def test_get_paper_not_found(self):
        with patch("research_library.tools.zc.get_paper", return_value=None):
            result = await handle_tool("get_paper", {"key": "MISSING"})
        assert "No paper found" in result[0].text

    async def test_list_collections(self):
        fake = [{"key": "C1", "name": "Finance", "num_items": 10}]
        with patch("research_library.tools.zc.list_collections", return_value=fake):
            result = await handle_tool("list_collections", {})
        data = json.loads(result[0].text)
        assert data[0]["name"] == "Finance"

    async def test_get_recent_papers(self):
        fake = [{"key": "K1", "title": "Recent"}]
        with patch("research_library.tools.zc.get_recent_papers", return_value=fake):
            result = await handle_tool("get_recent_papers", {"limit": 5})
        data = json.loads(result[0].text)
        assert len(data) == 1

    async def test_get_tags(self):
        with patch("research_library.tools.zc.get_tags", return_value=["alpha", "beta"]):
            result = await handle_tool("get_tags", {})
        assert "alpha" in result[0].text

    async def test_get_fulltext_success(self):
        fake_ft = {
            "content": "The full paper text here.",
            "source": "zotero_index",
            "truncated": False,
            "total_chars": 24,
            "error": None,
        }
        with patch("research_library.tools.zc.get_fulltext", return_value=fake_ft):
            result = await handle_tool("get_fulltext", {"key": "K1"})
        assert "The full paper text here." in result[0].text

    async def test_get_fulltext_not_available(self):
        fake_ft = {"content": None, "source": "none", "truncated": False, "error": "not_indexed"}
        with patch("research_library.tools.zc.get_fulltext", return_value=fake_ft):
            result = await handle_tool("get_fulltext", {"key": "K1"})
        assert "not_indexed" in result[0].text

    async def test_get_fulltext_truncation_noted(self):
        fake_ft = {
            "content": "x" * 500,
            "source": "zotero_index",
            "truncated": True,
            "total_chars": 50000,
            "error": None,
        }
        with patch("research_library.tools.zc.get_fulltext", return_value=fake_ft):
            result = await handle_tool("get_fulltext", {"key": "K1"})
        assert "truncated" in result[0].text

    async def test_get_fulltext_batch(self):
        fake_papers = [
            {"key": "K1", "fulltext": {"content": "text1", "truncated": False}},
            {"key": "K2", "fulltext": {"content": None, "error": "not_indexed"}},
        ]
        with patch("research_library.tools.zc.get_fulltext_batch", return_value=fake_papers):
            result = await handle_tool("get_fulltext_batch", {"keys": ["K1", "K2"]})
        data = json.loads(result[0].text)
        assert data[0]["content"] == "text1"
        assert data[1]["error"] == "not_indexed"

    async def test_unknown_tool_returns_message(self):
        result = await handle_tool("nonexistent_tool", {})
        assert "Unknown tool" in result[0].text


# ── Semantic-index staleness flag ─────────────────────────────────────────────

@pytest.mark.asyncio
class TestIndexStalenessFlag:
    """Writes land in Zotero immediately but reach semantic_search only after
    `research-admin all`, so paper writes must say so."""

    async def test_add_paper_flags_stale_index(self):
        created = {"key": "K1", "title": "New", "duplicate": False}
        with patch("research_library.tools.zc.create_paper", return_value=created):
            result = await handle_tool("add_paper", {"title": "New", "authors": ["A"]})
        data = json.loads(result[0].text)
        assert data["semantic_index_stale"] is True
        assert "research-admin all" in data["index_note"]

    async def test_no_flag_when_paper_was_a_duplicate(self):
        existing = {"key": "K1", "title": "Old", "duplicate": True}
        with patch("research_library.tools.zc.create_paper", return_value=existing):
            result = await handle_tool("add_paper", {"title": "Old", "authors": ["A"]})
        data = json.loads(result[0].text)
        assert "semantic_index_stale" not in data

    async def test_batch_flags_only_when_something_created(self):
        nothing = {"created": [], "duplicates": [{"key": "K1"}], "failed": []}
        with patch("research_library.tools.zc.create_papers_batch", return_value=nothing):
            result = await handle_tool("add_papers_batch", {"papers": [{"title": "X"}]})
        assert "semantic_index_stale" not in json.loads(result[0].text)

        some = {"created": [{"key": "K2"}], "duplicates": [], "failed": []}
        with patch("research_library.tools.zc.create_papers_batch", return_value=some):
            result = await handle_tool("add_papers_batch", {"papers": [{"title": "X"}]})
        assert json.loads(result[0].text)["semantic_index_stale"] is True

    async def test_update_flags_only_for_indexed_fields(self):
        with patch("research_library.tools.zc.update_paper", return_value={"key": "K1"}):
            result = await handle_tool("update_paper", {"item_key": "K1", "title": "T"})
        assert json.loads(result[0].text)["semantic_index_stale"] is True

        with patch("research_library.tools.zc.update_paper", return_value={"key": "K1"}):
            result = await handle_tool("update_paper", {"item_key": "K1", "pages": "1-2"})
        assert "semantic_index_stale" not in json.loads(result[0].text)

    async def test_tag_and_collection_writes_are_not_flagged(self):
        """Categories come from paper metadata, not Zotero tags or collections."""
        with patch("research_library.tools.zc.add_tag", return_value={"ok": True}):
            result = await handle_tool("add_tag", {"item_key": "K1", "tag": "t"})
        assert "semantic_index_stale" not in json.loads(result[0].text)

        with patch("research_library.tools.zc.add_to_collection", return_value={"ok": True}):
            result = await handle_tool("add_to_collection",
                                       {"item_key": "K1", "collection_key": "C1"})
        assert "semantic_index_stale" not in json.loads(result[0].text)
