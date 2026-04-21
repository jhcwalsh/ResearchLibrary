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
