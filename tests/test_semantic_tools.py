"""Tests for the four index-aware / semantic MCP tools."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, "src")


@pytest.fixture
def fake_index_file(tmp_path: Path, monkeypatch) -> Path:
    """Write a fake library_index.json and repoint tools.INDEX_CACHE_PATH at it."""
    index = {
        "generated_at": "2026-01-01T00:00:00",
        "total_fetched": 3,
        "unique_papers": 3,
        "duplicates_removed": 0,
        "categories": {
            "Regime Switching": {
                "summary": "Models with regime shifts.",
                "subcategories": {
                    "Hidden Markov": ["K1"],
                    "Threshold Models": ["K2"],
                },
                "papers": [
                    {"key": "K1", "title": "HMM paper", "authors": ["Smith"],
                     "year": "2020", "abstract": "about HMM", "doi": "10.1/a"},
                    {"key": "K2", "title": "Threshold paper", "authors": ["Jones"],
                     "year": "2021", "abstract": "about thresholds", "doi": "10.1/b"},
                ],
            },
            "Risk Premia": {
                "summary": "Time-varying risk premia.",
                "subcategories": {},
                "papers": [
                    {"key": "K3", "title": "Risk premia paper", "authors": ["Brown"],
                     "year": "2019", "abstract": "about premia", "doi": "10.1/c"},
                ],
            },
        },
    }
    path = tmp_path / "library_index.json"
    path.write_text(json.dumps(index), encoding="utf-8")

    from research_library import tools
    monkeypatch.setattr(tools, "INDEX_CACHE_PATH", path)
    return path


# ── list_categories ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestListCategories:
    async def test_returns_categories_with_subcategories(self, fake_index_file):
        from research_library.tools import handle_tool
        result = await handle_tool("list_categories", {})
        data = json.loads(result[0].text)
        names = {c["name"] for c in data}
        assert names == {"Regime Switching", "Risk Premia"}
        rs = next(c for c in data if c["name"] == "Regime Switching")
        assert rs["paper_count"] == 2
        sub_names = {s["name"] for s in rs["subcategories"]}
        assert sub_names == {"Hidden Markov", "Threshold Models"}

    async def test_missing_cache_returns_helpful_message(self, tmp_path, monkeypatch):
        from research_library import tools
        monkeypatch.setattr(tools, "INDEX_CACHE_PATH", tmp_path / "missing.json")
        result = await tools.handle_tool("list_categories", {})
        assert "No library index cached" in result[0].text


# ── get_category_papers ───────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestGetCategoryPapers:
    async def test_returns_papers_for_known_category(self, fake_index_file):
        from research_library.tools import handle_tool
        result = await handle_tool("get_category_papers", {"category": "Regime Switching"})
        data = json.loads(result[0].text)
        assert {p["key"] for p in data} == {"K1", "K2"}

    async def test_subcategory_filter_narrows_results(self, fake_index_file):
        from research_library.tools import handle_tool
        result = await handle_tool("get_category_papers",
                                   {"category": "Regime Switching",
                                    "subcategory": "Hidden Markov"})
        data = json.loads(result[0].text)
        assert [p["key"] for p in data] == ["K1"]

    async def test_unknown_category(self, fake_index_file):
        from research_library.tools import handle_tool
        result = await handle_tool("get_category_papers", {"category": "Nonexistent"})
        assert "not found" in result[0].text

    async def test_unknown_subcategory(self, fake_index_file):
        from research_library.tools import handle_tool
        result = await handle_tool("get_category_papers",
                                   {"category": "Regime Switching", "subcategory": "Whatever"})
        assert "Subcategory 'Whatever' not found" in result[0].text


# ── semantic_search ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestSemanticSearch:
    async def test_hydrates_hits_from_cached_index(self, fake_index_file):
        from research_library.tools import handle_tool

        fake_hits = [
            {"key": "K2", "score": 0.92, "category": "Regime Switching"},
            {"key": "K3", "score": 0.81, "category": "Risk Premia"},
        ]
        with patch("research_library.embeddings.search", return_value=fake_hits):
            result = await handle_tool("semantic_search",
                                       {"query": "threshold risk premium"})
        data = json.loads(result[0].text)
        assert [p["key"] for p in data] == ["K2", "K3"]
        assert data[0]["category"] == "Regime Switching"
        assert data[0]["score"] == 0.92

    async def test_missing_embeddings_returns_message(self, fake_index_file):
        from research_library.tools import handle_tool
        with patch("research_library.embeddings.search",
                   side_effect=FileNotFoundError("no embeddings.npz")):
            result = await handle_tool("semantic_search", {"query": "x"})
        assert "no embeddings.npz" in result[0].text


# ── ask_library ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAskLibrary:
    async def test_end_to_end_with_mocks(self, fake_index_file):
        from research_library.tools import handle_tool

        ranking = {
            "categories": [{"name": "Regime Switching", "score": 0.88}],
            "papers": [
                {"key": "K1", "score": 0.87, "category": "Regime Switching"},
                {"key": "K2", "score": 0.85, "category": "Regime Switching"},
            ],
        }
        # Shaped like a Sonnet 5.5 reply: an (empty) thinking block comes first.
        fake_msg = SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="", signature="sig"),
                SimpleNamespace(type="text", text="HMM models dominate [K1]; thresholds complement [K2]."),
            ],
            stop_reason="end_turn",
            stop_details=None,
        )
        sent = {}

        def create(**kwargs):
            sent.update(kwargs)
            return fake_msg

        fake_client = SimpleNamespace(messages=SimpleNamespace(create=create))

        with patch("research_library.embeddings.search_by_category", return_value=ranking), \
             patch("anthropic.Anthropic", return_value=fake_client):
            result = await handle_tool("ask_library",
                                       {"question": "what does the library say about regimes?"})

        data = json.loads(result[0].text)
        assert "HMM models dominate [K1]" in data["answer"]
        assert data["categories_consulted"][0]["name"] == "Regime Switching"
        assert sent["model"] == "claude-sonnet-5-5"
        assert sent["output_config"] == {"effort": "medium"}
        assert {p["key"] for p in data["papers_consulted"]} == {"K1", "K2"}
        assert data["used_fulltext_for"] == []

    async def test_fulltext_looks_past_the_top_hits(self, fake_index_file):
        """The top semantic hits often have no PDF. Slicing to the top 3 before
        fetching meant ask_library answered from abstracts alone even when a
        slightly lower-ranked paper was available in full."""
        from research_library.tools import handle_tool

        ranking = {
            "categories": [{"name": "Regime Switching", "score": 0.9}],
            "papers": [
                {"key": "K1", "score": 0.90, "category": "Regime Switching"},
                {"key": "K2", "score": 0.88, "category": "Regime Switching"},
                {"key": "K3", "score": 0.86, "category": "Risk Premia"},
                {"key": "K4", "score": 0.84, "category": "Risk Premia"},
            ],
        }
        k4 = {"key": "K4", "title": "Deep paper", "authors": ["Lee"],
              "year": "2022", "abstract": "about depth", "doi": "10.1/d"}

        def fake_fulltext(key, char_limit=3000):
            if key == "K4":
                return {"content": "the full body of K4", "source": "zotero_index",
                        "truncated": False, "error": None, "total_chars": 19}
            return {"content": None, "source": "none", "truncated": False,
                    "error": "no_attachment"}

        captured = {}

        def fake_create(**kwargs):
            captured["prompt"] = kwargs["messages"][0]["content"]
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="answer [K4]")],
                                   stop_reason="end_turn", stop_details=None)

        fake_client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))

        with patch("research_library.embeddings.search_by_category", return_value=ranking), \
             patch("research_library.tools.zc.get_paper", return_value=k4), \
             patch("research_library.zotero_client.get_fulltext", side_effect=fake_fulltext), \
             patch("anthropic.Anthropic", return_value=fake_client):
            result = await handle_tool("ask_library",
                                       {"question": "regimes?", "use_fulltext": True})

        data = json.loads(result[0].text)
        assert data["used_fulltext_for"] == ["K4"]
        assert "the full body of K4" in captured["prompt"]

    async def test_missing_embeddings_returns_message(self, fake_index_file):
        from research_library.tools import handle_tool
        with patch("research_library.embeddings.search_by_category",
                   side_effect=FileNotFoundError("no embeddings.npz")):
            result = await handle_tool("ask_library", {"question": "anything"})
        assert "no embeddings.npz" in result[0].text


# ── get_nearest_papers ────────────────────────────────────────────────────────

@pytest.fixture
def fake_embeddings():
    import numpy as np
    vectors = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return {"keys": ["K1", "K2", "K3"], "vectors": vectors,
            "paper_cats": ["Regime Switching", "Regime Switching", "Risk Premia"],
            "cat_names": [], "cat_vectors": np.zeros((0, 3))}


@pytest.mark.asyncio
class TestGetNearestPapers:
    async def test_ranks_neighbours_with_topic_and_link(self, fake_index_file, fake_embeddings):
        from research_library.tools import handle_tool
        with patch("research_library.embeddings.load_embeddings", return_value=fake_embeddings):
            result = await handle_tool("get_nearest_papers", {"key": "K1", "limit": 5})
        data = json.loads(result[0].text)
        assert data["paper"] == {"key": "K1", "title": "HMM paper", "link": "https://doi.org/10.1/a"}
        assert [n["key"] for n in data["nearest"]] == ["K2", "K3"]
        assert data["nearest"][0]["score"] > data["nearest"][1]["score"]
        assert data["nearest"][0]["topic"] == "Regime Switching"
        assert data["nearest"][0]["link"] == "https://doi.org/10.1/b"
        assert "not citation" in data["note"]

    async def test_limit_is_respected(self, fake_index_file, fake_embeddings):
        from research_library.tools import handle_tool
        with patch("research_library.embeddings.load_embeddings", return_value=fake_embeddings):
            result = await handle_tool("get_nearest_papers", {"key": "K1", "limit": 1})
        assert len(json.loads(result[0].text)["nearest"]) == 1

    async def test_unknown_key(self, fake_index_file, fake_embeddings):
        from research_library.tools import handle_tool
        with patch("research_library.embeddings.load_embeddings", return_value=fake_embeddings):
            result = await handle_tool("get_nearest_papers", {"key": "NOPE"})
        assert "not in the cached index" in result[0].text

    async def test_missing_embeddings(self, fake_index_file):
        from research_library.tools import handle_tool
        with patch("research_library.embeddings.load_embeddings", return_value=None):
            result = await handle_tool("get_nearest_papers", {"key": "K1"})
        assert "research-admin embed" in result[0].text
