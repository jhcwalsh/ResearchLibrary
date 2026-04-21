"""Tests for the embeddings module — mocks Voyage and file I/O."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, "src")


@pytest.fixture(autouse=True)
def _reset_voyage_client():
    from research_library import embeddings
    embeddings._client = None
    yield
    embeddings._client = None


@pytest.fixture
def fake_index(tmp_path: Path) -> Path:
    index = {
        "generated_at": "2026-01-01T00:00:00",
        "total_fetched": 3,
        "unique_papers": 3,
        "duplicates_removed": 0,
        "categories": {
            "Cat A": {
                "summary": "Papers about A.",
                "subcategories": {},
                "papers": [
                    {"key": "K1", "title": "Alpha paper", "abstract": "about alpha"},
                    {"key": "K2", "title": "Beta paper", "abstract": "about beta"},
                ],
            },
            "Cat B": {
                "summary": "Papers about B.",
                "subcategories": {},
                "papers": [
                    {"key": "K3", "title": "Gamma paper", "abstract": "about gamma"},
                ],
            },
        },
    }
    cache = tmp_path / "library_index.json"
    cache.write_text(json.dumps(index), encoding="utf-8")
    return cache


def _fake_embed_factory(dim: int = 8, seed: int = 0):
    """Return a deterministic fake Voyage embed that produces distinct vectors per input."""
    rng = np.random.default_rng(seed)

    def embed(texts, model=None, input_type=None):
        vectors = []
        for t in texts:
            h = abs(hash(t)) % (10**6)
            local = np.random.default_rng(h).normal(size=dim)
            vectors.append(local.astype(np.float32).tolist())
        return SimpleNamespace(embeddings=vectors)

    return embed


# ── Normalisation + search math ───────────────────────────────────────────────

class TestSearchMath:
    def test_normalize_zero_vector_is_safe(self):
        from research_library.embeddings import _normalize
        result = _normalize(np.zeros((1, 4), dtype=np.float32))
        assert not np.isnan(result).any()

    def test_cosine_ranks_correctly(self, tmp_path, fake_index, monkeypatch):
        from research_library import embeddings

        # Stub out Voyage — deterministic vectors per text
        monkeypatch.setattr(embeddings, "EMBEDDINGS_PATH", tmp_path / "emb.npz")
        monkeypatch.setattr(embeddings, "INDEX_CACHE_PATH", fake_index)
        fake_embed = _fake_embed_factory()
        monkeypatch.setattr(embeddings, "get_client",
                            lambda: SimpleNamespace(embed=fake_embed))

        embeddings.build_embeddings()

        # Query using the same text as paper K2 → it should rank first for that query.
        cache = embeddings.load_embeddings()
        assert len(cache["keys"]) == 3
        assert cache["vectors"].shape == (3, 8)

        results = embeddings.search("Beta paper\n\nabout beta", top_k=3)
        assert results[0]["key"] == "K2"
        assert results[0]["score"] > results[-1]["score"]


# ── Cache round-trip ──────────────────────────────────────────────────────────

class TestCacheRoundTrip:
    def test_save_and_load(self, tmp_path, fake_index, monkeypatch):
        from research_library import embeddings
        monkeypatch.setattr(embeddings, "EMBEDDINGS_PATH", tmp_path / "emb.npz")
        monkeypatch.setattr(embeddings, "INDEX_CACHE_PATH", fake_index)
        fake_embed = _fake_embed_factory()
        monkeypatch.setattr(embeddings, "get_client",
                            lambda: SimpleNamespace(embed=fake_embed))

        result = embeddings.build_embeddings()
        assert result["paper_count"] == 3
        assert result["category_count"] == 2

        cache = embeddings.load_embeddings()
        assert cache["keys"] == ["K1", "K2", "K3"]
        assert cache["cat_names"] == ["Cat A", "Cat B"]
        assert cache["paper_cats"] == ["Cat A", "Cat A", "Cat B"]

    def test_load_returns_none_when_missing(self, tmp_path, monkeypatch):
        from research_library import embeddings
        monkeypatch.setattr(embeddings, "EMBEDDINGS_PATH", tmp_path / "nothing.npz")
        assert embeddings.load_embeddings() is None

    def test_search_raises_when_cache_absent(self, tmp_path, monkeypatch):
        from research_library import embeddings
        monkeypatch.setattr(embeddings, "EMBEDDINGS_PATH", tmp_path / "nothing.npz")
        monkeypatch.setattr(embeddings, "get_client",
                            lambda: SimpleNamespace(embed=_fake_embed_factory()))
        with pytest.raises(FileNotFoundError):
            embeddings.search("anything")


# ── Batching ──────────────────────────────────────────────────────────────────

class TestBatching:
    def test_embeds_in_batches_of_size_limit(self, monkeypatch):
        from research_library import embeddings
        monkeypatch.setattr(embeddings, "BATCH_SIZE", 2)

        calls = []

        def spy_embed(texts, model=None, input_type=None):
            calls.append(len(texts))
            return SimpleNamespace(
                embeddings=[[0.1] * 4 for _ in texts]
            )
        monkeypatch.setattr(embeddings, "get_client",
                            lambda: SimpleNamespace(embed=spy_embed))

        result = embeddings._embed_all(["a", "b", "c", "d", "e"], input_type="document")
        assert calls == [2, 2, 1]
        assert result.shape == (5, 4)


# ── search_by_category ────────────────────────────────────────────────────────

class TestSearchByCategory:
    def test_ranks_categories_then_papers(self, tmp_path, fake_index, monkeypatch):
        from research_library import embeddings
        monkeypatch.setattr(embeddings, "EMBEDDINGS_PATH", tmp_path / "emb.npz")
        monkeypatch.setattr(embeddings, "INDEX_CACHE_PATH", fake_index)
        monkeypatch.setattr(embeddings, "get_client",
                            lambda: SimpleNamespace(embed=_fake_embed_factory()))

        embeddings.build_embeddings()
        out = embeddings.search_by_category("Alpha paper\n\nabout alpha",
                                            top_k_cats=1, top_k_papers=5)
        assert len(out["categories"]) == 1
        # All returned papers should belong to the winning category.
        winning_cat = out["categories"][0]["name"]
        assert all(p["category"] == winning_cat for p in out["papers"])
