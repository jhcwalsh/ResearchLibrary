"""
Semantic search over the Zotero library via Voyage AI embeddings.

Embeds every paper (title + abstract) and every category (name + summary),
caches vectors to .cache/embeddings.npz, and provides cosine-similarity
search used by the semantic_search and ask_library MCP tools.
"""
import json
import os
from pathlib import Path

import numpy as np
import voyageai
from dotenv import load_dotenv

from .analysis import CACHE_PATH as INDEX_CACHE_PATH

load_dotenv(override=True)

EMBEDDINGS_PATH = INDEX_CACHE_PATH.parent / "embeddings.npz"

# voyage-3 supports 128 inputs per batch and 16k tokens per input
VOYAGE_MODEL = "voyage-3"
BATCH_SIZE = 128

_client: voyageai.Client | None = None


def get_client() -> voyageai.Client:
    global _client
    if _client is None:
        _client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    return _client


# ── Embedding ─────────────────────────────────────────────────────────────────

def _paper_text(paper: dict) -> str:
    parts = [paper.get("title") or ""]
    if paper.get("abstract"):
        parts.append(paper["abstract"])
    return "\n\n".join(p for p in parts if p).strip() or "(no content)"


def _category_text(name: str, summary: str) -> str:
    return f"{name}\n\n{summary}".strip() or name


def _embed_batch(texts: list[str], input_type: str) -> np.ndarray:
    client = get_client()
    result = client.embed(texts, model=VOYAGE_MODEL, input_type=input_type)
    return np.asarray(result.embeddings, dtype=np.float32)


def _embed_all(texts: list[str], input_type: str, progress=None) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        if progress:
            progress(f"Embedding {i + len(batch)}/{len(texts)}...")
        vectors.append(_embed_batch(batch, input_type=input_type))
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(vectors)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


# ── Build + cache ─────────────────────────────────────────────────────────────

def build_embeddings(progress=None) -> dict:
    """
    Read .cache/library_index.json, embed every paper and every category,
    write .cache/embeddings.npz.
    """
    def step(msg):
        if progress:
            progress(msg)

    if not INDEX_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"No library index at {INDEX_CACHE_PATH}. Run `research-admin refresh` first."
        )

    index = json.loads(INDEX_CACHE_PATH.read_text(encoding="utf-8"))
    categories = index["categories"]

    # Flatten papers, remembering which category they belong to.
    paper_keys: list[str] = []
    paper_texts: list[str] = []
    paper_cats: list[str] = []
    for cat_name, cat_data in categories.items():
        for p in cat_data["papers"]:
            paper_keys.append(p["key"])
            paper_texts.append(_paper_text(p))
            paper_cats.append(cat_name)

    step(f"Embedding {len(paper_texts)} papers...")
    paper_vectors = _normalize(_embed_all(paper_texts, input_type="document", progress=progress))

    cat_names = list(categories.keys())
    cat_texts = [_category_text(n, categories[n].get("summary", "")) for n in cat_names]
    step(f"Embedding {len(cat_texts)} categories...")
    cat_vectors = _normalize(_embed_all(cat_texts, input_type="document", progress=progress))

    EMBEDDINGS_PATH.parent.mkdir(exist_ok=True)
    np.savez(
        EMBEDDINGS_PATH,
        keys=np.array(paper_keys),
        vectors=paper_vectors,
        paper_cats=np.array(paper_cats),
        cat_names=np.array(cat_names),
        cat_vectors=cat_vectors,
    )
    step(f"Saved embeddings for {len(paper_keys)} papers and {len(cat_names)} categories.")
    return {
        "paper_count": len(paper_keys),
        "category_count": len(cat_names),
        "path": str(EMBEDDINGS_PATH),
    }


def load_embeddings() -> dict | None:
    if not EMBEDDINGS_PATH.exists():
        return None
    data = np.load(EMBEDDINGS_PATH, allow_pickle=False)
    return {
        "keys": data["keys"].tolist(),
        "vectors": data["vectors"],
        "paper_cats": data["paper_cats"].tolist(),
        "cat_names": data["cat_names"].tolist(),
        "cat_vectors": data["cat_vectors"],
    }


# ── Search ────────────────────────────────────────────────────────────────────

def _embed_query(query: str) -> np.ndarray:
    vec = _embed_batch([query], input_type="query")[0]
    return vec / (np.linalg.norm(vec) or 1.0)


def search(query: str, top_k: int = 20) -> list[dict]:
    """Return top-k papers by cosine similarity to query. Each result has
    {key, score, category}."""
    cache = load_embeddings()
    if cache is None:
        raise FileNotFoundError(
            f"No embeddings cache at {EMBEDDINGS_PATH}. Run `research-admin embed`."
        )
    q = _embed_query(query)
    scores = cache["vectors"] @ q
    top_k = min(top_k, len(scores))
    idx = np.argpartition(-scores, top_k - 1)[:top_k]
    idx = idx[np.argsort(-scores[idx])]
    return [
        {
            "key": cache["keys"][i],
            "score": float(scores[i]),
            "category": cache["paper_cats"][i],
        }
        for i in idx
    ]


def search_by_category(
    query: str, top_k_cats: int = 3, top_k_papers: int = 10
) -> dict:
    """
    Rank categories by query similarity, then return the top papers drawn from
    those categories. Returns {categories: [...], papers: [...]}.
    """
    cache = load_embeddings()
    if cache is None:
        raise FileNotFoundError(
            f"No embeddings cache at {EMBEDDINGS_PATH}. Run `research-admin embed`."
        )
    q = _embed_query(query)

    cat_scores = cache["cat_vectors"] @ q
    top_k_cats = min(top_k_cats, len(cat_scores))
    cat_idx = np.argpartition(-cat_scores, top_k_cats - 1)[:top_k_cats]
    cat_idx = cat_idx[np.argsort(-cat_scores[cat_idx])]
    top_cats = [
        {"name": cache["cat_names"][i], "score": float(cat_scores[i])}
        for i in cat_idx
    ]
    top_cat_names = {c["name"] for c in top_cats}

    paper_scores = cache["vectors"] @ q
    mask = np.array([c in top_cat_names for c in cache["paper_cats"]])
    if not mask.any():
        return {"categories": top_cats, "papers": []}

    masked_scores = np.where(mask, paper_scores, -np.inf)
    top_k_papers = min(top_k_papers, int(mask.sum()))
    paper_idx = np.argpartition(-masked_scores, top_k_papers - 1)[:top_k_papers]
    paper_idx = paper_idx[np.argsort(-masked_scores[paper_idx])]
    papers = [
        {
            "key": cache["keys"][i],
            "score": float(paper_scores[i]),
            "category": cache["paper_cats"][i],
        }
        for i in paper_idx
    ]
    return {"categories": top_cats, "papers": papers}
