"""Cross-paper linkage map: a nearest-neighbour graph over the paper embeddings.

Edges mean "similar content" (cosine similarity of the voyage-finance-2 vectors), not
citation. Everything here is a pure function over the two caches, so the Streamlit page
and any future MCP tool share one answer.
"""
import re

import numpy as np

from .contents import short_cite
from .links import _extract_year, paper_link

DEFAULT_NEIGHBOURS = 4
DEFAULT_MIN_SIMILARITY = 0.55


def _clean_title(title: str) -> str:
    return re.sub(r"<[^>]+>", "", title or "").strip()


def _papers_by_key(index: dict) -> dict[str, tuple[str, dict]]:
    return {p["key"]: (cat, p)
            for cat, data in index.get("categories", {}).items()
            for p in data.get("papers", [])}


def _select(index: dict, emb: dict, topic: str | None):
    """Rows of the embedding arrays for papers that are in the index (and topic)."""
    by_key = _papers_by_key(index)
    rows = [i for i, k in enumerate(emb["keys"])
            if k in by_key and (topic is None or by_key[k][0] == topic)]
    return by_key, rows


def _unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def build_link_graph(index: dict, emb: dict, topic: str | None = None,
                     neighbours: int = DEFAULT_NEIGHBOURS,
                     min_similarity: float = DEFAULT_MIN_SIMILARITY) -> dict:
    """Join each paper to its `neighbours` most similar papers scoring at least
    `min_similarity`. Returns {nodes, edges, stats}; edges are undirected and unique."""
    by_key, rows = _select(index, emb, topic)
    if not rows:
        return {"nodes": [], "edges": [], "stats": {"papers": 0, "links": 0, "unlinked": 0}}

    keys = [emb["keys"][i] for i in rows]
    vectors = _unit(np.asarray(emb["vectors"], dtype=np.float32)[rows])
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -1.0)

    k = max(0, min(int(neighbours), len(rows) - 1))
    edges: dict[tuple[int, int], float] = {}
    if k:
        for a in range(len(rows)):
            row = sims[a]
            top = np.argpartition(-row, k - 1)[:k]
            for b in top:
                score = float(row[b])
                if score < min_similarity:
                    continue
                pair = (min(a, int(b)), max(a, int(b)))
                edges[pair] = max(score, edges.get(pair, -1.0))

    degree = [0] * len(rows)
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1

    nodes = []
    for i, key in enumerate(keys):
        cat, p = by_key[key]
        nodes.append({
            "key": key,
            "title": _clean_title(p.get("title")),
            "cite": short_cite(p),
            "year": _extract_year(p.get("year") or ""),
            "topic": cat,
            "url": paper_link(p),
            "degree": degree[i],
        })
    edge_list = [{"source": keys[a], "target": keys[b], "weight": round(w, 3)}
                 for (a, b), w in sorted(edges.items(), key=lambda kv: -kv[1])]
    return {
        "nodes": nodes,
        "edges": edge_list,
        "stats": {"papers": len(nodes), "links": len(edge_list),
                  "unlinked": sum(1 for d in degree if d == 0)},
    }


def nearest_papers(key: str, index: dict, emb: dict, n: int = 10) -> list[dict]:
    """The `n` papers most similar to `key`, best first: [{key, score, topic, paper}]."""
    by_key, rows = _select(index, emb, None)
    position = {emb["keys"][i]: j for j, i in enumerate(rows)}
    if key not in position:
        return []
    vectors = _unit(np.asarray(emb["vectors"], dtype=np.float32)[rows])
    scores = vectors @ vectors[position[key]]
    out = []
    for j in np.argsort(-scores):
        other = emb["keys"][rows[j]]
        if other == key:
            continue
        cat, p = by_key[other]
        out.append({"key": other, "score": round(float(scores[j]), 4), "topic": cat, "paper": p})
        if len(out) >= n:
            break
    return out
