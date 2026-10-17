"""Score candidates against the library and pick a balanced shortlist.

Per-topic fit is the mean cosine of a candidate's three nearest papers *within*
a topic. A candidate goes to the followed topic where it fits best. Voting over
global neighbours would hand nearly everything to the two topics that hold 38%
of the library, so small topics would never see a new paper.
"""
import numpy as np

from .. import embeddings


def embed_candidates(cands: list[dict], embed=None) -> np.ndarray:
    """Unit vectors for each candidate, embedded the same way as library papers."""
    if embed is None:
        embed = lambda texts: embeddings._embed_all(texts, input_type="document")  # noqa: E731
    texts = [embeddings._paper_text(c) for c in cands]
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    return embeddings._normalize(np.asarray(embed(texts), dtype=np.float32))


def score_candidates(cands: list[dict], vectors: np.ndarray, library: dict, followed: list[str]) -> None:
    """Set topic, fit, nearest and vector on each candidate, in place."""
    if not cands:
        return
    lib_vectors = np.asarray(library["vectors"], dtype=np.float32)
    cats = np.asarray(library["paper_cats"])
    sims = vectors @ lib_vectors.T  # (n candidates, m library papers)

    topic_cols = {t: np.flatnonzero(cats == t) for t in followed}
    topic_cols = {t: idx for t, idx in topic_cols.items() if len(idx)}

    for i, cand in enumerate(cands):
        row = sims[i]
        top = np.argsort(-row)[:3]
        cand["nearest"] = [{"key": library["keys"][j], "score": round(float(row[j]), 4), "category": str(cats[j])}
                           for j in top]
        cand["vector"] = [round(float(x), 6) for x in vectors[i]]  # 6 dp keeps digests small
        best_topic, best_fit = None, -1.0
        for topic, idx in topic_cols.items():
            fit = float(np.sort(row[idx])[-3:].mean())
            if fit > best_fit:
                best_topic, best_fit = topic, fit
        cand["topic"] = best_topic
        cand["fit"] = round(best_fit, 4) if best_topic else 0.0


def fit_floor(cand: dict, follow: dict) -> float:
    """Title-only vectors sit closer to everything, so they get their own, higher floor."""
    if cand.get("abstract"):
        return float(follow.get("min_fit", 0.50))
    return float(follow.get("min_fit_title_only", 0.76))


def select_candidates(cands: list[dict], follow: dict) -> list[dict]:
    """Apply the fit floors and the per-topic cap, then fill slots round-robin across topics.

    Within a topic candidates rank by their margin over their own floor, which puts a
    paper with an abstract at 0.60 ahead of a title-only one at 0.80.
    """
    per_topic = int(follow.get("per_topic", 3))
    max_total = int(follow.get("max_total", 15))
    by_topic: dict[str, list[dict]] = {}
    margins: dict[str, float] = {}
    for c in cands:
        if not c.get("topic"):
            continue
        margin = float(c.get("fit", 0.0)) - fit_floor(c, follow)
        if margin < 0:
            continue
        margins[c["id"]] = margin
        by_topic.setdefault(c["topic"], []).append(c)
    queues = []
    for topic, items in by_topic.items():
        items.sort(key=lambda c: -margins[c["id"]])
        queues.append(items[:per_topic])
    queues.sort(key=lambda q: -margins[q[0]["id"]])  # topics ordered by their best candidate
    picked: list[dict] = []
    rank = 0
    while len(picked) < max_total and any(len(q) > rank for q in queues):
        for q in queues:
            if len(q) > rank and len(picked) < max_total:
                picked.append(q[rank])
        rank += 1
    return picked
