"""Tests for discovery scoring: embedding, per-topic fit and the balanced shortlist."""
import numpy as np

from research_library.discovery import scoring


def _lib():
    # 4-dim unit vectors; Big has 3 papers near e1, Small has 1 paper at e2
    v = np.array([[1, 0, 0, 0], [.9, .1, 0, 0], [.8, .2, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return {"keys": ["B1", "B2", "B3", "S1"], "vectors": v, "paper_cats": ["Big", "Big", "Big", "Small"],
            "cat_names": ["Big", "Small"], "cat_vectors": v[:2]}


def _c(title, abstract="abs"):
    return {"title": title, "abstract": abstract, "id": title}


def test_embed_candidates_uses_paper_text_and_normalises():
    seen = []

    def fake_embed(texts):
        seen.extend(texts)
        return np.array([[3.0, 4.0, 0, 0]] * len(texts), dtype=np.float32)

    vecs = scoring.embed_candidates([_c("T", "A")], embed=fake_embed)
    assert seen == ["T\n\nA"] and np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


def test_score_assigns_small_topic_by_per_topic_fit():
    cands = [_c("near e2"), _c("near e1")]
    vecs = np.array([[0.1, 1, 0, 0], [1, 0.05, 0, 0]], dtype=np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    scoring.score_candidates(cands, vecs, _lib(), followed=["Big", "Small"])
    assert cands[0]["topic"] == "Small" and cands[1]["topic"] == "Big"
    assert cands[0]["nearest"][0]["key"] == "S1" and len(cands[0]["nearest"]) == 3
    assert 0 < cands[1]["fit"] <= 1 and len(cands[1]["vector"]) == 4


def test_score_ignores_unfollowed_topics():
    cands = [_c("near e2")]
    vecs = np.array([[0, 1, 0, 0]], dtype=np.float32)
    scoring.score_candidates(cands, vecs, _lib(), followed=["Big"])
    assert cands[0]["topic"] == "Big"
    scoring.score_candidates(cands, vecs, _lib(), followed=[])
    assert cands[0]["topic"] is None


def test_select_applies_floors_per_topic_cap_and_round_robin():
    follow = {"min_fit": 0.6, "min_fit_title_only": 0.5, "per_topic": 2, "max_total": 3}
    cands = [
        {"id": "a", "topic": "Big", "fit": 0.9, "abstract": "x"},
        {"id": "b", "topic": "Big", "fit": 0.8, "abstract": "x"},
        {"id": "c", "topic": "Big", "fit": 0.7, "abstract": "x"},    # cut by per_topic
        {"id": "d", "topic": "Small", "fit": 0.55, "abstract": ""},  # title-only, above its floor
        {"id": "e", "topic": "Small", "fit": 0.55, "abstract": "x"},  # below min_fit
        {"id": "f", "topic": None, "fit": 0.99, "abstract": "x"},
    ]
    assert [c["id"] for c in scoring.select_candidates(cands, follow)] == ["a", "d", "b"]
