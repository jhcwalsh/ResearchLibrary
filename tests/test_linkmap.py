import numpy as np
import pytest

from research_library.linkmap import build_link_graph, nearest_papers


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def library():
    # Two tight pairs (A,B) and (C,D) in different topics, plus a loner E.
    index = {
        "categories": {
            "Alpha": {"summary": "", "subcategories": {}, "papers": [
                {"key": "A", "title": "Paper <i>A</i>", "authors": ["Smith, J"], "year": "2020",
                 "doi": "10.1/a", "abstract": ""},
                {"key": "B", "title": "Paper B", "authors": ["Jones, K", "Lee, M"], "year": "2021",
                 "doi": "", "abstract": ""},
            ]},
            "Beta": {"summary": "", "subcategories": {}, "papers": [
                {"key": "C", "title": "Paper C", "authors": [], "year": "", "doi": "", "abstract": ""},
                {"key": "D", "title": "Paper D", "authors": ["Wu, X"], "year": "2019", "doi": "", "abstract": ""},
                {"key": "E", "title": "Paper E", "authors": ["Ng, Y"], "year": "2018", "doi": "", "abstract": ""},
            ]},
        }
    }
    emb = {
        "keys": ["A", "B", "C", "D", "E", "ZOMBIE"],
        "vectors": np.vstack([
            _unit([1, 0.1, 0]), _unit([1, 0.2, 0]),
            _unit([0, 1, 0.1]), _unit([0, 1, 0.2]),
            _unit([0, 0, 1]),
            _unit([1, 1, 1]),  # in the embeddings but not the index: must be ignored
        ]),
        "paper_cats": ["Alpha", "Alpha", "Beta", "Beta", "Beta", "Alpha"],
    }
    return index, emb


def test_graph_links_similar_papers_once_each_way(library):
    index, emb = library
    g = build_link_graph(index, emb, neighbours=1, min_similarity=0.9)
    pairs = {frozenset((e["source"], e["target"])) for e in g["edges"]}
    assert pairs == {frozenset("AB"), frozenset("CD")}
    assert len(g["edges"]) == 2  # A->B and B->A collapse to one line
    assert {n["key"] for n in g["nodes"]} == {"A", "B", "C", "D", "E"}
    assert g["stats"] == {"papers": 5, "links": 2, "unlinked": 1}


def test_node_payload(library):
    index, emb = library
    g = build_link_graph(index, emb, neighbours=1, min_similarity=0.9)
    a = next(n for n in g["nodes"] if n["key"] == "A")
    assert a["title"] == "Paper A"  # html stripped
    assert a["cite"] == "Smith 2020"
    assert a["topic"] == "Alpha"
    assert a["url"] == "https://doi.org/10.1/a"
    assert a["degree"] == 1
    e = next(n for n in g["nodes"] if n["key"] == "E")
    assert e["degree"] == 0
    edge = next(e for e in g["edges"] if {e["source"], e["target"]} == {"A", "B"})
    assert 0.99 < edge["weight"] <= 1.0


def test_topic_scope_and_floor(library):
    index, emb = library
    g = build_link_graph(index, emb, topic="Beta", neighbours=3, min_similarity=0.0)
    assert {n["key"] for n in g["nodes"]} == {"C", "D", "E"}
    assert len(g["edges"]) == 3  # every pair, floor is zero
    g2 = build_link_graph(index, emb, topic="Beta", neighbours=3, min_similarity=0.5)
    assert {frozenset((e["source"], e["target"])) for e in g2["edges"]} == {frozenset("CD")}


def test_missing_topic_gives_empty_graph(library):
    index, emb = library
    g = build_link_graph(index, emb, topic="Nope")
    assert g == {"nodes": [], "edges": [], "stats": {"papers": 0, "links": 0, "unlinked": 0}}


def test_nearest_papers_ranks_and_skips_self(library):
    index, emb = library
    near = nearest_papers("A", index, emb, n=2)
    assert len(near) == 2
    assert near[0]["key"] == "B"
    assert near[0]["topic"] == "Alpha"
    assert near[0]["paper"]["title"] == "Paper B"
    assert near[0]["score"] > near[1]["score"]
    assert "ZOMBIE" not in {x["key"] for x in nearest_papers("A", index, emb, n=10)}
    assert nearest_papers("UNKNOWN", index, emb) == []
