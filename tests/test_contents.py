"""Tests for the Contents page: cited topic summaries and their rendering."""
import json
from types import SimpleNamespace

from research_library import contents
from research_library.links import paper_link


def _paper(key, title="A Paper", authors=None, year="2020", doi="", url="", abstract="abs"):
    return {"key": key, "title": title, "authors": authors if authors is not None else ["Smith, Ann"],
            "year": year, "doi": doi, "url": url, "abstract": abstract}


PAPERS = {
    "AAAA1111": _paper("AAAA1111", "Expected Returns", ["Ilmanen, Antti", "Chandra, S", "McQuinn, N"],
                       "2020", doi="10.1/x"),
    "BBBB2222": _paper("BBBB2222", "Liquidity", ["Nadauld, Taylor", "Sensoy, Berk"], "2019",
                       url="https://ex.org/p"),
    "CCCC3333": _paper("CCCC3333", "Cash Flows", ["Robinson, David"], ""),
}


# ── Links and citations ────────────────────────────────────────────────────────

def test_paper_link_prefers_doi_then_url_then_scholar():
    assert paper_link(PAPERS["AAAA1111"]) == "https://doi.org/10.1/x"
    assert paper_link(PAPERS["BBBB2222"]) == "https://ex.org/p"
    assert paper_link(PAPERS["CCCC3333"]).startswith("https://scholar.google.com/scholar?q=")


def test_short_cite_by_author_count():
    assert contents.short_cite(PAPERS["AAAA1111"]) == "Ilmanen et al. 2020"
    assert contents.short_cite(PAPERS["BBBB2222"]) == "Nadauld & Sensoy 2019"
    assert contents.short_cite(PAPERS["CCCC3333"]) == "Robinson n.d."
    assert contents.short_cite(_paper("X", "Untitled Report On Things", [], "2021")) == "Untitled Report On Things 2021"


def test_render_citations_links_each_key():
    out = contents.render_citations("Returns are modest [AAAA1111].", PAPERS)
    assert out == "Returns are modest ([Ilmanen et al. 2020](https://doi.org/10.1/x))."


def test_render_citations_handles_grouped_keys():
    out = contents.render_citations("Premia exist [AAAA1111, BBBB2222].", PAPERS)
    assert "[Ilmanen et al. 2020](https://doi.org/10.1/x); [Nadauld & Sensoy 2019](https://ex.org/p)" in out


def test_render_citations_drops_unknown_keys_and_leaves_other_brackets():
    out = contents.render_citations("Claim [ZZZZ9999]. See [note] and [AAAA1111, ZZZZ9999].", PAPERS)
    assert "ZZZZ9999" not in out
    assert "Claim." in out
    assert "[note]" in out
    assert "([Ilmanen et al. 2020](https://doi.org/10.1/x))" in out


# ── Writing summaries ──────────────────────────────────────────────────────────

def _category(keys, subs=None):
    return {"summary": "short", "papers": [PAPERS[k] for k in keys],
            "subcategories": subs or {"Sub A": list(keys)}}


def _fake_client(payload, sent=None):
    msg = SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking="", signature="s"),
                 SimpleNamespace(type="text", text=json.dumps(payload))],
        stop_reason="end_turn", stop_details=None)

    def create(**kwargs):
        if sent is not None:
            sent.append(kwargs)
        return msg
    return SimpleNamespace(messages=SimpleNamespace(create=create))


def test_summarize_topic_uses_sonnet_with_a_schema_and_filters_unknown_keys():
    sent = []
    payload = {"overview": "Overview [AAAA1111].",
               "subtopics": [{"name": "Sub A", "summary": "About A [BBBB2222]."}],
               "key_papers": ["AAAA1111", "NOTAKEY1"]}
    out = contents.summarize_topic(_fake_client(payload, sent), "Private Equity",
                                   _category(["AAAA1111", "BBBB2222"]))
    assert sent[0]["model"] == "claude-sonnet-5-5"
    assert sent[0]["output_config"]["format"]["type"] == "json_schema"
    assert "AAAA1111" in sent[0]["messages"][0]["content"]
    assert out["overview"] == "Overview [AAAA1111]."
    assert out["key_papers"] == ["AAAA1111"]
    assert out["subtopics"] == {"Sub A": "About A [BBBB2222]."}


def _write_index(tmp_path, categories):
    path = tmp_path / "library_index.json"
    path.write_text(json.dumps({"generated_at": "2026-10-13T00:00:00", "unique_papers": 3,
                                "categories": categories}), encoding="utf-8")
    return path


def test_build_skips_unchanged_topics_rewrites_changed_and_drops_removed(tmp_path, monkeypatch):
    index_path = _write_index(tmp_path, {
        "Same": _category(["AAAA1111"]),
        "Changed": _category(["BBBB2222", "CCCC3333"]),
    })
    out_path = tmp_path / "topic_summaries.json"
    same_fp = contents.topic_fingerprint(_category(["AAAA1111"]))
    out_path.write_text(json.dumps({"topics": {
        "Same": {"overview": "kept", "fingerprint": same_fp},
        "Changed": {"overview": "old", "fingerprint": "stale"},
        "Removed": {"overview": "gone", "fingerprint": "x"},
    }}), encoding="utf-8")

    written = []

    def fake_summarize(client, name, cat):
        written.append(name)
        return {"overview": f"new {name}", "subtopics": {}, "key_papers": []}

    monkeypatch.setattr(contents, "summarize_topic", fake_summarize)
    result = contents.build_topic_summaries(index_path=index_path, out_path=out_path, client=object())

    saved = json.loads(out_path.read_text(encoding="utf-8"))["topics"]
    assert written == ["Changed"]
    assert saved["Same"]["overview"] == "kept"
    assert saved["Changed"]["overview"] == "new Changed"
    assert saved["Changed"]["fingerprint"] == contents.topic_fingerprint(_category(["BBBB2222", "CCCC3333"]))
    assert "Removed" not in saved
    assert result == {"written": 1, "skipped": 1, "failed": []}


def test_build_records_a_failed_topic_and_keeps_going(tmp_path, monkeypatch):
    index_path = _write_index(tmp_path, {"Bad": _category(["AAAA1111"]),
                                         "Good": _category(["BBBB2222"])})
    out_path = tmp_path / "topic_summaries.json"

    def fake_summarize(client, name, cat):
        if name == "Bad":
            raise RuntimeError("declined")
        return {"overview": "ok", "subtopics": {}, "key_papers": []}

    monkeypatch.setattr(contents, "summarize_topic", fake_summarize)
    result = contents.build_topic_summaries(index_path=index_path, out_path=out_path, client=object())

    saved = json.loads(out_path.read_text(encoding="utf-8"))["topics"]
    assert result["failed"] == ["Bad"]
    assert "Good" in saved and "Bad" not in saved


def test_force_rewrites_everything(tmp_path, monkeypatch):
    index_path = _write_index(tmp_path, {"Same": _category(["AAAA1111"])})
    out_path = tmp_path / "topic_summaries.json"
    out_path.write_text(json.dumps({"topics": {"Same": {
        "overview": "kept", "fingerprint": contents.topic_fingerprint(_category(["AAAA1111"]))}}}),
        encoding="utf-8")
    monkeypatch.setattr(contents, "summarize_topic",
                        lambda c, n, cat: {"overview": "fresh", "subtopics": {}, "key_papers": []})
    contents.build_topic_summaries(index_path=index_path, out_path=out_path, client=object(), force=True)
    assert json.loads(out_path.read_text(encoding="utf-8"))["topics"]["Same"]["overview"] == "fresh"
