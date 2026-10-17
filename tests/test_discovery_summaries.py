"""Tests for the Haiku calls in discovery: per-topic search phrases and candidate summaries."""
import json
from types import SimpleNamespace

from research_library.contents import topic_fingerprint
from research_library.discovery import summaries


def _client(payload, sent):
    msg = SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking="", signature="s"),
                                   SimpleNamespace(type="text", text=json.dumps(payload))],
                          stop_reason="end_turn", stop_details=None)

    def create(**kw):
        sent.append(kw)
        return msg

    return SimpleNamespace(messages=SimpleNamespace(create=create))


def _cat(keys):
    return {"summary": "s", "papers": [{"key": k, "title": f"Title {k}", "authors": ["Doe, J"], "year": "2020", "abstract": "a"} for k in keys],
            "subcategories": {"Sub": list(keys)}}


def test_generate_queries_uses_haiku_schema_and_caps_at_six():
    sent = []
    out = summaries.generate_queries(_client({"queries": [f"q{i}" for i in range(8)]}, sent), "Private Equity", _cat(["A", "B"]))
    assert out == [f"q{i}" for i in range(6)]
    assert sent[0]["model"] == "claude-haiku-5-5" and sent[0]["output_config"]["effort"] == "low"
    assert sent[0]["output_config"]["format"]["type"] == "json_schema" and "Title A" in sent[0]["messages"][0]["content"]


def test_ensure_queries_caches_by_fingerprint():
    sent = []
    cats = {"PE": _cat(["A"]), "Risk": _cat(["B"])}
    follow = {"queries": {"PE": {"fingerprint": topic_fingerprint(cats["PE"]), "queries": ["cached"]}}}
    out = summaries.ensure_queries(_client({"queries": ["fresh", "two", "three", "four"]}, sent), follow, cats, ["PE", "Risk"])
    assert out == {"PE": ["cached"], "Risk": ["fresh", "two", "three", "four"]} and len(sent) == 1
    assert follow["queries"]["Risk"]["fingerprint"] == topic_fingerprint(cats["Risk"])
    summaries.ensure_queries(_client({"queries": ["x", "y", "z", "w"]}, sent), follow, cats, ["PE"], force=True)
    assert follow["queries"]["PE"]["queries"] == ["x", "y", "z", "w"]


def test_summarize_candidate_with_and_without_abstract():
    sent = []
    payload = {"summary": "S", "why_it_fits": "Extends Doe 2020.", "recommendation": "keep", "reason": "R"}
    cand = {"title": "T", "authors": ["Doe, J"], "venue": "arXiv q-fin.PM", "published": "2026-10-15", "abstract": "The abstract.", "extra": {}}
    out = summaries.summarize_candidate(_client(payload, sent), cand, [{"key": "A", "title": "Title A", "authors": ["Doe, J"], "year": "2020", "abstract": "a"}])
    assert out == {"summary": "S", "why_it_fits": "Extends Doe 2020.", "recommendation": "keep", "reason": "R"}
    assert "The abstract." in sent[0]["messages"][0]["content"] and "Doe 2020" in sent[0]["messages"][0]["content"]
    cand2 = {**cand, "abstract": "", "extra": {"affiliations": ["LBS"], "pages": 52, "downloads": 11}}
    out2 = summaries.summarize_candidate(_client(payload, sent), cand2, [])
    assert out2["summary"] is None and "52" in sent[1]["messages"][0]["content"] and "no abstract" in sent[1]["messages"][0]["content"].lower()
